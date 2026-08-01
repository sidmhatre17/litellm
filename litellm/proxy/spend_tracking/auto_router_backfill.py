"""One-shot replay of historical spend logs into the auto-router session rollup.

The rollup only sees traffic written after it ships, so without this the
dashboard reads empty until enough new sessions accumulate. This is the only
place that still reads ``LiteLLM_SpendLogs``; the dashboard's own path never
does.

Correctness comes from replaying through the same ``fold_turn`` the live writer
uses, in the order the turns happened, so a backfilled session and a live one are
folded by identical code rather than by a SQL transliteration of it.

Historical rows predate the routing metadata that carries the counterfactual
baseline, so the baseline is taken from the router as configured now. That prices
old traffic against today's flagship, which is the honest reading of "what would
this have cost on one model" for a deployment whose ladder has since changed.
"""

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy.spend_tracking.auto_router_benchmarks import clamp_window
from litellm.proxy.spend_tracking.auto_router_session_queue import SessionKey
from litellm.proxy.spend_tracking.auto_router_sessions import (
    EMPTY_SESSION_STATE,
    SessionState,
    TurnDelta,
    as_epoch,
    fold_turn,
    merge_deltas,
    state_column,
    turn_from_spend_payload,
)
from litellm.proxy.spend_tracking.savings import compute_autorouter_savings, usage_from_spend_log
from litellm.repositories.table_repositories import AutoRouterSessionRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router

BACKFILL_MAX_ROWS = 500_000

_COUNTER_FIELDS = (
    "turns",
    "turns_with_usage",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "ephemeral_5m_tokens",
    "ephemeral_1h_tokens",
    "spend",
    "baseline_spend",
    "same_model_turns",
    "same_model_hits",
    "first_visit_turns",
    "first_visit_hits",
    "return_turns",
    "return_hits",
    "stale_return_misses",
    "savable_return_misses",
    "rescued_spend",
    "replay_spend",
)


class BackfillResult(BaseModel):
    start_date: str
    end_date: str
    rows_read: int
    sessions_written: int
    truncated: bool


_BACKFILL_SQL = """
SELECT
    session_id,
    model_group,
    model,
    custom_llm_provider,
    "startTime" AS started_at,
    COALESCE(prompt_tokens, 0)::bigint AS prompt_tokens,
    COALESCE(completion_tokens, 0)::bigint AS completion_tokens,
    COALESCE(total_tokens, 0)::bigint AS total_tokens,
    COALESCE(spend, 0.0) AS spend,
    COALESCE((metadata::jsonb)->'usage_object', '{}'::jsonb)::text AS usage_object
FROM "LiteLLM_SpendLogs"
WHERE model_group = ANY($1::text[])
  AND session_id IS NOT NULL
  AND model IS NOT NULL
  AND "startTime" >= ($2::timestamptz AT TIME ZONE 'UTC')
  AND "startTime" < (($3::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
ORDER BY session_id, model_group, "startTime"
LIMIT $4
"""


class _RawTurnRow(BaseModel):
    session_id: str
    model_group: str
    model: str
    custom_llm_provider: str | None
    started_at: datetime
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    spend: float
    usage_object: str


_TURN_ROWS = TypeAdapter(tuple[_RawTurnRow, ...])


class _Accumulated(NamedTuple):
    router_kind: str
    baseline_model: str | None
    state: SessionState
    delta: TurnDelta
    first_turn_at: float
    last_turn_at: float


def baseline_models(router: "Router") -> Mapping[str, str | None]:
    """The counterfactual baseline each auto-router measures itself against.

    Only semantic auto-routers carry one. The other strategy routers have no
    configured baseline, so their historical savings stay at zero rather than
    being invented against a model they may never have been able to pick.
    """
    return {  # mutable-ok: a JSON object is a dict by definition
        model_group: tagged[0].strategy.savings_baseline_model
        for model_group, tagged in (
            router.auto_routers or {}  # mutable-ok: empty fallback for an absent mapping
        ).items()  # mutable-ok: empty fallback for an absent mapping
        if tagged
    }


def _parse_usage(raw: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}  # mutable-ok: empty fallback for an absent mapping
    return parsed if isinstance(parsed, dict) else {}  # mutable-ok: empty fallback for an absent mapping


def _cache_tokens(usage_object: Mapping[str, object]) -> tuple[int, int]:
    """``(read, created)`` cache tokens, across the shapes providers report them in."""
    read = int(usage_object.get("cache_read_input_tokens") or 0)
    created = int(usage_object.get("cache_creation_input_tokens") or 0)
    if read or created:
        return read, created
    details = usage_object.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return read, created
    return (
        int(details.get("cached_tokens") or 0),
        int(details.get("cache_write_tokens") or details.get("cache_creation_tokens") or 0),
    )


def _savings(row: _RawTurnRow, baseline_model: str | None, usage_object: Mapping[str, object]) -> float:
    if baseline_model is None:
        return 0.0
    usage = usage_from_spend_log(usage_object)
    if usage is None:
        return 0.0
    return compute_autorouter_savings(
        baseline_model=baseline_model,
        selected_model=row.model,
        selected_provider=row.custom_llm_provider,
        usage=usage,
    )


def fold_rows(
    rows: tuple[_RawTurnRow, ...],
    group_kinds: Mapping[str, str],
    baselines: Mapping[str, str | None],
) -> Mapping[SessionKey, _Accumulated]:
    """Replay ordered spend rows into one accumulated row per session.

    The rows arrive ordered by session then time, so each session's turns fold in
    the sequence the caller actually sent them, which is the whole reason the
    cache buckets and warming economics come out the same as they would live.
    """
    accumulated: dict[SessionKey, _Accumulated] = {}  # mutable-ok: single-pass accumulator over an ordered scan
    for row in rows:
        key: SessionKey = (row.session_id, row.model_group)
        usage_object = _parse_usage(row.usage_object)
        cache_read, cache_created = _cache_tokens(usage_object)
        baseline_model = baselines.get(row.model_group)
        started_at = as_epoch(row.started_at)
        current = accumulated.get(key)
        delta = fold_turn(
            current.state if current is not None else EMPTY_SESSION_STATE,
            turn_from_spend_payload(
                model=row.model,
                started_at=row.started_at,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                total_tokens=row.total_tokens,
                spend=row.spend,
                autorouter_savings=_savings(row, baseline_model, usage_object),
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_created,
                usage_object=usage_object,
            ),
        )
        accumulated[key] = _Accumulated(
            router_kind=group_kinds.get(row.model_group, "auto_router"),
            baseline_model=baseline_model,
            state=delta.state,
            delta=delta if current is None else merge_deltas(current.delta, delta),
            first_turn_at=min(current.first_turn_at, started_at) if current is not None else started_at,
            last_turn_at=max(current.last_turn_at, started_at) if current is not None else started_at,
        )
    return accumulated


async def backfill_sessions(
    prisma_client: "PrismaClient",
    router: "Router",
    group_kinds: Mapping[str, str],
    start_date: str,
    end_date: str,
) -> BackfillResult:
    """Replay the window's auto-router spend logs into the session rollup.

    Rows are replaced rather than incremented, so running this twice leaves the
    same numbers instead of doubling them; that makes it safe to re-run after
    widening the window or fixing a baseline.
    """
    window = clamp_window(start_date, end_date)
    raw = await prisma_client.db.query_raw(
        _BACKFILL_SQL,
        list(group_kinds.keys()),  # mutable-ok: query_raw binds a list for the text[] parameter
        window.start,
        window.end,
        BACKFILL_MAX_ROWS,
    )
    rows = _TURN_ROWS.validate_python(raw)
    if len(rows) == BACKFILL_MAX_ROWS:
        verbose_proxy_logger.warning(
            "auto_router backfill: hit the %s row cap; sessions beyond it were not replayed. "
            "Re-run over a narrower window to cover them",
            BACKFILL_MAX_ROWS,
        )
    accumulated = fold_rows(rows, group_kinds, baseline_models(router))
    written = sum(
        [  # mutable-ok: a JSON object is a dict by definition
            await _replace(key, entry, prisma_client) for key, entry in accumulated.items()
        ]  # mutable-ok: a JSON object is a dict by definition
    )  # mutable-ok: await inside a comprehension needs a list, not a generator
    if written != len(accumulated):
        verbose_proxy_logger.error(
            "auto_router backfill: %d of %d sessions failed to write; see the errors above",
            len(accumulated) - written,
            len(accumulated),
        )
    return BackfillResult(
        start_date=window.start,
        end_date=window.end,
        rows_read=len(rows),
        sessions_written=written,
        truncated=len(rows) == BACKFILL_MAX_ROWS,
    )


async def _replace(key: SessionKey, entry: _Accumulated, prisma_client: "PrismaClient") -> bool:
    """Write one session, reporting whether it landed.

    The caller counts these rather than counting what it intended to write; a
    backfill that reported success while every upsert failed would be worse than
    one that failed outright.
    """
    session_id, model_group = key
    record = {  # mutable-ok: prisma's write API takes dict payloads
        "router_kind": entry.router_kind,
        "baseline_model": entry.baseline_model,
        "first_turn_at": datetime.fromtimestamp(entry.first_turn_at, tz=timezone.utc),
        "last_turn_at": datetime.fromtimestamp(entry.last_turn_at, tz=timezone.utc),
        "last_model": entry.state.last_model,
        "model_state": state_column(entry.state),
        **{  # mutable-ok: a JSON object is a dict by definition
            field: getattr(entry.delta, field) for field in _COUNTER_FIELDS
        },  # mutable-ok: spread into the prisma payload immediately below
    }
    try:
        await AutoRouterSessionRepository(prisma_client).table.upsert(
            where={  # mutable-ok: prisma's write API takes dict payloads
                "session_id_model_group": {  # mutable-ok: a JSON object is a dict by definition
                    "session_id": session_id,
                    "model_group": model_group,
                }
            },
            data={  # mutable-ok: prisma's write API takes dict payloads
                "create": {  # mutable-ok: prisma's write API takes dict payloads
                    "session_id": session_id,
                    "model_group": model_group,
                    **record,
                },
                "update": record,
            },
        )
    except Exception as e:  # noqa: BLE001  # one session must not abort the whole replay
        verbose_proxy_logger.exception("auto_router backfill: failed to write session %s (%s)", key, e)
        return False
    return True
