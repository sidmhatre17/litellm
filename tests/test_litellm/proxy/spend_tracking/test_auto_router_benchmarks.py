from datetime import datetime, timedelta, timezone

import pytest

from litellm.proxy.spend_tracking.auto_router_backfill import backfill_sessions
from litellm.proxy.spend_tracking.auto_router_benchmarks import (
    BENCHMARKS_MAX_WINDOW_DAYS,
    clamp_window,
    compute_benchmarks,
)

GROUP_KINDS = {"claude-auto": "semantic"}


class _FakeTable:
    """Records every upsert so a test can assert on what would be written."""

    def __init__(self):
        self.upserts = []

    async def upsert(self, where, data):
        self.upserts.append((where, data))


class _FakeDb:
    """Returns rows the way prisma really does: a list of plain dicts."""

    def __init__(self, rows, table=None):
        self._rows = rows
        self.queries = []
        self.litellm_autoroutersession = table or _FakeTable()

    async def query_raw(self, sql, *args):
        self.queries.append((sql, args))
        return self._rows


class _FakePrisma:
    def __init__(self, rows, table=None):
        self.db = _FakeDb(rows, table)


def group_row(**overrides):
    row = {
        "model_group": "claude-auto",
        "baseline_model": "anthropic/claude-opus-4-8",
        "sessions": 10,
        "turns": 100,
        "total_session_seconds": 36000.0,
        "total_tokens": 1_000_000,
        "actual_spend": 10.0,
        "baseline_spend": 100.0,
        "turns_with_usage": 100,
        "ephemeral_5m_tokens": 0,
        "ephemeral_1h_tokens": 5000,
        "same_model_turns": 60,
        "same_model_hits": 57,
        "first_visit_turns": 10,
        "first_visit_hits": 2,
        "return_turns": 30,
        "return_hits": 24,
        "stale_return_misses": 4,
        "savable_return_misses": 2,
        "rescued_spend": 6.76,
        "replay_spend": 3.91,
    }
    row.update(overrides)
    return row


async def benchmarks_for(**overrides):
    prisma = _FakePrisma([group_row(**overrides)])
    return await compute_benchmarks(prisma, GROUP_KINDS, "2026-07-02", "2026-08-01")


class TestWindowClamping:
    def test_a_wider_request_is_clamped_to_the_maximum_window(self):
        window = clamp_window("2020-01-01", "2026-08-01")
        expected = (datetime(2026, 8, 1, tzinfo=timezone.utc) - timedelta(days=BENCHMARKS_MAX_WINDOW_DAYS)).date()
        assert window.start == expected.isoformat()

    def test_a_narrower_request_is_served_as_asked(self):
        assert clamp_window("2026-07-25", "2026-08-01").start == "2026-07-25"

    def test_the_response_echoes_the_window_actually_served(self):
        window = clamp_window("2020-01-01", "2026-08-01")
        assert window.end == "2026-08-01"


@pytest.mark.asyncio
class TestSessionShape:
    async def test_turns_per_session_divides_turns_by_sessions(self):
        result = await benchmarks_for()
        assert result.groups[0].avg_turns_per_session == pytest.approx(10.0)

    async def test_session_length_averages_the_summed_durations(self):
        result = await benchmarks_for()
        assert result.groups[0].avg_session_length_seconds == pytest.approx(3600.0)

    async def test_tokens_per_session_divides_tokens_by_sessions(self):
        result = await benchmarks_for()
        assert result.groups[0].avg_tokens_per_session == pytest.approx(100_000.0)

    async def test_a_group_with_no_sessions_is_omitted_rather_than_zeroed(self):
        result = await benchmarks_for(sessions=0)
        assert result.groups == ()


@pytest.mark.asyncio
class TestSavings:
    async def test_savings_is_baseline_minus_actual(self):
        result = await benchmarks_for()
        assert result.groups[0].savings == pytest.approx(90.0)
        assert result.groups[0].savings_pct == pytest.approx(90.0)

    async def test_a_route_that_cost_more_than_the_baseline_reports_a_loss(self):
        """Signed on purpose: a cache-thrashing router must not read as zero."""
        result = await benchmarks_for(actual_spend=120.0, baseline_spend=100.0)
        assert result.groups[0].savings == pytest.approx(-20.0)
        assert result.groups[0].savings_pct == pytest.approx(-20.0)

    async def test_an_unpriced_baseline_reports_no_percentage_instead_of_dividing_by_zero(self):
        result = await benchmarks_for(baseline_spend=0.0)
        assert result.groups[0].savings_pct == 0.0


@pytest.mark.asyncio
class TestCacheBuckets:
    async def test_the_three_buckets_sum_to_the_reported_turn_count(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.same_model_turns + cache.first_visit_turns + cache.return_turns == cache.turns

    async def test_the_headline_rate_is_weighted_by_turns_not_an_average_of_buckets(self):
        """57+2+24 hits over 100 turns is 83%, not the 61% mean of the three rates."""
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.hit_rate_pct == pytest.approx(83.0)

    async def test_each_bucket_reports_its_own_hit_rate(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.same_model_hit_rate_pct == pytest.approx(95.0)
        assert cache.first_visit_hit_rate_pct == pytest.approx(20.0)
        assert cache.return_hit_rate_pct == pytest.approx(80.0)

    async def test_stale_share_is_measured_against_return_misses_only(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.stale_miss_share_pct == pytest.approx(100.0 * 4 / 6)

    async def test_savable_share_is_measured_against_every_miss(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.warming_savable_miss_pct == pytest.approx(100.0 * 2 / 17)

    async def test_cache_is_omitted_when_nothing_reported_usage(self):
        result = await benchmarks_for(turns_with_usage=0)
        assert result.groups[0].cache is None

    async def test_coverage_is_the_share_of_turns_that_reported_usage(self):
        cache = (await benchmarks_for(turns_with_usage=50)).groups[0].cache
        assert cache is not None
        assert cache.usage_coverage_pct == pytest.approx(50.0)


@pytest.mark.asyncio
class TestWarmingEstimate:
    async def test_net_is_rescued_less_replays(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.warming_net_spend == pytest.approx(6.76 - 3.91)

    async def test_break_even_follows_the_ttl_in_use(self):
        one_hour = (await benchmarks_for()).groups[0].cache
        five_min = (await benchmarks_for(ephemeral_1h_tokens=0, ephemeral_5m_tokens=5000)).groups[0].cache
        assert one_hour is not None and five_min is not None
        assert one_hour.ttl_seconds == 3600
        assert one_hour.warming_break_even_pct == 5.0
        assert five_min.ttl_seconds == 300
        assert five_min.warming_break_even_pct == 9.0

    async def test_no_ephemeral_evidence_reads_as_the_five_minute_tier(self):
        cache = (await benchmarks_for(ephemeral_1h_tokens=0, ephemeral_5m_tokens=0)).groups[0].cache
        assert cache is not None
        assert cache.ttl_seconds == 300


@pytest.mark.asyncio
class TestReadPathSource:
    async def test_the_dashboard_query_never_touches_the_spend_logs(self):
        prisma = _FakePrisma([group_row()])
        await compute_benchmarks(prisma, GROUP_KINDS, "2026-07-02", "2026-08-01")
        sql = prisma.db.queries[0][0]
        assert "LiteLLM_SpendLogs" not in sql
        assert "LiteLLM_AutoRouterSession" in sql

    async def test_one_query_covers_every_configured_auto_router(self):
        prisma = _FakePrisma([group_row(), group_row(model_group="claude-router-2")])
        result = await compute_benchmarks(
            prisma, {"claude-auto": "semantic", "claude-router-2": "complexity"}, "2026-07-02", "2026-08-01"
        )
        assert len(prisma.db.queries) == 1
        assert {g.model_group for g in result.groups} == {"claude-auto", "claude-router-2"}

    async def test_each_group_is_labelled_with_its_router_kind(self):
        prisma = _FakePrisma([group_row(model_group="claude-router-2")])
        result = await compute_benchmarks(prisma, {"claude-router-2": "complexity"}, "2026-07-02", "2026-08-01")
        assert result.groups[0].router_kind == "complexity"


def spend_row(session_id, model, started_at, read=0, created=0, spend=0.01):
    usage = f'{{"cache_read_input_tokens": {read}, "cache_creation_input_tokens": {created}}}'
    return {
        "session_id": session_id,
        "model_group": "claude-auto",
        "model": model,
        "custom_llm_provider": "anthropic",
        "started_at": started_at,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "spend": spend,
        "usage_object": usage,
    }


class _NoAutoRouters:
    auto_routers: dict = {}


@pytest.mark.asyncio
class TestBackfill:
    async def test_a_session_is_replayed_into_one_row_with_its_turns_folded(self):
        base = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        table = _FakeTable()
        prisma = _FakePrisma(
            [
                spend_row("s1", "anthropic/claude-haiku-4-5", base, created=5000),
                spend_row("s1", "anthropic/claude-haiku-4-5", base + timedelta(seconds=60), read=5000),
                spend_row("s1", "anthropic/claude-sonnet-4-5", base + timedelta(seconds=120), created=5000),
            ],
            table,
        )
        result = await backfill_sessions(prisma, _NoAutoRouters(), GROUP_KINDS, "2026-07-02", "2026-08-01")

        assert result.rows_read == 3
        assert result.sessions_written == 1
        assert len(table.upserts) == 1
        created = table.upserts[0][1]["create"]
        assert created["turns"] == 3
        assert created["same_model_turns"] + created["first_visit_turns"] + created["return_turns"] == 3
        assert created["same_model_turns"] == 1
        assert created["first_visit_turns"] == 2
        assert created["same_model_hits"] == 1

    async def test_separate_sessions_get_separate_rows(self):
        base = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        table = _FakeTable()
        prisma = _FakePrisma(
            [
                spend_row("s1", "anthropic/claude-haiku-4-5", base, created=5000),
                spend_row("s2", "anthropic/claude-haiku-4-5", base + timedelta(seconds=1), created=5000),
            ],
            table,
        )
        result = await backfill_sessions(prisma, _NoAutoRouters(), GROUP_KINDS, "2026-07-02", "2026-08-01")
        assert result.sessions_written == 2
        assert {u[0]["session_id_model_group"]["session_id"] for u in table.upserts} == {"s1", "s2"}

    async def test_first_and_last_turn_bound_the_session_window(self):
        base = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        table = _FakeTable()
        prisma = _FakePrisma(
            [
                spend_row("s1", "anthropic/claude-haiku-4-5", base),
                spend_row("s1", "anthropic/claude-haiku-4-5", base + timedelta(hours=2)),
            ],
            table,
        )
        await backfill_sessions(prisma, _NoAutoRouters(), GROUP_KINDS, "2026-07-02", "2026-08-01")
        created = table.upserts[0][1]["create"]
        assert (created["last_turn_at"] - created["first_turn_at"]).total_seconds() == pytest.approx(7200.0)

    async def test_replaying_the_same_window_twice_does_not_double_the_counters(self):
        """Backfill replaces rather than increments, so a re-run is idempotent."""
        base = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        table = _FakeTable()
        prisma = _FakePrisma([spend_row("s1", "anthropic/claude-haiku-4-5", base, created=5000)], table)
        await backfill_sessions(prisma, _NoAutoRouters(), GROUP_KINDS, "2026-07-02", "2026-08-01")
        await backfill_sessions(prisma, _NoAutoRouters(), GROUP_KINDS, "2026-07-02", "2026-08-01")
        assert table.upserts[0][1]["update"]["turns"] == table.upserts[1][1]["update"]["turns"] == 1

    async def test_model_state_is_written_through_prismas_json_wrapper(self):
        """Model names contain a slash; a raw dict fails prisma's GraphQL parse.

        Regression lock: this shipped as a backfill that reported writing 97
        sessions while every upsert failed, because a fake table happily accepts
        the dict that the real client rejects.
        """
        import prisma

        base = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        table = _FakeTable()
        prisma_client = _FakePrisma([spend_row("s1", "anthropic/claude-haiku-4-5", base, created=5000)], table)
        await backfill_sessions(prisma_client, _NoAutoRouters(), GROUP_KINDS, "2026-07-02", "2026-08-01")
        assert isinstance(table.upserts[0][1]["create"]["model_state"], prisma.Json)

    async def test_a_failed_write_is_not_counted_as_a_written_session(self):
        """The count reports what landed, not what was attempted."""

        class _FailingTable(_FakeTable):
            async def upsert(self, where, data):
                raise RuntimeError("upsert rejected")

        base = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        prisma_client = _FakePrisma(
            [spend_row("s1", "anthropic/claude-haiku-4-5", base, created=5000)], _FailingTable()
        )
        result = await backfill_sessions(prisma_client, _NoAutoRouters(), GROUP_KINDS, "2026-07-02", "2026-08-01")
        assert result.rows_read == 1
        assert result.sessions_written == 0

    async def test_a_session_with_no_configured_baseline_reports_no_savings(self):
        base = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        table = _FakeTable()
        prisma = _FakePrisma([spend_row("s1", "anthropic/claude-haiku-4-5", base, spend=0.02)], table)
        await backfill_sessions(prisma, _NoAutoRouters(), GROUP_KINDS, "2026-07-02", "2026-08-01")
        created = table.upserts[0][1]["create"]
        assert created["spend"] == pytest.approx(0.02)
        assert created["baseline_spend"] == pytest.approx(0.02)
