"""
Deferred close of HTTP/SDK clients that the LLM client cache has evicted.

Eviction only drops the cache's reference to a client. Every OpenAI/Azure SDK
client is a reference cycle (each resource namespace holds the client back), so
an evicted client and its pooled TCP connections survive until a generational
collection runs, which under load is thousands of requests later.

Closing at eviction time is not an option: a request that was handed the client
just before it was evicted is still using it, and closing it underneath that
request raises ``RuntimeError: Cannot send a request, as the client has been
closed.``

So an evicted client is closed once a grace window has passed since its
eviction, by which point any request that already held it has finished. Only
clients litellm itself created are closed; a client the caller supplied is left
alone because litellm does not own its lifecycle.

A client that closes synchronously is closed from wherever the cache is next
used. One whose close is a coroutine needs the event loop it was evicted on,
so it waits for a call from that loop rather than having work scheduled onto a
loop it does not belong to.

The queue holds its clients weakly, so waiting out a grace window never keeps
alive anything the collector would have reclaimed first.
"""

import asyncio
import inspect
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from litellm.constants import EVICTED_LLM_CLIENT_CLOSE_GRACE_SECONDS


@dataclass(frozen=True, slots=True)
class _PendingClose:
    """A queued close.

    The client is held weakly, so queueing one never keeps alive anything the
    collector would otherwise have reclaimed first.

    ``needs_loop`` is set for a client whose close is a coroutine; those can only
    be closed from the event loop they were evicted on, recorded in ``loop_id``.
    A client that closes synchronously carries neither constraint.
    """

    client_ref: "weakref.ref[object]"
    loop_id: int | None
    needs_loop: bool
    close_after: float


def _running_loop_id() -> int | None:
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        return None


def _close_function(client: object) -> Callable[[], object] | None:
    close_fn: Callable[[], object] | None = getattr(client, "aclose", None) or getattr(client, "close", None)
    return close_fn


async def _close_quietly(closing: Awaitable[object]) -> None:
    try:
        await closing
    except Exception:  # noqa: BLE001 - a discarded client's close must never surface to callers
        pass


class EvictedClientCloser:
    """Closes evicted, litellm-owned clients once their grace window elapses."""

    def __init__(
        self,
        grace_seconds: float = EVICTED_LLM_CLIENT_CLOSE_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._grace_seconds = grace_seconds
        self._clock = clock
        self._owned: weakref.WeakSet[object] = weakref.WeakSet()
        self._pending: tuple[_PendingClose, ...] = ()
        self._queue_lock = threading.Lock()  # the cache is reachable from every worker thread's loop
        self._close_tasks: set[asyncio.Task[None]] = set()  # mutable-ok: strong refs to running closes

    def mark_owned(self, client: object) -> None:
        """Record that litellm created this client, so it may be closed on eviction."""
        try:
            self._owned.add(client)
        except TypeError:
            pass  # values that cannot be weak-referenced are never litellm clients

    def _is_owned(self, client: object) -> bool:
        try:
            return client in self._owned
        except TypeError:
            return False  # unhashable values are never litellm clients

    def schedule(self, client: object) -> None:
        """Queue an evicted client for closing once its grace window elapses."""
        if client is None or not self._is_owned(client):
            return
        close_fn = _close_function(client)
        if close_fn is None:
            return
        pending = _PendingClose(
            client_ref=weakref.ref(client),
            loop_id=_running_loop_id(),
            needs_loop=inspect.iscoroutinefunction(close_fn),
            close_after=self._clock() + self._grace_seconds,
        )
        with self._queue_lock:
            self._pending += (pending,)

    def reap(self) -> None:
        """Close every queued client that is due and closable from here.

        Called from the cache's read path, so the empty-queue exit comes first.
        """
        if not self._pending:
            return
        loop_id = _running_loop_id()
        now = self._clock()

        def is_due(pending: _PendingClose) -> bool:
            if pending.close_after > now:
                return False
            if not pending.needs_loop:
                return True
            return loop_id is not None and pending.loop_id in (None, loop_id)

        with self._queue_lock:
            due = tuple(pending for pending in self._pending if is_due(pending))
            if not due:
                return
            self._pending = tuple(pending for pending in self._pending if not is_due(pending))
        for pending in due:
            client = pending.client_ref()
            if client is not None:
                self._close(client)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _close(self, client: object) -> None:
        close_fn = _close_function(client)
        if close_fn is None:
            return
        try:
            closing = close_fn()
        except Exception:  # noqa: BLE001 - a discarded client's close must never surface to callers
            return
        if not inspect.isawaitable(closing):
            return
        task = asyncio.get_running_loop().create_task(_close_quietly(closing))
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)


default_evicted_client_closer = EvictedClientCloser()
