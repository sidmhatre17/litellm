"""
Tests for EvictedClientCloser.

An evicted client must stay open long enough for a request that already holds it
to finish, and must then actually be closed, otherwise its connection pool is
retained until a generational collection runs. A client the caller supplied is
never closed, because litellm does not own its lifecycle.
"""

import asyncio
import gc
import weakref

import pytest

from litellm.caching.evicted_client_closer import EvictedClientCloser


class FakeClock:
    """Hand-advanced monotonic clock, so grace windows need no real waiting."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class AsyncClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class SyncClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def make_closer(clock: FakeClock, grace_seconds: float = 60.0) -> EvictedClientCloser:
    return EvictedClientCloser(grace_seconds=grace_seconds, clock=clock)


@pytest.mark.asyncio
async def test_owned_client_is_closed_once_the_grace_window_elapses():
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()

    closer.mark_owned(client)
    closer.schedule(client)
    clock.advance(61.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is True
    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_owned_client_stays_open_inside_the_grace_window():
    """A request handed the client just before eviction is still using it."""
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()

    closer.mark_owned(client)
    closer.schedule(client)
    clock.advance(59.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is False
    assert closer.pending_count == 1


@pytest.mark.asyncio
async def test_caller_supplied_client_is_never_closed():
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()

    closer.schedule(client)
    clock.advance(3600.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is False
    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_sync_client_is_closed_once_the_grace_window_elapses():
    clock = FakeClock()
    closer = make_closer(clock)
    client = SyncClient()

    closer.mark_owned(client)
    closer.schedule(client)
    clock.advance(61.0)
    closer.reap()

    assert client.closed is True


@pytest.mark.asyncio
async def test_a_failing_close_does_not_propagate_or_block_the_others():
    class ExplodingClient:
        async def close(self) -> None:
            raise RuntimeError("connection already gone")

    clock = FakeClock()
    closer = make_closer(clock)
    exploding, healthy = ExplodingClient(), AsyncClient()

    for client in (exploding, healthy):
        closer.mark_owned(client)
        closer.schedule(client)
    clock.advance(61.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert healthy.closed is True


@pytest.mark.asyncio
async def test_an_unhashable_cached_value_does_not_break_eviction():
    """The cache holds arbitrary values; an ownership test must never raise on one."""

    class Unhashable:
        __hash__ = None  # pyright: ignore[reportAssignmentType]  # unhashable by construction

    clock = FakeClock()
    closer = make_closer(clock)

    closer.mark_owned(Unhashable())
    closer.schedule(Unhashable())

    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_values_with_nothing_to_close_are_never_queued():
    """The cache holds plain values too; those have nothing to reclaim."""

    class NotAClient:
        pass

    clock = FakeClock()
    closer = make_closer(clock)
    value = NotAClient()

    closer.mark_owned(value)
    closer.schedule(value)

    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_a_queued_client_is_not_kept_alive_by_the_queue():
    """Waiting out a grace window must not retain what the collector would free first."""
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()
    gone = weakref.ref(client)

    closer.mark_owned(client)
    closer.schedule(client)
    del client
    gc.collect()

    assert gone() is None, "the pending queue is holding the client alive"

    clock.advance(61.0)
    closer.reap()
    assert closer.pending_count == 0


def test_sync_client_evicted_outside_an_event_loop_is_still_closed():
    """The sync httpx handler is cached and evicted from call sites with no loop."""
    clock = FakeClock()
    closer = make_closer(clock)
    client = SyncClient()

    closer.mark_owned(client)
    closer.schedule(client)
    assert closer.pending_count == 1

    clock.advance(61.0)
    closer.reap()

    assert client.closed is True
    assert closer.pending_count == 0


@pytest.mark.asyncio
async def test_an_async_client_waits_for_a_loop_rather_than_being_dropped():
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()
    closer.mark_owned(client)

    def schedule_outside_a_loop() -> None:
        closer.schedule(client)
        clock.advance(61.0)
        closer.reap()

    await asyncio.to_thread(schedule_outside_a_loop)
    assert client.closed is False, "no loop was running, so it could not have been closed"
    assert closer.pending_count == 1

    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is True


@pytest.mark.asyncio
async def test_a_client_evicted_on_another_event_loop_is_left_alone():
    """Closing a client bound to a different loop would schedule work on that loop."""
    clock = FakeClock()
    closer = make_closer(clock)
    client = AsyncClient()
    closer.mark_owned(client)

    def schedule_on_its_own_loop() -> None:
        asyncio.run(_schedule())

    async def _schedule() -> None:
        closer.schedule(client)

    await asyncio.to_thread(schedule_on_its_own_loop)
    assert closer.pending_count == 1

    clock.advance(61.0)
    closer.reap()
    await asyncio.sleep(0.05)

    assert client.closed is False
    assert closer.pending_count == 1
