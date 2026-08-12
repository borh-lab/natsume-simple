from threading import Event
from types import SimpleNamespace

import anyio
import duckdb
import pytest
from anyio import CapacityLimiter

from natsume_simple import api
from natsume_simple.api import PublicApiError, run_bounded_query


class InterruptibleConnection:
    def __init__(self):
        self.interrupted = Event()
        self.interrupt_count = 0
        self.closed = False

    def interrupt(self):
        self.interrupt_count += 1
        self.interrupted.set()

    def close(self):
        self.closed = True


def test_query_capacity_rejects_instead_of_queueing():
    async def exercise():
        limiter = CapacityLimiter(1)
        occupied_by = object()
        limiter.acquire_on_behalf_of_nowait(occupied_by)
        try:
            with pytest.raises(PublicApiError) as raised:
                await run_bounded_query(
                    InterruptibleConnection(), limiter, lambda: None, timeout=1
                )
        finally:
            limiter.release_on_behalf_of(occupied_by)

        assert raised.value.status_code == 429
        assert raised.value.code == "capacity_exceeded"
        assert raised.value.headers == {"Retry-After": "1"}

    anyio.run(exercise)


def test_timed_out_request_closes_its_connection_and_the_next_uses_a_new_one(
    monkeypatch,
):
    opened: list[InterruptibleConnection] = []

    def connect(*_args, **_kwargs):
        connection = InterruptibleConnection()
        opened.append(connection)
        return connection

    monkeypatch.setattr(api.duckdb, "connect", connect)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(database_path="x"))
    )

    async def exercise():
        limiter = CapacityLimiter(1)
        dependency = api.database_connection(request)
        connection = await dependency.__anext__()

        def wait_for_interrupt():
            assert connection.interrupted.wait(1)
            raise duckdb.InterruptException("query interrupted")

        try:
            with pytest.raises(PublicApiError) as raised:
                await run_bounded_query(
                    connection, limiter, wait_for_interrupt, timeout=0.01
                )
        finally:
            await dependency.aclose()

        assert raised.value.status_code == 504
        assert raised.value.code == "query_timeout"
        assert limiter.borrowed_tokens == 0
        assert connection.closed

        next_dependency = api.database_connection(request)
        next_connection = await next_dependency.__anext__()
        try:
            assert (
                await run_bounded_query(
                    next_connection, limiter, lambda: "next", timeout=1
                )
                == "next"
            )
        finally:
            await next_dependency.aclose()

        assert next_connection is not connection
        assert next_connection.closed

    anyio.run(exercise)
    assert opened[0].interrupt_count == 1


def test_completed_query_cancels_its_interrupt_timer():
    connection = InterruptibleConnection()

    async def exercise():
        assert (
            await run_bounded_query(
                connection, CapacityLimiter(1), lambda: "done", timeout=0.01
            )
            == "done"
        )
        await anyio.sleep(0.02)

    anyio.run(exercise)
    assert connection.interrupt_count == 0
