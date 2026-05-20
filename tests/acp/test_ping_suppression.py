"""Tests for acp_adapter.entry._BenignProbeMethodFilter.

Covers both the isolated filter logic and the full end-to-end path where a
client sends a bare JSON-RPC ``ping`` request over stdio and the acp runtime
surfaces the resulting ``RequestError`` via ``logging.exception("Background
task failed", ...)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from io import StringIO

import pytest

from acp.exceptions import RequestError

from acp_adapter.entry import _BenignProbeMethodFilter


# -- Unit tests on the filter itself ----------------------------------------


def _make_record(msg: str, exc: BaseException | None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="root",
        level=logging.ERROR,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=(),
        exc_info=(type(exc), exc, exc.__traceback__) if exc else None,
    )
    return record


def _bake_tb(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as e:  # noqa: BLE001
        return e


@pytest.mark.parametrize("method", ["ping", "health", "healthcheck"])
def test_filter_suppresses_benign_probe(method: str) -> None:
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RequestError.method_not_found(method))
    record = _make_record("Background task failed", exc)
    assert f.filter(record) is False


def test_filter_allows_real_method_not_found() -> None:
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RequestError.method_not_found("session/custom"))
    record = _make_record("Background task failed", exc)
    assert f.filter(record) is True


def test_filter_allows_non_request_error() -> None:
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RuntimeError("boom"))
    record = _make_record("Background task failed", exc)
    assert f.filter(record) is True


def test_filter_allows_different_message_even_for_ping() -> None:
    """Only 'Background task failed' is muted — other messages pass through."""
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RequestError.method_not_found("ping"))
    record = _make_record("Some other context", exc)
    assert f.filter(record) is True


def test_filter_allows_request_error_with_different_code() -> None:
    f = _BenignProbeMethodFilter()
    exc = _bake_tb(RequestError.invalid_params({"method": "ping"}))
    record = _make_record("Background task failed", exc)
    assert f.filter(record) is True


def test_filter_allows_log_without_exc_info() -> None:
    f = _BenignProbeMethodFilter()
    record = _make_record("Background task failed", None)
    assert f.filter(record) is True


# -- End-to-end: drive a real JSON-RPC `ping` through acp.run_agent ---------


class _FakeAgent:
    """Minimal acp.Agent stub — we only need the router to build."""

    async def initialize(self, **kwargs):  # noqa: ANN003
        from acp.schema import AgentCapabilities, InitializeResponse

        return InitializeResponse(protocol_version=1, agent_capabilities=AgentCapabilities())

    async def new_session(self, cwd, mcp_servers=None, **kwargs):  # noqa: ANN001, ANN003
        from acp.schema import NewSessionResponse

        return NewSessionResponse(session_id="test")

    async def prompt(self, session_id, prompt, **kwargs):  # noqa: ANN001, ANN003
        from acp.schema import PromptResponse

        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id, **kwargs):  # noqa: ANN001, ANN003
        pass

    async def authenticate(self, **kwargs):  # noqa: ANN003
        pass

    def on_connect(self, conn):  # noqa: ANN001
        pass


class _MemoryTransport(asyncio.Transport):
    """In-memory write transport that feeds bytes to a peer StreamReader."""

    def __init__(self, loop: asyncio.AbstractEventLoop, peer: asyncio.StreamReader):
        super().__init__()
        self._loop = loop
        self._peer = peer
        self._protocol: asyncio.BaseProtocol | None = None
        self._closing = False

    def set_protocol(self, protocol: asyncio.BaseProtocol) -> None:
        self._protocol = protocol

    def get_protocol(self) -> asyncio.BaseProtocol | None:
        return self._protocol

    def is_closing(self) -> bool:
        return self._closing

    def write(self, data: bytes) -> None:
        if self._closing:
            raise RuntimeError("transport is closing")
        self._peer.feed_data(data)

    def writelines(self, list_of_data) -> None:  # noqa: ANN001
        for data in list_of_data:
            self.write(data)

    def can_write_eof(self) -> bool:
        return True

    def write_eof(self) -> None:
        if not self._closing:
            self._peer.feed_eof()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._peer.feed_eof()
        if self._protocol is not None:
            self._loop.call_soon(self._protocol.connection_lost, None)

    def get_extra_info(self, name: str, default=None):  # noqa: ANN001, ANN201
        return default


def _make_memory_stream_pair(loop: asyncio.AbstractEventLoop):
    """Return two connected (reader, writer) stream endpoints."""
    left_reader = asyncio.StreamReader(limit=1024 * 1024, loop=loop)
    right_reader = asyncio.StreamReader(limit=1024 * 1024, loop=loop)
    left_protocol = asyncio.StreamReaderProtocol(left_reader, loop=loop)
    right_protocol = asyncio.StreamReaderProtocol(right_reader, loop=loop)
    left_transport = _MemoryTransport(loop, right_reader)
    right_transport = _MemoryTransport(loop, left_reader)
    left_transport.set_protocol(left_protocol)
    right_transport.set_protocol(right_protocol)
    left_protocol.connection_made(left_transport)
    right_protocol.connection_made(right_transport)
    left_writer = asyncio.StreamWriter(left_transport, left_protocol, left_reader, loop)
    right_writer = asyncio.StreamWriter(right_transport, right_protocol, right_reader, loop)
    return (left_reader, left_writer), (right_reader, right_writer)


@pytest.mark.asyncio
async def test_bare_ping_request_produces_proper_response_and_no_stderr_noise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bare ``ping`` must get a JSON-RPC -32601 back AND leave stderr clean
    when the filter is installed on the handler.
    """


    # Attach the filter to a fresh stream handler that mirrors entry._setup_logging.
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s|%(levelname)s|%(message)s"))
    handler.addFilter(_BenignProbeMethodFilter())
    root = logging.getLogger()
    prior_handlers = root.handlers[:]
    prior_level = root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # Also suppress propagation of caplog's default handler interfering with
    # our stream (caplog still captures via its own propagation hook).
    try:
        import acp

        loop = asyncio.get_running_loop()
        (client_input, client_output), (agent_input, agent_output) = (
            _make_memory_stream_pair(loop)
        )

        agent_task = asyncio.create_task(
            acp.run_agent(
                _FakeAgent(),
                input_stream=agent_output,
                output_stream=agent_input,
                use_unstable_protocol=True,
            )
        )

        # Send a bare `ping`
        request = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
        client_output.write((json.dumps(request) + "\n").encode())
        await client_output.drain()

        response_line = await asyncio.wait_for(client_input.readline(), timeout=5.0)
        # Give the supervisor task a tick to fire (filter should eat it)
        await asyncio.sleep(0.2)

        response = json.loads(response_line.decode())
        assert response["error"]["code"] == -32601, response
        assert response["error"]["data"] == {"method": "ping"}, response

        logs = stream.getvalue()
        assert "Background task failed" not in logs, (
            f"ping noise leaked to stderr:\n{logs}"
        )

        # Clean shutdown
        client_output.close()
        try:
            await asyncio.wait_for(agent_task, timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            agent_task.cancel()
            try:
                await agent_task
            except BaseException:  # noqa: BLE001
                pass
    finally:
        root.handlers = prior_handlers
        root.setLevel(prior_level)
