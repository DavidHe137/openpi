import asyncio
import logging
import queue
import ssl
import threading
from typing import Optional

from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived

from openpi_client.transport.base import ClientTransport

logger = logging.getLogger(__name__)

_QUIC_ALPN = ["openpi/1"]


class _StreamFramedProtocol(QuicConnectionProtocol):
    """Treats each stream as a single message (ends when stream fin arrives)."""

    def __init__(self, *args, inbox: "asyncio.Queue[bytes]", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inbox = inbox
        self._buffers: dict[int, bytearray] = {}

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, StreamDataReceived):
            buf = self._buffers.setdefault(event.stream_id, bytearray())
            buf.extend(event.data)
            if event.end_stream:
                data = bytes(self._buffers.pop(event.stream_id))
                self._inbox.put_nowait(data)

    async def send_one(self, data: bytes) -> None:
        stream_id = self._quic.get_next_available_stream_id(is_unidirectional=True)
        logger.debug("quic send_one: stream=%d bytes=%d", stream_id, len(data))
        self._quic.send_stream_data(stream_id, data, end_stream=True)
        self.transmit()
        logger.debug("quic send_one: stream=%d transmit() returned", stream_id)


class QuicClientTransport(ClientTransport):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
        protocol: _StreamFramedProtocol,
        client_ctx,  # async context manager from aioquic.asyncio.connect
        inbox_async: "asyncio.Queue[bytes]",
    ) -> None:
        self._loop = loop
        self._thread = thread
        self._protocol = protocol
        self._client_ctx = client_ctx
        self._inbox_async = inbox_async
        self._inbox_sync: queue.Queue = queue.Queue()
        self._drain_task = asyncio.run_coroutine_threadsafe(self._drain_inbox(), loop)
        self._closed = False

    async def _drain_inbox(self) -> None:
        while True:
            data = await self._inbox_async.get()
            self._inbox_sync.put(data)

    @classmethod
    def connect(
        cls,
        host: str,
        port: Optional[int],
    ) -> "QuicClientTransport":
        if port is None:
            raise ValueError("QUIC transport requires an explicit port")

        loop_ready = threading.Event()
        container: dict = {}

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Create the queue inside this loop so it binds to the correct loop
            # on Python 3.8 (where asyncio.Queue captures the current loop at
            # construction time).
            container["loop"] = loop
            container["inbox_async"] = asyncio.Queue()
            loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_runner, name="quic-client-loop", daemon=True)
        thread.start()
        loop_ready.wait()
        loop: asyncio.AbstractEventLoop = container["loop"]
        inbox_async: asyncio.Queue[bytes] = container["inbox_async"]

        config = QuicConfiguration(is_client=True, alpn_protocols=_QUIC_ALPN)
        config.verify_mode = ssl.CERT_NONE

        def _mk_protocol(*args, **kwargs):
            return _StreamFramedProtocol(*args, inbox=inbox_async, **kwargs)

        async def _open():
            ctx = connect(host, port, configuration=config, create_protocol=_mk_protocol)
            protocol = await ctx.__aenter__()
            await protocol.wait_connected()
            return ctx, protocol

        fut = asyncio.run_coroutine_threadsafe(_open(), loop)
        ctx, protocol = fut.result()

        return cls(loop=loop, thread=thread, protocol=protocol, client_ctx=ctx, inbox_async=inbox_async)

    def send_message(self, data: bytes) -> None:
        logger.debug("send_message: scheduling send of %d bytes", len(data))
        fut = asyncio.run_coroutine_threadsafe(self._protocol.send_one(data), self._loop)
        fut.result()
        logger.debug("send_message: send of %d bytes completed", len(data))

    def receive_message(self) -> bytes:
        logger.debug("receive_message: waiting on inbox")
        data = self._inbox_sync.get()
        logger.debug("receive_message: got %d bytes", len(data))
        return data

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _shutdown():
            try:
                await self._client_ctx.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing QUIC client")

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=5)
        except Exception:
            logger.exception("QUIC client shutdown failed")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
