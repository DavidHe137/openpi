import asyncio
from collections.abc import Awaitable, Callable
import datetime
import logging
from pathlib import Path
import tempfile

from aioquic.asyncio import serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ConnectionTerminated
from aioquic.quic.events import QuicEvent
from aioquic.quic.events import StreamDataReceived

from openpi.serving.transport.base import ServerTransport

logger = logging.getLogger(__name__)

_QUIC_ALPN = ["openpi/1"]


class _ServerStreamProtocol(QuicConnectionProtocol):
    """Treats each inbound QUIC stream as one framed message."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inbox: asyncio.Queue[bytes] = asyncio.Queue()
        self._buffers: dict[int, bytearray] = {}
        self._connection_closed = asyncio.Event()
        self.transport_obj: QuicServerTransport | None = None

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, StreamDataReceived):
            buf = self._buffers.setdefault(event.stream_id, bytearray())
            buf.extend(event.data)
            logger.debug(
                "quic server recv: stream=%d +%d bytes (total=%d) end=%s",
                event.stream_id,
                len(event.data),
                len(buf),
                event.end_stream,
            )
            if event.end_stream:
                data = bytes(self._buffers.pop(event.stream_id))
                self._inbox.put_nowait(data)
        elif isinstance(event, ConnectionTerminated):
            logger.warning(
                "quic server connection terminated: error_code=%s frame_type=%s reason=%r",
                getattr(event, "error_code", None),
                getattr(event, "frame_type", None),
                getattr(event, "reason_phrase", None),
            )
            self._connection_closed.set()
            # Unblock any pending recv by signaling EOF via sentinel.
            self._inbox.put_nowait(b"")


class QuicServerTransport(ServerTransport):
    def __init__(self, protocol: _ServerStreamProtocol) -> None:
        self._protocol = protocol
        protocol.transport_obj = self

    async def send_message(self, data: bytes) -> None:
        stream_id = self._protocol._quic.get_next_available_stream_id(is_unidirectional=True)
        self._protocol._quic.send_stream_data(stream_id, data, end_stream=True)
        self._protocol.transmit()

    async def receive_message(self) -> bytes:
        data = await self._protocol._inbox.get()
        if data == b"" and self._protocol._connection_closed.is_set():
            raise ConnectionError("QUIC connection terminated")
        return data

    async def close(self) -> None:
        self._protocol.close()


def _generate_self_signed_cert() -> tuple[Path, Path]:
    """Generate an ephemeral self-signed cert/key pair in a temp dir. Returns (cert_path, key_path)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "openpi")])
    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="openpi_quic_"))
    cert_path = tmpdir / "cert.pem"
    key_path = tmpdir / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    logger.info("Generated self-signed QUIC cert at %s", tmpdir)
    return cert_path, key_path


async def start_quic_listener(
    host: str,
    port: int,
    on_connection: Callable[[ServerTransport], Awaitable[None]],
) -> asyncio.Server:
    """Start an aioquic listener. Each new connection invokes on_connection(transport) as a task."""
    cert_path, key_path = _generate_self_signed_cert()

    config = QuicConfiguration(is_client=False, alpn_protocols=_QUIC_ALPN)
    config.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

    # We need to invoke on_connection once the handshake completes per connection.
    # aioquic's `serve` returns after binding but doesn't surface per-connection
    # events directly; we hook into the protocol lifecycle via a custom subclass.

    pending_tasks: set[asyncio.Task] = set()

    class _RoutingProtocol(_ServerStreamProtocol):
        def connection_made(self, transport) -> None:  # type: ignore[override]
            super().connection_made(transport)
            server_transport = QuicServerTransport(self)
            task = asyncio.create_task(on_connection(server_transport))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

    return await serve(
        host=host,
        port=port,
        configuration=config,
        create_protocol=_RoutingProtocol,
    )
