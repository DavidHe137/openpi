from openpi_client.transport.base import ClientTransport
from openpi_client.transport.websocket import WebSocketClientTransport

TransportKind = str  # Literal["ws", "quic"] requires py3.8+ typing_extensions; keep str for simplicity

__all__ = [
    "ClientTransport",
    "WebSocketClientTransport",
    "create_transport",
]


def create_transport(
    kind: TransportKind,
    *,
    host: str,
    port,
    api_key=None,
) -> ClientTransport:
    if kind == "ws":
        return WebSocketClientTransport.connect(host=host, port=port, api_key=api_key)
    if kind == "quic":
        from openpi_client.transport.quic import QuicClientTransport

        return QuicClientTransport.connect(host=host, port=port)
    raise ValueError(f"Unknown transport kind: {kind!r}")
