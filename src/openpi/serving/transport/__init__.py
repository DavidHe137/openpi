from openpi.serving.transport.base import ServerTransport
from openpi.serving.transport.websocket import FastApiWebSocketTransport

__all__ = ["FastApiWebSocketTransport", "ServerTransport", "start_quic_listener"]


def start_quic_listener(*args, **kwargs):
    """Lazy import so aioquic is only loaded when actually using QUIC."""
    from openpi.serving.transport.quic import start_quic_listener as _impl

    return _impl(*args, **kwargs)
