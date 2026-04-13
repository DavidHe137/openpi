import logging
import time
from typing import Optional

import requests
import websockets.sync.client

from openpi_client.transport.base import ClientTransport

logger = logging.getLogger(__name__)


def _parse_urls(host: str, port: Optional[int]):
    explicit_scheme = False
    if host.startswith("https://"):
        ws_scheme, http_scheme = "wss", "https"
        host = host[len("https://") :]
        explicit_scheme = True
    elif host.startswith("http://"):
        ws_scheme, http_scheme = "ws", "http"
        host = host[len("http://") :]
        explicit_scheme = True
    else:
        ws_scheme, http_scheme = "ws", "http"
    base = host if (port is None or explicit_scheme) else f"{host}:{port}"
    return f"{ws_scheme}://{base}/ws", f"{http_scheme}://{base}"


def wait_for_server(host: str, port: Optional[int], api_key: Optional[str] = None) -> dict:
    """Poll /metadata until server responds. Returns the raw metadata dict."""
    _, http_base = _parse_urls(host, port)
    logging.info(f"Waiting for server at {http_base}...")
    while True:
        try:
            resp = requests.get(
                f"{http_base}/metadata",
                headers={"Authorization": f"Api-Key {api_key}"} if api_key else None,
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            logging.info("Still waiting for server...")
            time.sleep(5)


class WebSocketClientTransport(ClientTransport):
    def __init__(self, ws: "websockets.sync.client.ClientConnection") -> None:
        self._ws = ws

    @classmethod
    def connect(
        cls,
        host: str,
        port: Optional[int],
        *,
        api_key: Optional[str] = None,
        tunnel_url: Optional[str] = None,
    ) -> "WebSocketClientTransport":
        ws_uri, _ = _parse_urls(host, port)
        if tunnel_url:
            tunnel_host = tunnel_url.replace("https://", "", 1)
            ws_uri = f"wss://{tunnel_host}/ws"
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        ws = websockets.sync.client.connect(
            ws_uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
        )
        return cls(ws)

    def send_message(self, data: bytes) -> None:
        self._ws.send(data)

    def receive_message(self) -> bytes:
        msg = self._ws.recv()
        if isinstance(msg, str):
            return msg.encode()
        return msg

    def close(self) -> None:
        self._ws.close()
