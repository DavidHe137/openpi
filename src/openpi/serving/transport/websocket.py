from fastapi import WebSocket

from openpi.serving.transport.base import ServerTransport


class FastApiWebSocketTransport(ServerTransport):
    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def send_message(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def receive_message(self) -> bytes:
        return await self._ws.receive_bytes()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code=code, reason=reason)
