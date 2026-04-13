from abc import ABC
from abc import abstractmethod


class ServerTransport(ABC):
    """Async message-oriented transport. One framed message per send/receive."""

    @abstractmethod
    async def send_message(self, data: bytes) -> None: ...

    @abstractmethod
    async def receive_message(self) -> bytes: ...

    @abstractmethod
    async def close(self) -> None: ...
