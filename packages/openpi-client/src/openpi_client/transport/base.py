from abc import ABC, abstractmethod


class ClientTransport(ABC):
    """Sync message-oriented transport. One framed message per send/receive."""

    @abstractmethod
    def send_message(self, data: bytes) -> None: ...

    @abstractmethod
    def receive_message(self) -> bytes: ...

    @abstractmethod
    def close(self) -> None: ...
