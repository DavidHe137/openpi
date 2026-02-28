from dataclasses import dataclass
import itertools
import time

from openpi_client.messages import InferRequest

_request_id_counter = itertools.count(1)


@dataclass(frozen=True)
class ArrivedRequest:
    infer_request: InferRequest
    request_id: int
    arrival_timestamp: float

    @classmethod
    def receive(cls, req: InferRequest) -> "ArrivedRequest":
        return cls(
            infer_request=req,
            request_id=next(_request_id_counter),
            arrival_timestamp=time.time(),
        )

    def dequeue(self) -> "DequeuedRequest":
        return DequeuedRequest(
            infer_request=self.infer_request,
            request_id=self.request_id,
            arrival_timestamp=self.arrival_timestamp,
            dequeue_timestamp=time.time(),
        )


@dataclass(frozen=True)
class DequeuedRequest:
    infer_request: InferRequest
    request_id: int
    arrival_timestamp: float
    dequeue_timestamp: float

    def send(self) -> "SentRequest":
        return SentRequest(
            infer_request=self.infer_request,
            request_id=self.request_id,
            arrival_timestamp=self.arrival_timestamp,
            dequeue_timestamp=self.dequeue_timestamp,
            send_timestamp=time.time(),
        )


@dataclass(frozen=True)
class SentRequest:
    infer_request: InferRequest
    request_id: int
    arrival_timestamp: float
    dequeue_timestamp: float
    send_timestamp: float


@dataclass(frozen=True)
class NewConnection:
    robot_id: str
    response_sock_addr: str
