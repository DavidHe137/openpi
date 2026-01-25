from dataclasses import dataclass
import time

import numpy as np
from openpi_client.messages import InferRequest
from openpi_client.messages import InferType
from openpi_client.messages import RTCParams
from openpi_client.messages import TrainTimeRTCParams
from openpi_client.messages import VlashParams


@dataclass
class InferRequestForServer:
    request_id: int
    robot_id: str
    observation: dict
    infer_type: InferType
    params: RTCParams | VlashParams | TrainTimeRTCParams | None
    return_debug_data: bool = False
    noise: np.ndarray | None = None  # optional, noise for deterministic repla
    deadline: float | None = None
    arrival_timestamp: float | None = None
    dequeue_timestamp: float | None = None
    send_timestamp: float | None = None

    last_request_id: int = 0

    @classmethod
    def from_infer_request(cls, infer_request: InferRequest) -> "InferRequestForServer":
        cls.last_request_id += 1
        return cls(
            request_id=cls.last_request_id,
            robot_id=infer_request.robot_id,
            observation=infer_request.observation,
            infer_type=infer_request.infer_type,
            params=infer_request.params,
            return_debug_data=infer_request.return_debug_data,
            noise=infer_request.noise,
            deadline=infer_request.deadline,
            arrival_timestamp=None,
            dequeue_timestamp=None,
            send_timestamp=None,
        )

    def arrived(self) -> bool:
        self.arrival_timestamp = time.time()
        return True

    def dequeued(self) -> bool:
        self.dequeue_timestamp = time.time()
        return True

    def sent(self) -> bool:
        self.send_timestamp = time.time()
        return True


class InferResponseForServer:
    request_id: int
    actions: np.ndarray
    execution_horizon: int
    times: dict
    debug_data: dict | None = None
