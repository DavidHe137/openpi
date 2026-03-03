from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from openpi_client.messages import InferType
from openpi_client.messages import RTCParams
from openpi_client.messages import TrainTimeRTCParams
from openpi_client.messages import VlashParams

DEFAULT_EXECUTION_HORIZON = 10
_request_id_counter = itertools.count(1)


@dataclass(frozen=True)
class SlotRequest:
    """Flows end-to-end: built by WS → sent to Scheduler → put in batch_queue → received by GPU."""

    slot_index: int
    robot_id: str
    request_id: int
    arrival_timestamp: float  # when WS received the request (server-side)
    start_step: int
    request_timestamp: float
    deadline: float
    infer_type: InferType
    params: RTCParams | VlashParams | TrainTimeRTCParams | None
    noise: np.ndarray | None
    min_execution_horizon: int = 0  # minimum steps to execute before server will re-infer this robot


@dataclass(frozen=True)
class CompletionNotification:
    """Sent from GPU to scheduler after inference so the scheduler can update its state."""

    robot_id: str
    start_step: int


@dataclass(frozen=True)
class BatchProfile:
    """Latency profile per batch size (ms). Sent once from GPU to scheduler after warmup."""

    latency_ms: dict[int, float]
