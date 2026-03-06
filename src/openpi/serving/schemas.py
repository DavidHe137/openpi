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
    estimated_d_param: int = 0  # filled by scheduler before batching


@dataclass(frozen=True)
class CompletionNotification:
    """Sent from GPU to scheduler after inference so the scheduler can update its state."""

    robot_id: str
    start_step: int
    request_id: int
    batch_size: int
    inference_duration_ms: float


@dataclass(frozen=True)
class AckNotification:
    """Sent from WS to scheduler when a client acks receipt of an InferResponse."""

    robot_id: str
    request_id: int
    receive_time: float
    server_send_time: float


@dataclass(frozen=True)
class BatchProfile:
    """Latency profile per batch size (ms). Sent once from GPU to scheduler after warmup."""

    latency_ms: dict[int, float]


@dataclass
class WarmupSeed:
    robot_id: str
    obs_samples: list[tuple[float, float]]  # (arrival_ts, request_ts) per ping
    delivery_samples: list[tuple[float, float]]  # (client_receive_time, server_send_time) per ack


@dataclass(frozen=True)
class SchedulerTimingSample:
    """A single scheduler timing sample emitted by the scheduler process."""

    scheduler_name: str
    metric_name: str
    duration_ms: float
    recorded_at: float
