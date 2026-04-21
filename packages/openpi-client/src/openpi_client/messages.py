from dataclasses import dataclass
from enum import Enum
import numpy as np
import time
from typing import Literal, Optional, Union
from jaxtyping import Float


class SyncedClock:
    """Client-side clock that can report time adjusted to the server's clock.

    The offset is estimated during warmup using the NTP formula:
        offset = ((t2 - t1) + (t3 - t4)) / 2
    where t1/t4 are client timestamps and t2/t3 are server timestamps,
    such that: server_time ≈ client_time + offset.
    """

    def __init__(self) -> None:
        self._offset: float = 0.0

    def set_offset(self, offset: float) -> None:
        """Set estimated offset: server_clock - client_clock."""
        self._offset = offset

    def now(self) -> float:
        """Current time in local client clock."""
        return time.time()

    def now_server(self) -> float:
        """Current time adjusted to server clock."""
        return time.time() + self._offset


# TODO: merge with broker types
class InferType(Enum):
    SYNC = "sync"
    INFERENCE_TIME_RTC = "inference_time_rtc"
    TRAIN_TIME_RTC = "train_time_rtc"
    VLASH = "vlash"


@dataclass
class RTCParams:
    prev_action: Float[np.ndarray, "action_horizon action_dim"]
    s_param: int
    d_param: int


@dataclass
class VlashParams:
    # TODO:
    pass


@dataclass
class TrainTimeRTCParams:
    # TODO:
    pass


# message types shared between client and server
@dataclass(frozen=True)
class InferRequest:
    robot_id: str
    observation: dict
    observation_step: int
    action_start_step: int
    request_timestamp: float
    deadline: float
    execution_horizon: int
    infer_type: InferType
    params: Optional[Union[RTCParams, VlashParams, TrainTimeRTCParams]] = None
    noise: Optional[Float[np.ndarray, "action_horizon noise_dim"]] = None
    type: Literal["infer"] = "infer"

    def __post_init__(self) -> None:
        if isinstance(self.infer_type, str):
            object.__setattr__(self, "infer_type", InferType(self.infer_type))

        if isinstance(self.params, dict):
            if self.infer_type == InferType.INFERENCE_TIME_RTC:
                object.__setattr__(self, "params", RTCParams(**self.params))


@dataclass(frozen=True)
class ResetRequest:
    robot_id: str
    type: Literal["reset"] = "reset"


@dataclass(frozen=True)
class InferResponse:
    robot_id: str
    request_id: int  # for routing response to correct connection
    observation_step: int  # from request
    action_start_step: int  # from request
    request_timestamp: float  # from request
    actions: Float[np.ndarray, "1 action_horizon action_dim"]  # TODO: check the type on this
    execution_horizon: int
    noise: Optional[Float[np.ndarray, "action_horizon noise_dim"]] = None
    # Lifecycle timestamps (server fields use server clock; receive_time_server uses client SyncedClock):
    server_arrival_time: float = 0.0  # WS: when observation arrived (server clock)
    inference_start_time: float = 0.0  # GPU: before infer_batch (server clock)
    inference_end_time: float = 0.0  # GPU: after infer_batch (server clock)
    server_send_time: float = 0.0  # WS: just before websocket.send_bytes() (server clock)
    receive_time_server: float = 0.0  # client: when response was received, adjusted to server clock


@dataclass(frozen=True)
class ResponseAck:
    request_id: int  # matches InferResponse.request_id
    receive_time: float  # client receipt time, adjusted to server clock via SyncedClock.now_server()
    execution_start_step: int  # client step when new chunk became available
    first_executed_index: int = 0  # index within chunk where actual execution started
    type: Literal["ack"] = "ack"


@dataclass(frozen=True)
class EpisodeStart:
    task_suite_name: str
    task_id: int
    episode_idx: int
    max_episode_steps: int
    task_language: str
    type: Literal["episode_start"] = "episode_start"


@dataclass(frozen=True)
class EpisodeStep:
    type: Literal["episode_step"] = "episode_step"


@dataclass(frozen=True)
class EpisodeEnd:
    task_suite_name: str
    task_id: int
    episode_idx: int
    success: bool
    duration_s: float
    steps_taken: int
    type: Literal["episode_end"] = "episode_end"


@dataclass(frozen=True)
class ConnectRequest:
    robot_id: str
    control_hz: float
    type: Literal["connect"] = "connect"


@dataclass(frozen=True)
class ConnectResponse:
    type: Literal["connect_response"] = "connect_response"


@dataclass(frozen=True)
class ClockSyncPing:
    client_timestamp: float
    type: Literal["clock_sync_ping"] = "clock_sync_ping"


@dataclass(frozen=True)
class ClockSyncPong:
    client_timestamp: float  # echoed from ClockSyncPing
    server_receive_time: float
    server_send_time: float
    type: Literal["clock_sync_pong"] = "clock_sync_pong"


@dataclass(frozen=True)
class WarmupPing:
    client_timestamp: float
    payload: bytes  # dummy bytes, same size as a typical packed InferRequest
    type: Literal["warmup_ping"] = "warmup_ping"


@dataclass(frozen=True)
class WarmupPong:
    client_timestamp: float  # echoed from WarmupPing
    server_receive_time: float
    server_send_time: float
    payload: bytes  # dummy bytes, same size as a typical packed InferResponse
    type: Literal["warmup_pong"] = "warmup_pong"


@dataclass(frozen=True)
class WarmupAck:
    server_send_time: float  # echoed from WarmupPong, so server can compute delivery latency
    client_receive_time: float
    type: Literal["warmup_ack"] = "warmup_ack"
