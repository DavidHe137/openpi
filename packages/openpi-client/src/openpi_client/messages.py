from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Literal, Optional, Union
from jaxtyping import Float


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
    min_execution_horizon: int
    infer_type: InferType
    params: Optional[Union[RTCParams, VlashParams, TrainTimeRTCParams]] = None
    noise: Optional[Float[np.ndarray, "action_horizon noise_dim"]] = None
    type: Literal["infer"] = "infer"


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
    # Lifecycle timestamps (filled by server, all time.time()):
    server_arrival_time: float = 0.0  # WS: when observation arrived
    inference_start_time: float = 0.0  # GPU: before infer_batch
    inference_end_time: float = 0.0  # GPU: after infer_batch
    server_send_time: float = 0.0  # WS: just before websocket.send_bytes()


@dataclass(frozen=True)
class ResponseAck:
    request_id: int  # matches InferResponse.request_id
    receive_time: float  # time.time() on client at receipt
    execution_start_step: int  # client step when new chunk became available
    type: Literal["ack"] = "ack"


@dataclass(frozen=True)
class TaskResult:
    task_suite_name: str
    task_id: int
    episode_idx: int
    success: bool
    duration_s: float
    steps_taken: int
    task_language: Optional[str] = None
    total_episodes: Optional[int] = None
    max_episode_steps: Optional[int] = None
    max_duration_s: Optional[float] = None
    type: Literal["task_result"] = "task_result"


@dataclass(frozen=True)
class TaskProgress:
    task_suite_name: str
    task_id: int
    episode_idx: int
    current_step: int
    max_episode_steps: int
    task_language: Optional[str] = None
    total_episodes: Optional[int] = None
    type: Literal["task_progress"] = "task_progress"


@dataclass(frozen=True)
class ConnectRequest:
    robot_id: str
    control_hz: float
    type: Literal["connect"] = "connect"


@dataclass(frozen=True)
class ConnectResponse:
    type: Literal["connect_response"] = "connect_response"


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
