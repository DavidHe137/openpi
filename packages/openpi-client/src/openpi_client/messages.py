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
    type: Literal["infer"] = "infer"
    robot_id: str
    observation: dict
    start_step: int  # TODO: can maybe fold into observation after typing
    request_timestamp: float
    deadline: float
    infer_type: InferType
    params: Optional[Union[RTCParams, VlashParams, TrainTimeRTCParams]] = None
    noise: Optional[Float[np.ndarray, "action_horizon noise_dim"]] = None
    min_execution_horizon: int = 0  # minimum steps to execute before server will re-infer this robot


@dataclass(frozen=True)
class ResetRequest:
    type: Literal["reset"] = "reset"
    robot_id: str


@dataclass(frozen=True)
class InferResponse:
    robot_id: str
    request_id: int  # for routing response to correct connection
    start_step: int  # from request
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
    type: Literal["ack"] = "ack"
    request_id: int  # matches InferResponse.request_id
    receive_time: float  # time.time() on client at receipt
