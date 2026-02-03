from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional, Union
from jaxtyping import Float


# TODO: merge with broker types
class InferType(Enum):
    SYNC = "sync"
    INFERENCE_TIME_RTC = "inference_time_rtc"
    TRAIN_TIME_RTC = "train_time_rtc"
    VLASH = "vlash"


@dataclass
class RTCParams:
    prev_action: np.ndarray
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
    start_step: int  # TODO: can maybe fold into observation after typing
    request_timestamp: float
    deadline: float
    infer_type: InferType
    params: Optional[Union[RTCParams, VlashParams, TrainTimeRTCParams]] = None
    noise: Optional[Float[np.ndarray, "action_horizon noise_dim"]] = None


@dataclass(frozen=True)
class InferResponse:
    robot_id: str
    request_id: int  # for routing response to correct connection
    start_step: int  # from request
    request_timestamp: float  # from request
    actions: Float[np.ndarray, "1 action_horizon action_dim"]  # TODO: check the type on this
    execution_horizon: int
    noise: Optional[Float[np.ndarray, "action_horizon noise_dim"]] = None
