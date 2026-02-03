from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Optional, Union


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


@dataclass
class InferRequest:
    observation: dict
    infer_type: InferType
    params: Optional[Union[RTCParams, VlashParams, TrainTimeRTCParams]] = None
    return_debug_data: bool = False
    noise: Optional[np.ndarray] = None  # optional, noise for deterministic replay


@dataclass
class InferResponse:
    actions: np.ndarray
    debug_data: Optional[dict] = field(default=None)
