from dataclasses import dataclass
from enum import Enum
from typing import Type

from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.action_chunkers.sync import SyncBroker
from openpi_client.action_chunkers.rtc import InferenceTimeRTCBroker


@dataclass
class SyncBrokerConfig:
    """Configuration for SyncBroker."""

    ws_client: _websocket_client_policy.BidirectionalWebsocket
    control_hz: int
    return_debug_data: bool = False


@dataclass
class RTCBrokerConfig(SyncBrokerConfig):
    """Configuration for InferenceTimeRTCBroker."""

    s_min: int = 5
    d_init: int = 3


# Mappings outside the enum to avoid conflicts
_CLASS_MAPPING = {
    "sync": SyncBroker,
    "rtc": InferenceTimeRTCBroker,
    # TODO:
    # "naive_async": NaiveAsyncBroker,
    # "temporal_ensembling": TemporalEnsemblingBroker,
    # "vlash": VLashBroker,
}

_CONFIG_MAPPING = {
    "sync": SyncBrokerConfig,
    "rtc": RTCBrokerConfig,
    # TODO:
    # "naive_async": NaiveAsyncBrokerConfig,
    # "temporal_ensembling": TemporalEnsemblingBrokerConfig,
    # "vlash": VLashBrokerConfig,
}


class ActionChunkBrokerType(Enum):
    SYNC = "sync"
    RTC = "rtc"
    # TODO: naive_async, temporal_ensembling, vlash

    def get_class(self) -> Type[ActionChunkBroker]:
        return _CLASS_MAPPING[self.value]

    def get_config_class(self):
        """Get the config dataclass for this broker type."""
        return _CONFIG_MAPPING[self.value]

    def create(self, config) -> ActionChunkBroker:
        """Create broker from a config dataclass."""
        return self.get_class()(**vars(config))

    @classmethod
    def from_string(cls, value: str) -> "ActionChunkBrokerType":
        """Get enum member by value."""
        return cls(value.lower())
