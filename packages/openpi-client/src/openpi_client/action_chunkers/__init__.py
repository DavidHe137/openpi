from dataclasses import dataclass
from enum import Enum
from typing import Type

from openpi_client import base_policy as _base_policy
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.action_chunkers.sync import SyncBroker, ReplanSyncBroker
from openpi_client.action_chunkers.rtc import InferenceTimeRTCBroker


@dataclass
class SyncBrokerConfig:
    """Configuration for SyncBroker."""

    policy: _base_policy.BasePolicy
    action_horizon: int
    return_debug_data: bool = False


@dataclass
class ReplanSyncBrokerConfig(SyncBrokerConfig):
    """Configuration for ReplanSyncBroker."""

    replan_after: int = 10


@dataclass
class RTCBrokerConfig(SyncBrokerConfig):
    """Configuration for InferenceTimeRTCBroker."""

    s_min: int = 5
    d_init: int = 3


# Mappings outside the enum to avoid conflicts
_CLASS_MAPPING = {
    "sync": SyncBroker,
    "replan_sync": ReplanSyncBroker,
    "rtc": InferenceTimeRTCBroker,
    # TODO:
    # "naive_async": NaiveAsyncBroker,
    # "temporal_ensembling": TemporalEnsemblingBroker,
    # "vlash": VLashBroker,
}

_CONFIG_MAPPING = {
    "sync": SyncBrokerConfig,
    "replan_sync": ReplanSyncBrokerConfig,
    "rtc": RTCBrokerConfig,
    # TODO:
    # "naive_async": NaiveAsyncBrokerConfig,
    # "temporal_ensembling": TemporalEnsemblingBrokerConfig,
    # "vlash": VLashBrokerConfig,
}


class ActionChunkBrokerType(Enum):
    SYNC = "sync"
    REPLAN_SYNC = "replan_sync"
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
