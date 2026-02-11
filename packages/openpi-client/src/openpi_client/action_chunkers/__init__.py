from dataclasses import dataclass
from enum import Enum
from typing import Type

from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.action_chunkers.sync import SyncBroker
from openpi_client.action_chunkers.naive_async import NaiveAsyncBroker
from openpi_client.action_chunkers.rtc import InferenceTimeRTCBroker


@dataclass
class BrokerConfig:
    """Configuration for SyncBroker."""

    ws_client: _websocket_client_policy.BidirectionalWebsocket
    control_hz: int


# Mappings outside the enum to avoid conflicts
_CLASS_MAPPING = {
    "sync": SyncBroker,
    "rtc": InferenceTimeRTCBroker,
    "naive_async": NaiveAsyncBroker,
    # "temporal_ensembling": TemporalEnsemblingBroker,
    # "vlash": VLashBroker,
}


class ActionChunkBrokerType(Enum):
    SYNC = "sync"
    NAIVE_ASYNC = "naive_async"
    RTC = "rtc"
    # TODO: naive_async, temporal_ensembling, vlash

    def get_class(self) -> Type[ActionChunkBroker]:
        return _CLASS_MAPPING[self.value]

    def create(self, config) -> ActionChunkBroker:
        """Create broker from a config dataclass."""
        return self.get_class()(**vars(config))

    @classmethod
    def from_string(cls, value: str) -> "ActionChunkBrokerType":
        """Get enum member by value."""
        return cls(value.lower())
