import time

from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.client import BidirectionalWebsocket
from typing_extensions import override


class NaiveAsyncBroker(ActionChunkBroker):
    """Streams observations continuously, re-inferring once half the current chunk has been executed."""

    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
    ):
        super().__init__(ws_client=ws_client, control_hz=control_hz, realtime=realtime)
        self.reset()
        self._background_thread.start()

    @override
    def _infer(self, obs: Observation) -> None:
        chunk = self.current_action_chunk
        min_horizon = chunk.execution_horizon // 2 if chunk else self._ws_client.server_metadata.action_horizon // 2
        deadline = time.time() + len(self._action_queue) * self._step_duration
        self._ws_client.send(obs, deadline=deadline, min_execution_horizon=min_horizon)
