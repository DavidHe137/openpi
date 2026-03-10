from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.client import BidirectionalWebsocket
from typing_extensions import override


class SyncBroker(ActionChunkBroker):
    """Streams observations continuously but gates server re-inference to the full execution horizon.

    The server will not re-infer until the previous chunk has been fully executed, equivalent to
    the original synchronous one-at-a-time behavior but without blocking the client.
    """

    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
        min_execution_horizon: int = 0,
    ):
        super().__init__(
            ws_client=ws_client, control_hz=control_hz, realtime=realtime, min_execution_horizon=min_execution_horizon
        )
        assert self._min_execution_horizon == self._ws_client.server_metadata.action_horizon

    @override
    def _infer(self, obs: Observation) -> None:
        if len(self._action_queue) > 0:
            return

        self._ws_client.send(obs, self.deadline, self._action_step, min_execution_horizon=self._min_execution_horizon)
