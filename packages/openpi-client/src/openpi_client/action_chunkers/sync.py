from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client import websocket_client_policy as _websocket_client_policy
from typing_extensions import override


class SyncBroker(ActionChunkBroker):
    """Wraps a policy to return action chunks asynchronously.

    The policy is called synchronously in the background thread whenever the current action chunk is exhausted.
    """

    def __init__(
        self,
        ws_client: _websocket_client_policy.BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
    ):
        """
        Args:
            ws_client: the websocket client to use for inference
            control_hz: the control frequency of the environment
            realtime: whether to run in realtime mode, setting this False essentially means inference latency is 0
        """
        super().__init__(ws_client=ws_client, control_hz=control_hz, realtime=realtime)
        self.reset()
        self._background_thread.start()

    @override
    def _infer(self, obs: Observation) -> None:
        if len(self._action_queue) > 0 or self._sent_request:
            return

        super()._infer(obs)
        self._sent_request = True

    @override
    def _receive_actions(self) -> None:
        while True:
            action_chunk = self._ws_client.receive()
            self._action_chunks.append(action_chunk)
            self._update_action_queue(action_chunk)
            self._sent_request = False

    @override
    def reset(self) -> None:
        with self._lock:
            self._action_queue.clear()
            self._action_chunks = []
            self._ws_client.reset()
            self._sent_request = False
