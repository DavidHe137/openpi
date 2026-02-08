import time
from openpi_client.schemas import Action, Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client import websocket_client_policy as _websocket_client_policy
from typing_extensions import override


# FIXME: Saver uses action_chunks, but the envy is not clear and it's easy to remove it from this class
# NOTE: use concurrent.futures to infer in background if this takes too long
# TODO: base policy class that lives on server should be different from policy that lives on client
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
        super().__init__(control_hz=control_hz, realtime=realtime)

        self._ws_client = ws_client
        self._sent_request = False

        self.reset()
        self._background_thread.start()

    @override
    def _infer(self, obs: Observation) -> None:
        deadline = time.time() + len(self._action_queue) * self._step_duration
        self._ws_client.send(obs, deadline=deadline)
        self._sent_request = True

    @override
    def _receive_actions(self) -> None:
        while True:
            action_chunk = self._ws_client.receive()

            with self._lock:
                self._action_chunks.append(action_chunk)
                while self._action_queue and self._action_queue[-1].step >= action_chunk.start_step:
                    self._action_queue.pop()

                # assumes that pausing is preferable to exeuting actions past the execution horizon
                self._action_queue.extend(
                    Action(
                        step=action_chunk.start_step + i,
                        action=action_chunk.get_action(i),
                        action_chunk_index=len(self._action_chunks) - 1,
                        index_in_chunk=i,
                    )
                    for i in range(action_chunk.execution_horizon)
                )
                self._sent_request = False

    @override
    def reset(self) -> None:
        with self._lock:
            self._action_queue.clear()
            self._action_chunks = []
            self._sent_request = False
