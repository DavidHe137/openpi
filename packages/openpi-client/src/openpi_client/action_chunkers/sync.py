from collections import deque
from typing import List

import threading
import time
from openpi_client.schemas import ActionChunk, Action, Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client import websocket_client_policy as _websocket_client_policy


# FIXME: Saver uses action_chunks, but the envy is not clear and it's easy to remove it from this class
# TODO: add debug data, though I think there should be a cleaner way to do this
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
        return_debug_data: bool = False,  # TODO: add debug data
    ):
        self._ws_client = ws_client

        self._action_queue: deque[Action] = deque()
        self._action_chunks: List[ActionChunk] = []
        self._step_duration = 1 / control_hz

        self._lock = threading.Lock()
        self._background_thread = threading.Thread(target=self._receive_actions, daemon=True)
        self._background_thread.start()
        self._sent_request = False

    def _infer(self, obs: Observation) -> None:
        deadline = time.time() + len(self._action_queue) * self._step_duration
        self._ws_client.send(obs, deadline=deadline)
        self._sent_request = True

    def _receive_actions(self):
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

    def _create_null_action(self, obs: Observation) -> Action:
        # FIXME: hardcoded, should move this outside of this class
        import numpy as np

        action = np.zeros(7)
        action[-1] = self.current_action_chunk.get_action(-1)[-1] if self.current_action_chunk is not None else 0.0

        return Action(
            step=obs.step,
            action=action,
            action_chunk_index=None,
            index_in_chunk=None,
        )

    def _should_infer(self) -> bool:
        return len(self._action_queue) == 0 and self._sent_request is False

    def infer(self, obs: Observation) -> Action:
        with self._lock:
            action = self._action_queue.popleft() if self._action_queue else self._create_null_action(obs)

            if self._should_infer():
                self._infer(obs)

        return action

    def reset(self) -> None:
        with self._lock:
            self._action_queue.clear()
            self._action_chunks = []
            self._sent_request = False
