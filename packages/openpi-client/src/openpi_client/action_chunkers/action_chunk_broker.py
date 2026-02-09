import time
from typing import List
from typing import Optional
from openpi_client.schemas import ActionChunk
from abc import ABC
from collections import deque
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.schemas import Action, Observation
import threading


# FIXME: Saver uses action_chunks, but the envy is not clear and it's easy to remove it from this class
# NOTE: use concurrent.futures to infer in background if this takes too long
# TODO: base policy class that lives on server should be different from policy that lives on client
class ActionChunkBroker(ABC):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(
        self, ws_client: _websocket_client_policy.BidirectionalWebsocket, control_hz: int, realtime: bool = True
    ) -> None:
        self._ws_client = ws_client
        self._action_queue: deque[Action] = deque()
        self._action_chunks: List[ActionChunk] = []

        self._step_duration = 1 / control_hz
        self._realtime = realtime

        self._lock = threading.Lock()
        self._background_thread = threading.Thread(target=self._receive_actions, daemon=True)

    def infer(self, obs: Observation) -> Action:
        """Client continuously streams observations to the server."""
        with self._lock:
            action = self._action_queue.popleft() if self._action_queue else self._create_null_action(obs)

            self._infer(obs)

        return action

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

    def _receive_actions(self) -> None:
        while True:
            action_chunk = self._ws_client.receive()
            self._action_chunks.append(action_chunk)
            self._update_action_queue(action_chunk)

    def _update_action_queue(self, action_chunk: ActionChunk):
        with self._lock:
            while self._action_queue and self._action_queue[-1].step >= action_chunk.start_step:
                self._action_queue.pop()

            # assumes that pausing is preferable to executing actions past the execution horizon
            self._action_queue.extend(
                Action(
                    step=action_chunk.start_step + i,
                    action=action_chunk.get_action(i),
                    action_chunk_index=len(self._action_chunks) - 1,
                    index_in_chunk=i,
                )
                for i in range(action_chunk.execution_horizon)
            )

    def _infer(self, obs: Observation) -> None:
        deadline = time.time() + len(self._action_queue) * self._step_duration
        self._ws_client.send(obs, deadline=deadline)

    def reset(self) -> None:
        with self._lock:
            self._action_queue.clear()
            self._action_chunks = []

    @property
    def action_chunks(self) -> List[ActionChunk]:
        assert all(chunk.start_step >= 0 for chunk in self._action_chunks), (
            "An action chunk did not have a start step set"
        )
        return self._action_chunks

    @property
    def current_action_chunk(self) -> Optional[ActionChunk]:
        return self._action_chunks[-1] if self._action_chunks else None
