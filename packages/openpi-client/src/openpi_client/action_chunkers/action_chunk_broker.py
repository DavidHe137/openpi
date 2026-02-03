from typing import List
from typing import Optional
from openpi_client.schemas import ActionChunk
from abc import ABC
from collections import deque
from openpi_client.schemas import Action, Observation
import threading


class ActionChunkBroker(ABC):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, control_hz: int, realtime: bool = True) -> None:
        self._action_queue: deque[Action] = deque()
        self._action_chunks: List[ActionChunk] = []

        self._step_duration = 1 / control_hz
        self._realtime = realtime

        self._lock = threading.Lock()
        self._background_thread = threading.Thread(target=self._receive_actions, daemon=True)

    def infer(self, obs: Observation) -> Action:
        with self._lock:
            action = self._action_queue.popleft() if self._action_queue else self._create_null_action(obs)

            if self._should_infer():
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

    def _receive_actions(self):
        pass

    def _should_infer(self) -> bool:
        pass

    def _infer(self, obs: Observation) -> None:
        pass

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
