import time
from typing import List
from typing import Optional
from typing import Protocol
from openpi_client.schemas import ActionChunk
from abc import ABC
from collections import deque
from openpi_client.client import BidirectionalWebsocket
from openpi_client.schemas import Action, Observation
import threading


class _StartupReleaseEvent(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...


# FIXME: Saver uses action_chunks, but the envy is not clear and it's easy to remove it from this class
# NOTE: use concurrent.futures to infer in background if this takes too long
# TODO: base policy class that lives on server should be different from policy that lives on client
# TODO: review and implement learnings from DRTC (https://jackvial.com/posts/distributed-real-time-chunking.html)
# FIXME: there is an off by one error for min-horizon, but it is fine for now since we are streaming requests
class ActionChunkBroker(ABC):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
        min_execution_horizon: int = 0,
        block_until_first_chunk: bool = True,
        startup_release_event: Optional[_StartupReleaseEvent] = None,
    ) -> None:
        self._ws_client = ws_client
        self._action_queue: deque[Action] = deque()
        self._action_chunks: List[ActionChunk] = []
        self._next_observation_step: int = 0  # next observation step to see
        self._next_action_step: int = 0  # next action step to execute

        self._step_duration = 1 / control_hz
        self._realtime = realtime
        self._min_execution_horizon = min_execution_horizon
        self._block_until_first_chunk = block_until_first_chunk
        self._received_first_chunk = False
        self._startup_release_event = startup_release_event

        self._prev_action: Action = self._create_null_action(-1)

        self._lock = threading.Lock()
        self._actions_available = threading.Condition(self._lock)
        self._actions_left_history: list[int] = []
        self._background_thread = threading.Thread(target=self._receive_actions, daemon=True)

        self.reset()
        self._background_thread.start()

    def infer(self, obs: Observation) -> Action:
        """Client continuously streams observations to the server."""
        with self._lock:
            # count actions left in queue before we pop the next action
            self._actions_left_history.append(len(self._action_queue))

            self._next_observation_step = obs.step + 1
            request_sent = False
            if self._action_queue:
                action = self._action_queue.popleft()
                self._next_action_step += 1
            else:
                # Startup block is released by either:
                # 1) this robot receiving its first action chunk, or
                # 2) a shared startup-release event set by another robot's first dispatch.
                if self._block_until_first_chunk and not self._startup_released():
                    self._infer(obs)
                    request_sent = True
                    while not self._action_queue and not self._startup_released():
                        # Poll with timeout so we can observe shared startup release
                        # without requiring a local chunk arrival notification.
                        self._actions_available.wait(timeout=0.01)
                if self._action_queue:
                    action = self._action_queue.popleft()
                    self._next_action_step += 1
                else:
                    action = self._create_null_action(obs.step)

            self._prev_action = action

            if not request_sent:
                self._infer(obs)

            return action

    def _create_null_action(self, observation_step: int) -> Action:
        # FIXME: hardcoded, should move this outside of this class
        import numpy as np

        action = np.zeros(7)
        action[-1] = self.current_action_chunk.get_action(-1)[-1] if self.current_action_chunk is not None else 0.0

        return Action(
            step=observation_step,
            action=action,
            action_chunk_index=None,
            index_in_chunk=None,
        )

    def _receive_actions(self) -> None:
        while True:
            infer_response = self._ws_client.receive()
            with self._lock:
                action_chunk = ActionChunk.from_infer_response(
                    infer_response=infer_response,
                    execution_start_step=self._next_observation_step,
                )
                # Discard chunks from a previous episode that arrive after reset.
                # After reset, _next_observation_step=0 and any legitimate first chunk
                # must have action_start_step near 0. A stale chunk from step ~500+
                # of the prior episode would contaminate the new episode's queue.
                if (
                    not self._received_first_chunk
                    and action_chunk.action_start_step
                    > self._next_observation_step + self._ws_client.server_metadata.action_horizon
                ):
                    continue
                self._action_chunks.append(action_chunk)
                self._update_action_queue(action_chunk)
                self._received_first_chunk = True
                if self._startup_release_event is not None:
                    self._startup_release_event.set()
                self._actions_available.notify_all()
                first_executed_index = max(0, self._next_action_step - action_chunk.action_start_step)
                self._ws_client.send_ack(
                    action_chunk.request_id,
                    action_chunk.response_timestamp,
                    action_chunk.execution_start_step,
                    first_executed_index,
                )

    def _update_action_queue(self, action_chunk: ActionChunk) -> None:
        while self._action_queue and self._action_queue[-1].step >= action_chunk.action_start_step:
            self._action_queue.pop()

        # assumes that pausing is preferable to executing actions past the execution horizon
        self._action_queue.extend(
            Action(
                step=action_chunk.action_start_step + i,
                action=action_chunk.get_action(i),
                action_chunk_index=len(self._action_chunks) - 1,
                index_in_chunk=i,
            )
            for i in range(action_chunk.execution_horizon)
            if action_chunk.action_start_step + i >= self._next_action_step
        )

    def _infer(self, obs: Observation) -> None:
        self._ws_client.send(
            obs,
            self.deadline,
            self._next_action_step,
            min_execution_horizon=self._min_execution_horizon,
        )

    def reset(self) -> None:
        with self._lock:
            self._next_observation_step = 0
            self._next_action_step = 0
            self._action_queue.clear()
            self._action_chunks = []
            self._actions_left_history = []
            self._received_first_chunk = False
            self._ws_client.reset()

    def _startup_released(self) -> bool:
        if self._received_first_chunk:
            return True
        if self._startup_release_event is None:
            return False
        return self._startup_release_event.is_set()

    @property
    def action_chunks(self) -> List[ActionChunk]:
        return self._action_chunks

    @property
    def current_action_chunk(self) -> Optional[ActionChunk]:
        return self._action_chunks[-1] if self._action_chunks else None

    @property
    def actions_left_history(self) -> List[int]:
        """Actions remaining in queue after each step (recorded inside the lock)."""
        return list(self._actions_left_history)

    @property
    def deadline(self) -> float:
        return time.time() + len(self._action_queue) * self._step_duration
