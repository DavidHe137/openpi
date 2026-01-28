import math
from collections import deque
from typing import List

import threading
import time
import numpy as np

from openpi_client.schemas import ActionChunk, Action, Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client import websocket_client_policy as _websocket_client_policy


class InferenceTimeRTCBroker(ActionChunkBroker):
    """Wraps a policy to return action chunks with inference time RTC support.

    The policy is called asynchronously in the background thread whenever the action queue is depleted.
    """

    def __init__(
        self,
        ws_client: _websocket_client_policy.BidirectionalWebsocket,
        control_hz: int,
        s_min: int = 5,
        d_init: int = 3,
        delay_buffer_size: int = 10,
    ):
        self._ws_client = ws_client

        self._action_queue: deque[Action] = deque()
        self._action_chunks: List[ActionChunk] = []
        self._step_duration = 1 / control_hz

        self._lock = threading.Lock()
        self._background_thread = threading.Thread(target=self._receive_actions, daemon=True)
        self._background_thread.start()

        self._s_min = s_min
        self._d_init = d_init
        self._delays: deque[int] = deque([self._d_init], maxlen=delay_buffer_size)
        self._steps_since_last_inference: int = 0
        self._sent_request = False

    def _convert_latency_to_delay(self, latency: float) -> int:
        """Convert latency (in seconds) to delay (in steps)."""
        return math.ceil(latency / self._step_duration)

    def _infer(self, obs: Observation) -> None:
        deadline = time.time() + len(self._action_queue) * self._step_duration
        estimated_delay = max(self._delays)
        prev_action = self.current_action_chunk.actions if self.current_action_chunk is not None else np.zeros((10, 7))

        self._ws_client.send(
            obs,
            deadline=deadline,
            use_rtc=True,
            prev_action=prev_action,
            s_param=self._steps_since_last_inference,
            d_param=estimated_delay,
        )
        self._sent_request = True

    def _receive_actions(self):
        while True:
            action_chunk = self._ws_client.receive()

            with self._lock:
                self._action_chunks.append(action_chunk)
                self._delays.append(self._convert_latency_to_delay(action_chunk.latency))

                while self._action_queue and self._action_queue[-1].step >= action_chunk.start_step:
                    self._action_queue.pop()

                self._action_queue.extend(
                    Action(
                        step=action_chunk.start_step + i,
                        action=action_chunk.get_action(i),
                        action_chunk_index=len(self._action_chunks) - 1,
                        index_in_chunk=i,
                    )
                    for i in range(action_chunk.execution_horizon)
                )

                self._steps_since_last_inference = 0
                self._sent_request = False

    def _create_null_action(self, obs: Observation) -> Action:
        # robot state and actions are in the same representation (x, y, z, qw, qx, qy, gripper)
        # FIXME: hardcoded, should move this outside of this class
        action = obs.state[:7].copy()
        action[-1] = self.current_action_chunk.get_action(-1)[-1] if self.current_action_chunk is not None else 0.0

        return Action(
            step=obs.step,
            action=action,
            action_chunk_index=None,
            index_in_chunk=None,
        )

    def _should_infer(self) -> bool:
        return len(self._action_queue) <= self._s_min and not self._sent_request

    def infer(self, obs: Observation) -> Action:
        with self._lock:
            action = self._action_queue.popleft() if self._action_queue else self._create_null_action(obs)
            self._steps_since_last_inference += 1

            if self._should_infer():
                self._infer(obs)

        return action

    def reset(self) -> None:
        with self._lock:
            self._action_queue.clear()
            self._action_chunks = []
            self._steps_since_last_inference = 0
            self._delays = deque([self._d_init], maxlen=self._delays.maxlen)
            self._sent_request = False
