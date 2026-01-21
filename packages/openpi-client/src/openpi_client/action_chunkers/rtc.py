import math
from typing import Dict, List

import threading
import time
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from examples.libero.schemas import ActionChunk
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client import websocket_client_policy as _websocket_client_policy
from collections import deque
import numpy as np


class InferenceTimeRTCBroker(ActionChunkBroker, _base_policy.BasePolicy):
    """Wraps a policy to return action chunks with inference time RTC support.

    The policy is called synchronously in the background thread whenever the current action chunk is exhausted.
    """

    def __init__(
        self,
        policy: _websocket_client_policy.WebsocketClientPolicy,
        action_horizon: int,
        return_debug_data: bool = False,
        s_min: int = 5,
        d_init: int = 3,
        delay_buffer_size: int = 10,
        step_duration: float = 0.05,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._return_debug_data = return_debug_data
        self._obs: Dict = {}
        self._cur_step: int = -1

        self._action_chunks: List[ActionChunk] = []
        self._action_index: int = 0

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._infer_thread = threading.Thread(target=self._background_infer, daemon=True)
        self._infer_thread.start()

        self._s_min = s_min
        self._d_init = d_init
        self._delays: deque[int] = deque([self._d_init], maxlen=delay_buffer_size)
        self._step_duration = step_duration

    def _infer(
        self,
        obs: Dict,
        infer_step: int,
        steps_since_last_inference: int,
        estimated_delay: int,
        prev_action: np.ndarray,
    ) -> ActionChunk:
        request_timestamp = time.time()
        response = self._policy.infer(
            obs,
            use_rtc=True,
            s_param=steps_since_last_inference,
            d_param=estimated_delay,
            prev_action=prev_action,
            return_debug_data=self._return_debug_data,
        )
        actions = response["actions"]
        debug_data = response.get("debug_data", {})
        if debug_data is None:
            debug_data = {}
        if not isinstance(debug_data, dict):
            debug_data = {"_server_debug_data": debug_data}

        # When debug data saving is enabled, persist RTC params needed for deterministic replay.
        # This is consumed by `examples/libero/replay_debug_data.py --use_rtc`.
        if self._return_debug_data:
            debug_data["rtc"] = {
                "prev_action": np.asarray(prev_action),
                "s_param": int(steps_since_last_inference),
                "d_param": int(estimated_delay),
            }
        response_timestamp = time.time()

        return ActionChunk(
            actions=actions,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
            start_step=infer_step,
            debug_data=debug_data,
        )

    def _convert_latency_to_delay(self, latency: float) -> int:
        """
        Convert latency (in seconds) to delay (in steps).
        """
        return math.ceil(latency / self._step_duration)

    def _background_infer(self):
        while True:
            with self._condition:
                self._condition.wait()

                obs = self._obs
                infer_step = self._cur_step
                steps_since_last_inference = self._action_index
                estimated_delay = max(self._delays)
                prev_action = self.current_action_chunk.actions

            action_chunk = self._infer(obs, infer_step, steps_since_last_inference, estimated_delay, prev_action)

            with self._condition:
                self._action_chunks.append(action_chunk)
                self._delays.append(self._convert_latency_to_delay(action_chunk.latency))
                self._condition.notify()
                self._action_index -= steps_since_last_inference

    def _create_null_action(self) -> List[float]:
        last_action = self.current_action_chunk.actions[-1].copy()
        last_action[:-1] = 0.0
        return last_action

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._condition:
            self._obs = obs
            self._cur_step = obs["step"]

            # Assume no latency for step 0, so we wait until we have an action chunk
            if len(self._action_chunks) == 0:
                assert self._cur_step == 0, "First inference should be for step 0"
                # TODO: don't hardcode the action dimension
                action_chunk = self._infer(
                    obs, self._cur_step, 0, 0, np.zeros((self._action_horizon, 7))
                )
                self._action_chunks.append(action_chunk)

            if self._action_index >= self._s_min:
                self._condition.notify()

            if self._action_index >= self.current_action_chunk.chunk_length:
                action = self._create_null_action()
                action_index = -1
            else:
                action = self.current_action_chunk.get_action(self._action_index)
                action_index = self._action_index
                self._action_index += 1

            results = {
                "actions": action,
                "action_chunk_index": len(self._action_chunks) - 1,
                "action_chunk_current_step": action_index,
            }

        return results

    @override
    def reset(self) -> None:
        with self._lock:
            self._policy.reset()
            self._action_chunks = []
            self._cur_step = -1
            self._action_index = 0
