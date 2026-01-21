from typing import Dict, List

import threading
import time
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from examples.libero.schemas import ActionChunk
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker


class SyncBroker(ActionChunkBroker, _base_policy.BasePolicy):
    """Wraps a policy to return action chunks asynchronously.

    The policy is called synchronously in the background thread whenever the current action chunk is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, return_debug_data: bool = False):
        self._policy = policy
        self._action_horizon = action_horizon
        self._return_debug_data = return_debug_data
        self._obs: Dict = {}
        self._cur_step: int = -1

        self._action_chunks: List[ActionChunk] = []
        self._action_index: int = -1

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._infer_thread = threading.Thread(target=self._background_infer, daemon=True)
        self._infer_thread.start()

    def _infer(self, obs: Dict, infer_step: int) -> ActionChunk:
        request_timestamp = time.time()
        response = self._policy.infer(obs, return_debug_data=self._return_debug_data)
        actions = response["actions"]
        debug_data = response.get("debug_data", {})
        response_timestamp = time.time()

        return ActionChunk(
            actions=actions,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
            start_step=infer_step,
            debug_data=debug_data,
        )

    def _background_infer(self):
        while True:
            with self._condition:
                self._condition.wait()

                obs = self._obs
                infer_step = self._cur_step

            # might get pre-empted here, but that's probably okay
            action_chunk = self._infer(obs, infer_step)

            with self._condition:
                self._action_chunks.append(action_chunk)
                self._action_index = -1

    def _create_null_action(self) -> List[float]:
        last_action = self.current_action_chunk.actions[-1].copy()
        last_action[:-1] = 0.0
        return last_action

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._condition:
            self._obs = obs
            self._cur_step = obs["step"]
            self._action_index += 1

            # Assume no latency for step 0, so we wait until we have an action chunk
            if len(self._action_chunks) == 0:
                assert self._cur_step == 0, "First inference should be for step 0"
                action_chunk = self._infer(obs, self._cur_step)
                self._action_chunks.append(action_chunk)

            if self._action_index == self.current_action_chunk.chunk_length:
                self._condition.notify()

            if self._action_index >= self.current_action_chunk.chunk_length:
                action = self._create_null_action()
                action_index = -1
            else:
                action = self.current_action_chunk.get_action(self._action_index)
                action_index = self._action_index

            results = {
                "actions": action,
                "action_chunk_index": len(self._action_chunks) - 1,
                "action_chunk_current_step": action_index,
            }

        return results

    @override
    def reset(self) -> None:
        with self._condition:
            self._policy.reset()
            self._action_chunks = []
            self._cur_step = -1
            self._action_index = -1


class ReplanSyncBroker(ActionChunkBroker, _base_policy.BasePolicy):
    """Sync-style broker that replans after a fixed number of actions.

    A new inference call to the inner policy is made after `replan_after` actions
    have been taken from the current chunk. The new chunk is used as soon as it
    arrives, even if the previous chunk was not exhausted.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        return_debug_data: bool = False,
        replan_after: int = 10,
    ):
        if replan_after < 1:
            raise ValueError("replan_after must be >= 1")
        if replan_after > action_horizon:
            raise ValueError(
                f"replan_after ({replan_after}) must be <= action_horizon ({action_horizon})"
            )
        self._policy = policy
        self._action_horizon = action_horizon
        self._return_debug_data = return_debug_data
        self._replan_after = replan_after
        self._obs: Dict = {}
        self._cur_step: int = -1

        self._action_chunks: List[ActionChunk] = []
        self._action_index: int = -1
        self._steps_since_last_request: int = 0

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._infer_thread = threading.Thread(target=self._background_infer, daemon=True)
        self._infer_thread.start()

    def _infer(self, obs: Dict, infer_step: int, steps_since_last_request: int) -> ActionChunk:
        request_timestamp = time.time()
        response = self._policy.infer(obs, return_debug_data=self._return_debug_data)
        actions = response["actions"]
        debug_data = response.get("debug_data", {})
        if debug_data is None:
            debug_data = {}
        if not isinstance(debug_data, dict):
            debug_data = {"_server_debug_data": debug_data}
        if self._return_debug_data:
            debug_data["replan_sync"] = {
                "replan_after": int(self._replan_after),
                "steps_since_last_request": int(steps_since_last_request),
                "trigger_step": int(infer_step),
            }
        response_timestamp = time.time()

        return ActionChunk(
            actions=actions,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
            start_step=infer_step,
            debug_data=debug_data,
        )

    def _background_infer(self):
        while True:
            with self._condition:
                self._condition.wait()

                obs = self._obs
                infer_step = self._cur_step
                steps_since_last_request = self._steps_since_last_request

            action_chunk = self._infer(obs, infer_step, steps_since_last_request)

            with self._condition:
                self._action_chunks.append(action_chunk)
                self._action_index = -1

    def _create_null_action(self) -> List[float]:
        last_action = self.current_action_chunk.actions[-1].copy()
        last_action[:-1] = 0.0
        return last_action

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._condition:
            self._obs = obs
            self._cur_step = obs["step"]
            self._action_index += 1

            # Assume no latency for step 0, so we wait until we have an action chunk
            if len(self._action_chunks) == 0:
                assert self._cur_step == 0, "First inference should be for step 0"
                action_chunk = self._infer(obs, self._cur_step, 0)
                self._action_chunks.append(action_chunk)

            if self._action_index == self._replan_after:
                self._steps_since_last_request = self._action_index
                self._condition.notify()

            if self._action_index >= self.current_action_chunk.chunk_length:
                action = self._create_null_action()
                action_index = -1
            else:
                action = self.current_action_chunk.get_action(self._action_index)
                action_index = self._action_index

            results = {
                "actions": action,
                "action_chunk_index": len(self._action_chunks) - 1,
                "action_chunk_current_step": action_index,
            }

        return results

    @override
    def reset(self) -> None:
        with self._condition:
            self._policy.reset()
            self._action_chunks = []
            self._cur_step = -1
            self._action_index = -1
            self._steps_since_last_request = 0
