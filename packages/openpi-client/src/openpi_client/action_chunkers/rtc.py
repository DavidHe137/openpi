import time
from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client import websocket_client_policy as _websocket_client_policy
from typing_extensions import override
from collections import deque
import numpy as np


class InferenceTimeRTCBroker(ActionChunkBroker):
    """Wraps a policy to return action chunks asynchronously.

    The policy is called synchronously in the background thread whenever the current action chunk is exhausted.
    """

    def __init__(
        self,
        ws_client: _websocket_client_policy.BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
        delay_buffer_size: int = 10,
    ):
        """
        Args:
            ws_client: the websocket client to use for inference
            control_hz: the control frequency of the environment
            realtime: whether to run in realtime mode, setting this False essentially means inference latency is 0
            delay_buffer_size: the size of the delay buffer
        """
        super().__init__(ws_client=ws_client, control_hz=control_hz, realtime=realtime)
        self._delay_buffer_size = delay_buffer_size

        self.reset()
        self._background_thread.start()

    @override
    def _infer(self, obs: Observation) -> None:
        # NOTE: server will store previous actions + handle s logic
        deadline = time.time() + len(self._action_queue) * self._step_duration
        estimated_delay = max(self._delays)
        current_step = obs.step
        previous_step = self.current_action_chunk.start_step if self.current_action_chunk else None
        steps_since_last_response = current_step - previous_step if previous_step is not None else 0

        self._ws_client.send(
            obs,
            deadline=deadline,
            use_rtc=True,
            prev_action=self.current_action_chunk.actions
            if self.current_action_chunk
            else np.zeros((50, 7)),  # FIXME: hardcoded
            s_param=steps_since_last_response,
            d_param=estimated_delay,
        )

    @override
    def reset(self) -> None:
        with self._lock:
            self._action_queue.clear()
            self._action_chunks = []
            # FIXME: estimate delay from server
            self._delays = deque([3], maxlen=self._delay_buffer_size)
            self._ws_client.reset()
