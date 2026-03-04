import time
from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.client import BidirectionalWebsocket
from typing_extensions import override
from collections import deque
import numpy as np


class InferenceTimeRTCBroker(ActionChunkBroker):
    """Wraps a policy to return action chunks asynchronously.

    The policy is called synchronously in the background thread whenever the current action chunk is exhausted.
    """

    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
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
        self._last_request_time = None
        self._delays = deque([4], maxlen=self._delay_buffer_size)

        self.reset()
        self._background_thread.start()

    @override
    def _infer(self, obs: Observation) -> None:
        # NOTE: server will store previous actions + handle s logic
        # Record when we send the request for delay tracking
        self._last_request_time = time.time()

        deadline = time.time() + len(self._action_queue) * self._step_duration
        estimated_delay = max(self._delays)
        current_step = obs.step
        previous_step = self.current_action_chunk.start_step if self.current_action_chunk else None
        steps_since_last_response = current_step - previous_step if previous_step is not None else 0

        chunk = self.current_action_chunk
        min_horizon = chunk.execution_horizon // 2 if chunk else self._ws_client.server_metadata.action_horizon // 2
        self._ws_client.send(
            obs,
            deadline=deadline,
            use_rtc=True,
            prev_action=self.current_action_chunk.actions
            if self.current_action_chunk
            else np.zeros((50, 7)),  # FIXME: hardcoded
            s_param=steps_since_last_response,
            d_param=estimated_delay,
            min_execution_horizon=min_horizon,
        )

    @override
    def _receive_actions(self) -> None:
        """Receive actions and track actual delays per Algorithm 1 line 22."""
        while True:
            action_chunk = self._ws_client.receive()
            receive_time = time.time()

            # Calculate actual observed delay: time from request to response
            # This corresponds to 't' in Algorithm 1 line 22: "enqueue t onto Q"
            if self._last_request_time is not None:
                delay_seconds = receive_time - self._last_request_time
                delay_steps = int(delay_seconds / self._step_duration)  # Convert to steps

                with self._lock:
                    self._delays.append(delay_steps)

            self._action_chunks.append(action_chunk)
            self._update_action_queue(action_chunk)

    @override
    def reset(self) -> None:
        super().reset()
        # Initialize per Algorithm 1: d_init = 4 steps (~200ms at 20Hz)
        self._delays = deque([4], maxlen=self._delay_buffer_size)
        self._last_request_time = None
