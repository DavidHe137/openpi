import time
from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client import websocket_client_policy as _websocket_client_policy
from typing_extensions import override
from collections import deque


class InferenceTimeRTCBroker(ActionChunkBroker):
    """Wraps a policy to return action chunks asynchronously.

    The policy is called synchronously in the background thread whenever the current action chunk is exhausted.
    """

    def __init__(
        self,
        ws_client: _websocket_client_policy.BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
        s_min: int = 5,
        d_init: int = 3,
        delay_buffer_size: int = 10,
    ):
        """
        Args:
            ws_client: the websocket client to use for inference
            control_hz: the control frequency of the environment
            realtime: whether to run in realtime mode, setting this False essentially means inference latency is 0
            s_min: the minimum number of steps to wait before sending an inference request
            d_init: the initial delay in seconds
            delay_buffer_size: the size of the delay buffer
        """
        super().__init__(ws_client=ws_client, control_hz=control_hz, realtime=realtime)
        self._s_min = s_min
        self._d_init = d_init
        self._delay_buffer_size = delay_buffer_size

        self.reset()
        self._background_thread.start()

    @override
    def _infer(self, obs: Observation) -> None:
        # NOTE: server will store previous actions + handle s logic
        deadline = time.time() + len(self._action_queue) * self._step_duration
        estimated_delay = max(self._delays)
        self._ws_client.send(
            obs,
            deadline=deadline,
            use_rtc=True,
            s_param=self._s_min,
            d_param=estimated_delay,
        )

    @override
    def reset(self) -> None:
        with self._lock:
            self._action_queue.clear()
            self._action_chunks = []
            self._delays = deque([self._d_init], maxlen=self._delay_buffer_size)
