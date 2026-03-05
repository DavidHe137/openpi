import time
from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.client import BidirectionalWebsocket
from typing_extensions import override
from openpi_client import messages


# FIXME: basically identical to naive async
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

        self.reset()
        self._background_thread.start()

    @override
    def _infer(self, obs: Observation) -> None:
        deadline = time.time() + len(self._action_queue) * self._step_duration

        chunk = self.current_action_chunk
        min_horizon = chunk.execution_horizon // 2 if chunk else self._ws_client.server_metadata.action_horizon // 2
        self._ws_client.send(
            obs,
            deadline=deadline,
            infer_type=messages.InferType.INFERENCE_TIME_RTC,
            min_execution_horizon=min_horizon,
        )

    @override
    def _receive_actions(self) -> None:
        """Receive actions and track actual delays per Algorithm 1 line 22."""
        while True:
            action_chunk = self._ws_client.receive()
            self._action_chunks.append(action_chunk)
            self._update_action_queue(action_chunk)
