from openpi_client.schemas import Observation
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.client import BidirectionalWebsocket
from typing_extensions import override
from openpi_client import messages


class InferenceTimeRTCBroker(ActionChunkBroker):
    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
        min_execution_horizon: int = 0,
        block_until_first_chunk: bool = True,
    ):
        """
        Args:
            ws_client: the websocket client to use for inference
            control_hz: the control frequency of the environment
            realtime: whether to run in realtime mode, setting this False essentially means inference latency is 0
            min_execution_horizon: tells the server the number of steps to wait before inferring a new action chunk
        """
        super().__init__(
            ws_client=ws_client,
            control_hz=control_hz,
            realtime=realtime,
            min_execution_horizon=min_execution_horizon,
            block_until_first_chunk=block_until_first_chunk,
        )

    @override
    def _infer(self, obs: Observation) -> None:
        self._ws_client.send(
            obs,
            self.deadline,
            self._next_action_step,
            infer_type=messages.InferType.INFERENCE_TIME_RTC,
            min_execution_horizon=self._min_execution_horizon,
        )
