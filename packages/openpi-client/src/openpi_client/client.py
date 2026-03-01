import logging
import time
from typing import Optional, Tuple

import numpy as np
from dataclasses import asdict
import websockets.sync.client

from openpi_client import msgpack_numpy
from openpi_client import messages
from openpi_client.schemas import ActionChunk, Observation, ServerMetadata


class BidirectionalWebsocket:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        robot_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._robot_id = robot_id
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._uri += f"/ws?robot_id={robot_id}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    @property
    def server_metadata(self) -> ServerMetadata:
        return self._server_metadata

    def _wait_for_server(
        self,
    ) -> Tuple[websockets.sync.client.ClientConnection, ServerMetadata]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata_dict = msgpack_numpy.unpackb(conn.recv())
                metadata = ServerMetadata(**metadata_dict)
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def send(
        self,
        obs: Observation,
        deadline: float,
        use_rtc: bool = False,
        prev_action: Optional[np.ndarray] = None,
        s_param: Optional[int] = None,
        d_param: Optional[int] = None,
        noise: Optional[np.ndarray] = None,
    ) -> None:
        infer_type = messages.InferType.SYNC
        params = None
        if use_rtc:
            assert s_param is not None
            assert d_param is not None
            assert prev_action is not None
            infer_type = messages.InferType.INFERENCE_TIME_RTC
            params = messages.RTCParams(prev_action=prev_action, s_param=s_param, d_param=d_param)
        request = messages.InferRequest(
            request_timestamp=time.time(),
            start_step=obs.step,
            robot_id=self._robot_id,
            observation=asdict(obs),
            deadline=deadline,
            infer_type=infer_type,
            params=params,
            noise=noise,
        )
        data = msgpack_numpy.packb(asdict(request))

        self._ws.send(data)  # type: ignore

    def receive(
        self,
    ) -> ActionChunk:  # noqa: UP006
        response = self._ws.recv()

        response = msgpack_numpy.unpackb(response)
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")

        infer_response = messages.InferResponse(**response)
        response_timestamp = time.time()
        action_chunk = ActionChunk(
            actions=infer_response.actions,
            request_timestamp=infer_response.request_timestamp,
            response_timestamp=response_timestamp,
            start_step=infer_response.start_step,
            execution_horizon=infer_response.execution_horizon,
            noise=infer_response.noise,
        )
        return action_chunk

    def reset(self) -> None:
        data = msgpack_numpy.packb({"reset": True, "robot_id": self._robot_id})
        self._ws.send(data)
