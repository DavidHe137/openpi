import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

import numpy as np
from dataclasses import asdict
import websockets.sync.client

from openpi_client import msgpack_numpy
from openpi_client import messages
from openpi_client.schemas import ActionChunk, Observation, ServerMetadata

logger = logging.getLogger(__name__)


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
        explicit_scheme = False
        if host.startswith("https://"):
            ws_scheme, http_scheme = "wss", "https"
            host = host[len("https://") :]
            explicit_scheme = True
        elif host.startswith("http://"):
            ws_scheme, http_scheme = "ws", "http"
            host = host[len("http://") :]
            explicit_scheme = True
        else:
            ws_scheme, http_scheme = "ws", "http"
        base = host if (port is None or explicit_scheme) else f"{host}:{port}"
        self._ws_uri = f"{ws_scheme}://{base}/ws?robot_id={robot_id}"
        self._http_base = f"{http_scheme}://{base}"
        self._api_key = api_key
        self._server_metadata = self._wait_for_server()
        if self._server_metadata.tunnel_url:
            tunnel_host = self._server_metadata.tunnel_url.replace("https://", "", 1)
            self._ws_uri = f"wss://{tunnel_host}/ws?robot_id={robot_id}"
        self._ws = self._connect_ws()

    @property
    def server_metadata(self) -> ServerMetadata:
        return self._server_metadata

    def _wait_for_server(self) -> ServerMetadata:
        logging.info(f"Waiting for server at {self._http_base}...")
        while True:
            try:
                req = urllib.request.Request(f"{self._http_base}/metadata")
                if self._api_key:
                    req.add_header("Authorization", f"Api-Key {self._api_key}")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return ServerMetadata(**json.loads(resp.read()))
            except (urllib.error.URLError, OSError):
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _connect_ws(self) -> websockets.sync.client.ClientConnection:
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        return websockets.sync.client.connect(
            self._ws_uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
        )

    def close(self) -> None:
        self._ws.close()

    def send(
        self,
        obs: Observation,
        deadline: float,
        use_rtc: bool = False,
        prev_action: Optional[np.ndarray] = None,
        s_param: Optional[int] = None,
        d_param: Optional[int] = None,
        noise: Optional[np.ndarray] = None,
        min_execution_horizon: int = 0,
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
            min_execution_horizon=min_execution_horizon,
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
        ack = messages.ResponseAck(request_id=infer_response.request_id, receive_time=response_timestamp)
        self._ws.send(msgpack_numpy.packb(asdict(ack)))

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
        data = msgpack_numpy.packb(asdict(messages.ResetRequest(robot_id=self._robot_id)))
        self._ws.send(data)
