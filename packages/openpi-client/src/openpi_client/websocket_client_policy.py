import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from dataclasses import asdict
from typing_extensions import override
import websockets.sync.client
import websockets.asyncio.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from openpi_client import messages
from openpi_client.client import _parse_urls
from openpi_client.messages import ConnectRequest, WarmupAck, WarmupPing, WarmupPong
from openpi_client.schemas import Observation, ServerMetadata

NUM_WARMUP = 10
WARMUP_OBS_BYTES = 3 * 224 * 224 * 3


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        robot_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        control_hz: float = 10.0,
    ) -> None:
        self._robot_id = robot_id
        self._ws_uri, self._http_base = _parse_urls(host, port)
        self._api_key = api_key
        self._ws_lock = threading.Lock()  # Thread-safe WebSocket access
        self._server_metadata = self._wait_for_server()
        if self._server_metadata.tunnel_url:
            tunnel_host = self._server_metadata.tunnel_url.replace("https://", "", 1)
            self._ws_uri = f"wss://{tunnel_host}/ws"
        self._ws = self._connect_ws()
        self._handshake(control_hz)
        self._warmup()
        self._observation_step: int = 0
        self._action_step: int = 0

    @property
    def server_metadata(self) -> ServerMetadata:
        return self._server_metadata

    def _wait_for_server(self) -> ServerMetadata:
        logging.info(f"Waiting for server at {self._http_base}...")
        while True:
            try:
                resp = requests.get(
                    f"{self._http_base}/metadata",
                    headers={"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None,
                    timeout=5,
                )
                resp.raise_for_status()
                return ServerMetadata(**resp.json())
            except requests.exceptions.RequestException:
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

    def _handshake(self, control_hz: float) -> None:
        self._ws.send(msgpack_numpy.packb(asdict(ConnectRequest(robot_id=self._robot_id, control_hz=control_hz))))
        msgpack_numpy.unpackb(self._ws.recv())  # ConnectResponse ack

    def _warmup(self) -> None:
        for _ in range(NUM_WARMUP):
            ping = WarmupPing(client_timestamp=time.time(), payload=bytes(WARMUP_OBS_BYTES))
            self._ws.send(msgpack_numpy.packb(asdict(ping)))
            pong = WarmupPong(**msgpack_numpy.unpackb(self._ws.recv()))
            ack = WarmupAck(server_send_time=pong.server_send_time, client_receive_time=time.time())
            self._ws.send(msgpack_numpy.packb(asdict(ack)))

    def reset(self) -> None:
        self._observation_step = 0
        self._action_step = 0

    @override
    def infer(
        self,
        obs: Observation,
        use_rtc: bool = False,
        s_param: Optional[int] = None,
        d_param: Optional[int] = None,
        noise: Optional[np.ndarray] = None,
    ) -> Dict:  # noqa: UP006
        execution_horizon = self._server_metadata.action_horizon
        infer_type = messages.InferType.SYNC
        params = None
        if use_rtc:
            infer_type = messages.InferType.INFERENCE_TIME_RTC
            params = messages.RTCParams(s_param=s_param, d_param=d_param)  # type: ignore
        request = messages.InferRequest(
            request_timestamp=time.time(),
            observation_step=self._observation_step,
            action_start_step=self._action_step,
            robot_id=self._robot_id,
            observation=asdict(obs),
            deadline=time.time(),  # execute immediately — no queue to drain
            infer_type=infer_type,
            execution_horizon=execution_horizon,
            params=params,
            noise=noise,
        )
        data = msgpack_numpy.packb(asdict(request))

        with self._ws_lock:
            self._ws.send(data)
            response = self._ws.recv()

        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")

        result = msgpack_numpy.unpackb(response)
        receive_time = time.time()
        ack = messages.ResponseAck(
            request_id=result["request_id"],
            receive_time=receive_time,
            execution_start_step=self._action_step,  # execution starts at the top of this chunk
            first_executed_index=0,
        )
        with self._ws_lock:
            self._ws.send(msgpack_numpy.packb(asdict(ack)))

        self._observation_step += 1
        self._action_step += execution_horizon
        return result


class AsyncWebsocketClientPolicy:
    """Async version of WebsocketClientPolicy for high-performance concurrent requests.

    This class uses async websockets with a fixed-size connection pool to enable true concurrent
    requests without blocking. Each request gets its own connection from the pool.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        num_connections: int = 100,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._server_metadata = None
        self._connection_pool: list[Any] = []
        self._pool_lock = asyncio.Lock()
        self._num_connections = num_connections

    async def connect(self) -> ServerMetadata:
        """Connect to the server and retrieve metadata."""
        results = await asyncio.gather(*[self._create_connection() for _ in range(self._num_connections)])
        self._connection_pool = [conn for conn, _ in results]
        self._server_metadata = results[0][1]
        return self._server_metadata

    async def _create_connection(
        self,
    ) -> Tuple[websockets.asyncio.client.ClientConnection, ServerMetadata]:
        """Create a new websocket connection and retrieve metadata."""
        logging.info(f"Waiting for server at {self._uri}...")
        start = time.time()
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = await websockets.asyncio.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata_bytes = await conn.recv()
                metadata_dict = msgpack_numpy.unpackb(metadata_bytes)
                metadata = ServerMetadata(**metadata_dict)
                return conn, metadata

            except ConnectionRefusedError:
                timeout = 300
                if time.time() - start > timeout:
                    raise RuntimeError(f"Failed to connect to server after {timeout} seconds")
                logging.info("Still waiting for server...")
                await asyncio.sleep(5)

    async def _get_connection(self) -> Any:
        """Get a connection from the pool."""
        async with self._pool_lock:
            assert self._connection_pool, (
                "No connections left in pool. Either allocate more connections or reduce the number of concurrent requests."
            )
            return self._connection_pool.pop()

    async def _return_connection(self, conn: Any) -> None:
        """Return a connection to the pool."""
        async with self._pool_lock:
            self._connection_pool.append(conn)

    async def infer(
        self,
        obs: Observation,
        use_rtc: bool = False,
        s_param: Optional[int] = None,
        d_param: Optional[int] = None,
    ) -> Dict:
        """Send an observation and receive an action asynchronously.

        Each request uses its own connection from the pool to avoid
        concurrent recv() conflicts.
        """
        if self._server_metadata is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        infer_type = messages.InferType.SYNC
        params = None
        if use_rtc:
            infer_type = messages.InferType.INFERENCE_TIME_RTC
            params = messages.RTCParams(s_param=s_param, d_param=d_param)  # type: ignore
        request = messages.InferRequest(
            robot_id=self._robot_id,
            observation=asdict(obs),
            infer_type=infer_type,
            params=params,
        )
        data = msgpack_numpy.packb(asdict(request))

        conn = await self._get_connection()
        try:
            await conn.send(data)
            response = await conn.recv()
            if isinstance(response, str):
                # we're expecting bytes; if the server sends a string, it's an error.
                raise RuntimeError(f"Error in inference server:\n{response}")
            result = msgpack_numpy.unpackb(response)
            await self._return_connection(conn)
            return result
        except Exception:
            # Don't return broken connections to pool
            await conn.close()
            raise

    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._pool_lock:
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
