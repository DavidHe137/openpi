import logging
import time
from typing import Callable
from typing import Optional

import numpy as np
from dataclasses import asdict

from openpi_client import messages
from openpi_client import msgpack_numpy
from openpi_client.messages import (
    ConnectRequest,
    WarmupAck,
    WarmupPing,
    WarmupPong,
)
from openpi_client.schemas import Observation, ServerMetadata
from openpi_client.transport import ClientTransport, create_transport
from openpi_client.transport.websocket import wait_for_server

logger = logging.getLogger(__name__)

NUM_WARMUP = 100
WARMUP_OBS_BYTES = 3 * 224 * 224 * 3  # 3 channels, 224x224 pixels, 3 bytes per pixel


class PolicyClient:
    """Transport-agnostic policy client.

    Owns msgpack framing, handshake, warmup, inference / ack / episode messaging.
    Transport (WebSocket or QUIC) is injected.
    """

    def __init__(
        self,
        transport: ClientTransport,
        robot_id: str,
        server_metadata: ServerMetadata,
        control_hz: float = 10.0,
        pre_send_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        self._transport = transport
        self._robot_id = robot_id
        self._server_metadata = server_metadata
        self._pre_send_hook = pre_send_hook
        self._handshake(control_hz)
        self._warmup()

    @classmethod
    def connect(
        cls,
        robot_id: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        transport: str = "ws",
        transport_port: Optional[int] = None,
        api_key: Optional[str] = None,
        control_hz: float = 10.0,
        pre_send_hook: Optional[Callable[[], None]] = None,
    ) -> "PolicyClient":
        """Wait for server, fetch metadata, build transport, run handshake + warmup.

        `port` is the HTTP port used to fetch /metadata. `transport_port` is the port
        the transport connects to (defaults to `port`; set this to the QUIC listener
        port when `transport="quic"`).
        """
        metadata_dict = wait_for_server(host, port, api_key=api_key)
        server_metadata = ServerMetadata(**metadata_dict)

        tport = transport_port if transport_port is not None else port

        # If the server advertises a tunnel, prefer it for the transport URL (WS path only).
        tunnel_url = server_metadata.tunnel_url
        if transport == "ws":
            from openpi_client.transport.websocket import WebSocketClientTransport

            client_transport: ClientTransport = WebSocketClientTransport.connect(
                host=host,
                port=tport,
                api_key=api_key,
                tunnel_url=tunnel_url,
            )
        else:
            client_transport = create_transport(transport, host=host, port=tport, api_key=api_key)

        return cls(
            transport=client_transport,
            robot_id=robot_id,
            server_metadata=server_metadata,
            control_hz=control_hz,
            pre_send_hook=pre_send_hook,
        )

    @property
    def server_metadata(self) -> ServerMetadata:
        return self._server_metadata

    def _handshake(self, control_hz: float) -> None:
        self._transport.send_message(
            msgpack_numpy.packb(asdict(ConnectRequest(robot_id=self._robot_id, control_hz=control_hz)))
        )
        msgpack_numpy.unpackb(self._transport.receive_message())  # ConnectResponse ack
        logger.info("Connected as robot_id=%s", self._robot_id)

    def _warmup(self) -> None:
        for i in range(NUM_WARMUP):
            logger.debug("warmup[%d]: sending ping", i)
            ping = WarmupPing(client_timestamp=time.time(), payload=bytes(WARMUP_OBS_BYTES))
            try:
                self._transport.send_message(msgpack_numpy.packb(asdict(ping)))
            except Exception:
                logger.exception("warmup[%d]: send ping failed", i)
                raise
            logger.debug("warmup[%d]: waiting for pong", i)
            try:
                raw = self._transport.receive_message()
            except Exception:
                logger.exception("warmup[%d]: receive pong failed", i)
                raise
            pong = WarmupPong(**msgpack_numpy.unpackb(raw))
            ack = WarmupAck(server_send_time=pong.server_send_time, client_receive_time=time.time())
            try:
                self._transport.send_message(msgpack_numpy.packb(asdict(ack)))
            except Exception:
                logger.exception("warmup[%d]: send ack failed", i)
                raise
            logger.debug("warmup[%d]: done", i)

    def close(self) -> None:
        self._transport.close()

    def send(
        self,
        obs: Observation,
        deadline: float,
        action_start_step: int,
        infer_type: messages.InferType = messages.InferType.SYNC,
        execution_horizon: int = 0,
        noise: Optional[np.ndarray] = None,
    ) -> None:
        if self._pre_send_hook is not None:
            self._pre_send_hook()

        request = messages.InferRequest(
            request_timestamp=time.time(),
            observation_step=obs.step,
            action_start_step=action_start_step,
            robot_id=self._robot_id,
            observation=asdict(obs),
            deadline=deadline,
            infer_type=infer_type,
            noise=noise,
            execution_horizon=execution_horizon,
        )
        self._transport.send_message(msgpack_numpy.packb(asdict(request)))

    def receive(self) -> messages.InferResponse:
        response = msgpack_numpy.unpackb(self._transport.receive_message())
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return messages.InferResponse(**response)

    def send_ack(
        self,
        request_id: int,
        receive_time: float,
        execution_start_step: int,
        first_executed_index: int = 0,
    ) -> None:
        ack = messages.ResponseAck(
            request_id=request_id,
            receive_time=receive_time,
            execution_start_step=execution_start_step,
            first_executed_index=first_executed_index,
        )
        self._transport.send_message(msgpack_numpy.packb(asdict(ack)))

    def reset(self) -> None:
        self._transport.send_message(msgpack_numpy.packb(asdict(messages.ResetRequest(robot_id=self._robot_id))))

    def send_episode_start(
        self,
        task_suite_name: str,
        task_id: int,
        episode_idx: int,
        max_episode_steps: int,
        task_language: str,
    ) -> None:
        payload = messages.EpisodeStart(
            task_suite_name=task_suite_name,
            task_id=task_id,
            episode_idx=episode_idx,
            max_episode_steps=max_episode_steps,
            task_language=task_language,
        )
        self._transport.send_message(msgpack_numpy.packb(asdict(payload)))

    def send_episode_step(self) -> None:
        payload = messages.EpisodeStep()
        self._transport.send_message(msgpack_numpy.packb(asdict(payload)))

    def send_episode_end(
        self,
        task_suite_name: str,
        task_id: int,
        episode_idx: int,
        success: bool,
        duration_s: float,
        steps_taken: int,
    ) -> None:
        payload = messages.EpisodeEnd(
            task_suite_name=task_suite_name,
            task_id=task_id,
            episode_idx=episode_idx,
            success=success,
            duration_s=duration_s,
            steps_taken=steps_taken,
        )
        self._transport.send_message(msgpack_numpy.packb(asdict(payload)))
