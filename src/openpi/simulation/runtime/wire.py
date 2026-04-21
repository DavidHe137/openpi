"""In-memory transport replacing the websocket between broker and server.

The real stack runs three OS processes connected by ZMQ + shared memory.
``SimWire`` collapses that into a single-process, event-loop-driven graph:

    Broker ──SimWsClient.send()──► SimWire.client_send_infer()
                                         │
                                         │ scheduled at now + d_net
                                         ▼
                                    SimServer._on_infer_request() ──► scheduler.update()
                                                                          │
                                                                          ▼
                                                                   scheduler.schedule()
                                                                          │
                                                                          ▼
                                                                    SimGPU.drain()

    SimGPU._dispatch() ──► SimWire.server_send_response()
                                    │
                                    │ scheduled at now + d_net
                                    ▼
                              Broker._on_response()  ──(ack)──► SimWire.client_send_ack()

All hops are ``event_loop.schedule(delay, cb)`` calls, so the timeline is
fully deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from openpi_client import messages
from openpi_client.messages import InferType
from openpi_client.schemas import Observation
from openpi_client.schemas import ServerMetadata

if TYPE_CHECKING:
    from openpi.simulation.runtime.event_loop import EventLoop


OnInferRequest = Callable[[str, Observation, int, int, int, float], None]
OnAck = Callable[[str, int, float, int, int], None]
OnReset = Callable[[str], None]
OnResponse = Callable[[messages.InferResponse], None]


class SimWire:
    """Routes messages between brokers and a single server, with scheduled delays."""

    def __init__(
        self,
        event_loop: "EventLoop",
        *,
        d_net_s: float = 0.0,
    ) -> None:
        self._loop = event_loop
        self._d_net_s = d_net_s

        # Server-side handlers installed by SimServer.
        self._on_infer: OnInferRequest | None = None
        self._on_ack: OnAck | None = None
        self._on_reset: OnReset | None = None

        # Client-side handlers installed by each SimWsClient.
        self._response_handlers: dict[str, OnResponse] = {}

    # ----- server installs handlers -----

    def bind_server(
        self,
        on_infer: OnInferRequest,
        on_ack: OnAck,
        on_reset: OnReset,
    ) -> None:
        self._on_infer = on_infer
        self._on_ack = on_ack
        self._on_reset = on_reset

    # ----- brokers install response handlers -----

    def register_client(self, robot_id: str, on_response: OnResponse) -> None:
        self._response_handlers[robot_id] = on_response

    def unregister_client(self, robot_id: str) -> None:
        self._response_handlers.pop(robot_id, None)

    # ----- client → server -----

    def client_send_infer(
        self,
        robot_id: str,
        obs: Observation,
        deadline_step: int,
        action_start_step: int,
        execution_horizon: int,
        request_timestamp: float,
    ) -> None:
        assert self._on_infer is not None, "SimServer must bind before clients send"
        on_infer = self._on_infer
        self._loop.schedule(
            self._d_net_s,
            lambda: on_infer(robot_id, obs, deadline_step, action_start_step, execution_horizon, request_timestamp),
        )

    def client_send_ack(
        self,
        robot_id: str,
        request_id: int,
        receive_time: float,
        execution_start_step: int,
        first_executed_index: int,
    ) -> None:
        assert self._on_ack is not None, "SimServer must bind before clients send"
        on_ack = self._on_ack
        self._loop.schedule(
            self._d_net_s,
            lambda: on_ack(robot_id, request_id, receive_time, execution_start_step, first_executed_index),
        )

    def client_send_reset(self, robot_id: str) -> None:
        assert self._on_reset is not None, "SimServer must bind before clients send"
        on_reset = self._on_reset
        self._loop.schedule(self._d_net_s, lambda: on_reset(robot_id))

    # ----- server → client -----

    def server_send_response(self, response: messages.InferResponse) -> None:
        handler = self._response_handlers.get(response.robot_id)
        if handler is None:
            return
        self._loop.schedule(self._d_net_s, lambda: handler(response))


class SimWsClient:
    """Drop-in stand-in for ``BidirectionalWebsocket`` that the broker holds.

    The broker calls ``send`` / ``send_ack`` / ``reset``; those route through
    ``SimWire`` and fire on the event loop after the configured network delay.
    ``receive`` is never invoked because the broker is constructed with
    ``start_receive_thread=False``; the sim injects responses directly via
    ``broker._on_response``.
    """

    def __init__(
        self,
        robot_id: str,
        wire: SimWire,
        clock,
        server_metadata: ServerMetadata,
    ) -> None:
        self._robot_id = robot_id
        self._wire = wire
        self._clock = clock
        self._server_metadata = server_metadata

    @property
    def server_metadata(self) -> ServerMetadata:
        return self._server_metadata

    def send(
        self,
        obs: Observation,
        deadline_step: int,
        action_start_step: int,
        infer_type: InferType = InferType.SYNC,
        execution_horizon: int = 0,
        noise: np.ndarray | None = None,
    ) -> None:
        self._wire.client_send_infer(
            robot_id=self._robot_id,
            obs=obs,
            deadline_step=deadline_step,
            action_start_step=action_start_step,
            execution_horizon=execution_horizon,
            request_timestamp=self._clock.time(),
        )

    def receive(self) -> messages.InferResponse:  # pragma: no cover - unused in sim
        raise RuntimeError(
            "SimWsClient.receive() is not used; the sim injects responses via broker._on_response."
        )

    def send_ack(
        self,
        request_id: int,
        receive_time: float,  # noqa: ARG002 — broker passes wall time; sim substitutes sim time
        execution_start_step: int,
        first_executed_index: int = 0,
    ) -> None:
        # The broker derives ``receive_time`` from ``ActionChunk.from_infer_response``
        # which calls ``time.time()`` — wall time. For sim parity we substitute the
        # event-loop clock so action_latency stays consistent with ``server_send_time``.
        self._wire.client_send_ack(
            robot_id=self._robot_id,
            request_id=request_id,
            receive_time=self._clock.time(),
            execution_start_step=execution_start_step,
            first_executed_index=first_executed_index,
        )

    def reset(self) -> None:
        self._wire.client_send_reset(self._robot_id)

    def close(self) -> None:
        self._wire.unregister_client(self._robot_id)
