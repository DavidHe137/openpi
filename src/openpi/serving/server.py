"""
3 processes:
    WS main process     - FastAPI ASGI app
    Scheduler process   - collects requests from WS main process and dispatches batches to GPU
    GPU process         - loads weights, runs batches on GPU


ZMQ topology (all ipc://, unique per server instance):
    WS main  ──[PUSH: ArrivedRequest / ResetMessage]──► Scheduler
    WS main  ◄──[PULL: InferResponse / BatchMetrics]─── Scheduler
    Scheduler ──[PUSH: List[ArrivedRequest]]───────────► GPU
    Scheduler ◄──[PULL: (batch, actions, t0, t1)]──────── GPU

    # TODO: document server architecture in a non-vibecoded way
    # TODO: note handshakes, scheduler tells main server its ready only after GPU says its ready
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
import logging
import multiprocessing as mp
import uuid

from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.concurrency import asynccontextmanager
from openpi_client import msgpack_numpy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
from openpi_client.messages import ResetRequest
from openpi_client.schemas import ServerMetadata  # TODO: i don't think this should be used in both client and server
import uvicorn
import zmq

from openpi.serving.engine import _run_gpu_worker
from openpi.serving.scheduler import _run_scheduler
from openpi.serving.schemas import ArrivedRequest

# TODO: set up multi process logging with queue handler in logging_config.py
logger = logging.getLogger(__name__)

# TODO: factor away global state, use a more pythonic way
_GLOBAL_STATE = None

app = FastAPI()


@dataclass
class ConnectionState:
    """State for a single WebSocket connection."""

    websocket: WebSocket  # unique per connection
    scheduler_sock: zmq.Socket  # shared for all connections
    response_sock: zmq.Socket  # unique per connection


async def _recv_loop(conn: ConnectionState) -> None:
    """Receives bytes from WebSocket, forwards ArrivedRequest / ResetMessage to scheduler."""
    while True:
        raw = await conn.websocket.receive_bytes()
        message = msgpack_numpy.unpackb(raw)

        if "reset" in message:
            await conn.scheduler_sock.send_pyobj(ResetRequest(robot_id=message["robot_id"]))
            continue

        arrived = ArrivedRequest.receive(InferRequest(**message))
        await conn.scheduler_sock.send_pyobj(arrived)


async def _send_loop(conn: ConnectionState) -> None:
    """Dequeues InferResponses and sends them to the WebSocket client."""
    while True:
        response: InferResponse = await conn.response_sock.get()
        await conn.websocket.send_bytes(msgpack_numpy.packb(asdict(response)))


def _start_backend(self) -> None:
    """Start the scheduler and GPU processes."""
    self._scheduler_proc = mp.Process(
        target=_run_scheduler,
        args=(
            self._sched_in_ep,
            self._sched_out_ep,
            self._gpu_in_ep,
            self._gpu_out_ep,
            self._metadata.max_batch_size,
            self._metadata.scheduling_algorithm,
        ),
        daemon=True,
    )
    self._scheduler_proc.start()
    logger.info("Starting scheduler subprocess…")

    self._gpu_proc = mp.Process(
        target=_run_gpu_worker,
        args=(self._policy_factory, self._metadata.max_batch_size, self._gpu_in_ep, self._gpu_out_ep),
        daemon=True,
    )
    self._gpu_proc.start()
    logger.info("Starting GPU subprocess…")


@dataclass
class ServerState:
    scheduler_sock: zmq.Socket
    gpu_proc: mp.Process
    scheduler_proc: mp.Process


def create_app(metadata: ServerMetadata, policy_factory: Callable) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # start backend processes
        scheduler_proc, gpu_proc = _start_backend(metadata, policy_factory)
        app.state.server = ServerState(
            scheduler_sock=zmq.Socket(zmq.PAIR),
            gpu_proc=gpu_proc,
            scheduler_proc=scheduler_proc,
        )
        yield
        # cleanup
        gpu_proc.terminate()
        scheduler_proc.terminate()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_handler(websocket: WebSocket):
        await websocket.accept()
        conn = ConnectionState(
            websocket=websocket,
            scheduler_sock=websocket.app.state.server.scheduler_sock,
            response_sock=websocket.app.state.server.response_sock,
        )
        await asyncio.gather(_recv_loop(conn), _send_loop(conn))

    @app.get("/metadata")
    async def server_metadata() -> str:
        # TODO:
        return "OK"

    @app.get("/healthz")
    async def health() -> str:
        return "OK"

    @app.post("/reset-metrics")
    async def reset_metrics() -> None:
        # TODO:
        pass

    @app.get("/metrics")
    async def metrics() -> str:
        # TODO: plot metrics and clear history
        return "OK"

    return app


class PolicyServer:
    def __init__(self, metadata: ServerMetadata, policy_factory: Callable):
        self._metadata = metadata
        self._policy_factory = policy_factory

        # TODO: factor away
        # Unique IPC endpoints for this server instance
        _uid = uuid.uuid4().hex[:8]
        self._sched_in_ep = f"ipc:///tmp/openpi_sched_in_{_uid}"
        self._sched_out_ep = f"ipc:///tmp/openpi_sched_out_{_uid}"
        self._gpu_in_ep = f"ipc:///tmp/openpi_gpu_in_{_uid}"
        self._gpu_out_ep = f"ipc:///tmp/openpi_gpu_out_{_uid}"

        self._gpu_proc: mp.Process | None = None
        self._scheduler_proc: mp.Process | None = None

    def serve_forever(self, host="0.0.0.0", port=8000):
        app = create_app(self._metadata, self._policy_factory)
        uvicorn.run(app, host=host, port=port)
