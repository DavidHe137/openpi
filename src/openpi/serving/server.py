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

from openpi.serving.engine import _run_gpu_worker
from openpi.serving.scheduler import _run_scheduler
from openpi.serving.schemas import ArrivedRequest
from openpi.serving.utils import ZmqAsyncPullQueue
from openpi.serving.utils import ZmqAsyncPushQueue

# TODO: set up multi process logging with queue handler in logging_config.py
logger = logging.getLogger(__name__)

# TODO: factor away global state, use a more pythonic way
_GLOBAL_STATE = None

app = FastAPI()


@dataclass
class ConnectionState:
    """State for a single WebSocket connection."""

    websocket: WebSocket  # unique per connection
    scheduler_sock: ZmqAsyncPushQueue[ResetRequest | ArrivedRequest]
    response_sock: ZmqAsyncPullQueue[InferResponse]


async def _recv_loop(conn: ConnectionState) -> None:
    """Receives bytes from WebSocket, forwards ArrivedRequest / ResetMessage to scheduler."""
    while True:
        raw = await conn.websocket.receive_bytes()
        message = msgpack_numpy.unpackb(raw)

        if "reset" in message:
            await conn.scheduler_sock.put(ResetRequest(robot_id=message["robot_id"]))
            continue

        arrived = ArrivedRequest.receive(InferRequest(**message))
        await conn.scheduler_sock.put(arrived)


async def _send_loop(conn: ConnectionState) -> None:
    """Dequeues InferResponses and sends them to the WebSocket client."""
    while True:
        response: InferResponse = await conn.response_sock.get()
        raw = msgpack_numpy.packb(asdict(response))
        assert raw is not None
        await conn.websocket.send_bytes(raw)


_uid = uuid.uuid4().hex[:8]
socket_addresses = {
    "sched_in_ep": f"ipc:///tmp/openpi_sched_in_{_uid}",
    "sched_out_ep": f"ipc:///tmp/openpi_sched_out_{_uid}",
    "gpu_in_ep": f"ipc:///tmp/openpi_gpu_in_{_uid}",
    "gpu_out_ep": f"ipc:///tmp/openpi_gpu_out_{_uid}",
}


def _start_backend(metadata: ServerMetadata, policy_factory: Callable) -> tuple[mp.Process, mp.Process]:
    """Start the scheduler and GPU processes."""
    # Unique IPC endpoints for this server instance

    scheduler_proc = mp.Process(
        target=_run_scheduler,
        args=(
            socket_addresses["sched_in_ep"],
            socket_addresses["sched_out_ep"],
            socket_addresses["gpu_in_ep"],
            socket_addresses["gpu_out_ep"],
            metadata.max_batch_size,
            metadata.scheduling_algorithm,
        ),
        daemon=True,
    )
    scheduler_proc.start()
    logger.info("Starting scheduler subprocess…")

    gpu_proc = mp.Process(
        target=_run_gpu_worker,
        args=(policy_factory, metadata.max_batch_size, socket_addresses["gpu_in_ep"], socket_addresses["gpu_out_ep"]),
        daemon=True,
    )
    gpu_proc.start()
    logger.info("Starting GPU subprocess…")
    return scheduler_proc, gpu_proc


@dataclass
class ServerState:
    scheduler_sock: ZmqAsyncPushQueue[ResetRequest | ArrivedRequest]
    gpu_proc: mp.Process
    scheduler_proc: mp.Process


def create_app(metadata: ServerMetadata, policy_factory: Callable) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # start backend processes
        scheduler_proc, gpu_proc = _start_backend(metadata, policy_factory)
        app.state.server = ServerState(
            scheduler_sock=ZmqAsyncPushQueue(socket_addresses["sched_in_ep"], create=True, encoder=lambda x: asdict(x)),
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
        # TODO: need to understand dealer-router, maybe we don't need to create another ZmqAsyncPullQueue here
        conn = ConnectionState(
            websocket=websocket,
            scheduler_sock=websocket.app.state.server.scheduler_sock,
            response_sock=ZmqAsyncPullQueue(
                socket_addresses["sched_out_ep"], create=False, decoder=lambda x: InferResponse(**x)
            ),
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

    def serve_forever(self, host="0.0.0.0", port=8000):
        app = create_app(self._metadata, self._policy_factory)
        uvicorn.run(app, host=host, port=port)
