"""
3 processes:
    WS main process     - FastAPI ASGI app
    Scheduler process   - collects requests from WS main; runs ILP; dispatches batches to GPU
    GPU process         - loads weights; runs batches; sends responses directly to WS main

ZMQ topology (all ipc://, unique per server instance):
    WS main  ──[PUSH: SlotRequest / ResetRequest]──► Scheduler [binds sched_in_ep]
    WS main  ──write_obs()─────────────────────────► mp.RawArray shared memory
    GPU      ──[PUSH: list[InferResponse]]──────────► WS main   [binds gpu_out_ep]
    GPU      ──[PUSH: list[CompletionNotification]]► Scheduler [binds result_ep]
    Scheduler ──[mp.Queue: list[SlotRequest]]───────► GPU
    GPU      ──read_obs()──────────────────────────► mp.RawArray shared memory

    GPU responses bypass the scheduler entirely so ILP solving cannot delay client delivery.
    A single _router_task in WS main reads from gpu_out_ep and dispatches to per-robot queues.
    Large numpy arrays (observations) cross zero process boundaries via ZMQ.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import dataclasses
from dataclasses import asdict
from dataclasses import dataclass
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event
import os
from pathlib import Path
import signal
import time
import uuid

from fastapi import FastAPI
from fastapi import Request
from fastapi import WebSocket
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import HTMLResponse
from openpi_client import msgpack_numpy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
from openpi_client.messages import ResetRequest
from openpi_client.schemas import ServerMetadata
from starlette.websockets import WebSocketDisconnect
import uvicorn
import zmq.asyncio

from openpi.serving.engine import _run_gpu_worker
from openpi.serving.metrics import MetricsStore
from openpi.serving.scheduler import _run_scheduler
from openpi.serving.schemas import SlotRequest
from openpi.serving.schemas import _request_id_counter
from openpi.serving.slots import RobotSlots

MAX_ROBOTS = 100
logger = logging.getLogger(__name__)

_uid = uuid.uuid4().hex[:8]
socket_addresses = {
    "sched_in_ep": f"ipc:///tmp/openpi_sched_in_{_uid}",
    "gpu_out_ep": f"ipc:///tmp/openpi_gpu_out_{_uid}",
    "result_ep": f"ipc:///tmp/openpi_result_{_uid}",
}


@dataclass
class ServerState:
    scheduler_sock: zmq.asyncio.Socket  # PUSH to scheduler
    response_queues: dict[str, asyncio.Queue]
    slots: RobotSlots  # WS manages slot allocation
    gpu_proc: mp.Process
    scheduler_proc: mp.Process
    metrics_store: MetricsStore


async def _router_task(
    response_sock: zmq.asyncio.Socket,
    response_queues: dict[str, asyncio.Queue],
    metrics_store: MetricsStore,
) -> None:
    """Reads batches of InferResponses directly from GPU and dispatches to per-robot queues."""
    while True:
        responses: list[InferResponse] = await response_sock.recv_pyobj()
        metrics_store.record_batch(responses)
        for response in responses:
            queue = response_queues.get(response.robot_id)
            logger.debug("Dispatching response to robot %s", response.robot_id)
            if queue is not None:
                await queue.put(response)
            else:
                logger.debug("No active connection for robot %s, dropping response", response.robot_id)


async def _watchdog_task(gpu_proc: mp.Process, scheduler_proc: mp.Process) -> None:
    """Crashes the server if either backend process dies unexpectedly."""
    while True:
        await asyncio.sleep(1)
        for proc in (gpu_proc, scheduler_proc):
            if not proc.is_alive():
                logger.critical("Backend process %s died (exit code %s), crashing server", proc.name, proc.exitcode)
                os.kill(os.getpid(), signal.SIGTERM)
                return


def _start_backend(
    metadata: ServerMetadata, policy_factory: Callable, log_queue: mp.Queue | None
) -> tuple[mp.Process, mp.Process, RobotSlots, Event, Event]:
    slots = RobotSlots(max_robots=MAX_ROBOTS)
    batch_queue: mp.Queue = mp.Queue()

    gpu_ready = mp.Event()
    sched_ready = mp.Event()

    gpu_proc = mp.Process(
        target=_run_gpu_worker,
        args=(
            policy_factory,
            metadata.max_batch_size,
            slots,
            batch_queue,
            socket_addresses["gpu_out_ep"],
            socket_addresses["result_ep"],
            gpu_ready,
            log_queue,
        ),
        daemon=True,
    )

    scheduler_proc = mp.Process(
        target=_run_scheduler,
        args=(
            socket_addresses["sched_in_ep"],
            socket_addresses["result_ep"],
            batch_queue,
            metadata.max_batch_size,
            metadata.scheduling_algorithm,
            sched_ready,
            log_queue,
            metadata.lookahead_horizon,
            metadata.lookahead_execution_horizon_s,
        ),
        daemon=True,
    )

    logger.info("Starting GPU subprocess…")
    gpu_proc.start()
    logger.info("Starting scheduler subprocess…")
    scheduler_proc.start()

    return scheduler_proc, gpu_proc, slots, sched_ready, gpu_ready


def create_app(metadata: ServerMetadata, policy_factory: Callable, log_queue: mp.Queue | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler_proc, gpu_proc, slots, sched_ready, gpu_ready = _start_backend(metadata, policy_factory, log_queue)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sched_ready.wait)
        logger.info("Scheduler ready")
        await loop.run_in_executor(None, gpu_ready.wait)
        logger.info("GPU worker ready")

        zmq_ctx = zmq.asyncio.Context()

        scheduler_sock = zmq_ctx.socket(zmq.PUSH)
        scheduler_sock.connect(socket_addresses["sched_in_ep"])

        # WS main binds gpu_out_ep so GPU can connect to us
        response_sock = zmq_ctx.socket(zmq.PULL)
        response_sock.bind(socket_addresses["gpu_out_ep"])

        response_queues: dict[str, asyncio.Queue] = {}
        metrics_store = MetricsStore()

        app.state.server = ServerState(
            scheduler_sock=scheduler_sock,
            response_queues=response_queues,
            slots=slots,
            gpu_proc=gpu_proc,
            scheduler_proc=scheduler_proc,
            metrics_store=metrics_store,
        )

        router = asyncio.create_task(_router_task(response_sock, response_queues, metrics_store))
        watchdog = asyncio.create_task(_watchdog_task(gpu_proc, scheduler_proc))

        yield

        watchdog.cancel()
        router.cancel()
        gpu_proc.terminate()
        scheduler_proc.terminate()

        loop = asyncio.get_event_loop()
        for proc in (gpu_proc, scheduler_proc):
            await loop.run_in_executor(None, proc.join, 5)
            if proc.is_alive():
                logger.warning("Process %s did not exit cleanly, killing", proc.name)
                proc.kill()
                await loop.run_in_executor(None, proc.join)

        scheduler_sock.close()
        response_sock.close()
        zmq_ctx.term()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_handler(websocket: WebSocket, robot_id: str):
        await websocket.accept()
        state: ServerState = websocket.app.state.server
        response_queue: asyncio.Queue = asyncio.Queue()

        slot_index = state.slots.register(robot_id)
        state.response_queues[robot_id] = response_queue

        async def recv():
            try:
                while True:
                    raw = await websocket.receive_bytes()
                    msg = msgpack_numpy.unpackb(raw)

                    if "reset" in msg:
                        await state.scheduler_sock.send_pyobj(ResetRequest(robot_id=robot_id))
                        continue

                    if "receive_time" in msg:  # ResponseAck
                        state.metrics_store.record_ack(robot_id, msg["request_id"], msg["receive_time"])
                        continue

                    req = InferRequest(**msg)

                    # Write obs directly to shared memory - no ZMQ copy of image arrays
                    state.slots.write_obs(slot_index, req.observation)

                    slot_req = SlotRequest(
                        slot_index=slot_index,
                        robot_id=robot_id,
                        request_id=next(_request_id_counter),
                        arrival_timestamp=time.time(),
                        start_step=req.start_step,
                        request_timestamp=req.request_timestamp,
                        deadline=req.deadline,
                        infer_type=req.infer_type,
                        params=req.params,
                        noise=req.noise,
                    )
                    await state.scheduler_sock.send_pyobj(slot_req)
                    logger.debug("Sent slot request to scheduler: %s", slot_req)
            except WebSocketDisconnect:
                # FIXME: this is a hack to reset the robot when the websocket disconnects, should make a special message for removing the robot from the scheduler
                await state.scheduler_sock.send_pyobj(ResetRequest(robot_id=robot_id))
                logger.debug("Robot %s disconnected", robot_id)

        async def send():
            while True:
                response: InferResponse = await response_queue.get()
                send_time = time.time()
                stamped = dataclasses.replace(response, server_send_time=send_time)
                state.metrics_store.record_send(robot_id, response.request_id, send_time)
                await websocket.send_bytes(msgpack_numpy.packb(asdict(stamped)))

        recv_task = asyncio.create_task(recv())
        send_task = asyncio.create_task(send())
        try:
            await recv_task
        finally:
            send_task.cancel()
            # FIXME: this is a hack to reset the robot when the websocket disconnects
            await state.scheduler_sock.send_pyobj(ResetRequest(robot_id=robot_id))
            state.slots.free(robot_id)
            state.response_queues.pop(robot_id, None)

    _dashboard_html = (Path(__file__).parent / "metrics" / "dashboard.html").read_text()

    # can also be used for health check
    @app.get("/metadata")
    async def server_metadata() -> dict:
        return asdict(metadata)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return _dashboard_html

    @app.get("/metrics")
    async def get_metrics(request: Request) -> dict:
        return request.app.state.server.metrics_store.snapshot()

    @app.get("/metrics/history")
    async def get_metrics_history(request: Request) -> dict:
        return request.app.state.server.metrics_store.history()

    @app.post("/reset-metrics")
    async def reset_metrics(request: Request) -> dict:
        request.app.state.server.metrics_store.reset()
        return {"status": "ok"}

    return app


class PolicyServer:
    def __init__(self, metadata: ServerMetadata, policy_factory: Callable, log_queue: mp.Queue | None = None):
        self._metadata = metadata
        self._policy_factory = policy_factory
        self._log_queue = log_queue

    def serve_forever(self, host="0.0.0.0", port=8000):
        app = create_app(self._metadata, self._policy_factory, self._log_queue)
        uvicorn.run(app, host=host, port=port)
