from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
from dataclasses import asdict
from dataclasses import dataclass
import http
import logging
import multiprocessing as mp
import signal
import time
import traceback
from typing import Any
import uuid

from openpi_client import msgpack_numpy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
import websockets.asyncio.server as _server
import websockets.frames
import zmq
import zmq.asyncio

from openpi.serving.metrics import BatchMetrics
from openpi.serving.metrics import MetricsCollector
from openpi.serving.metrics import plot_metrics
from openpi.serving.scheduling import RequestScheduler
from openpi.serving.scheduling import SchedulingAlgorithm
from openpi.serving.schemas import ArrivedRequest

logger = logging.getLogger(__name__)


@dataclass
class ConnectionState:
    """State for a single websocket connection."""

    websocket: _server.ServerConnection
    response_queue: asyncio.Queue[InferResponse]
    conn_id: str


class WebsocketPolicyServer:
    """Serves a policy over websocket with batched inference.

    Architecture:
    - Main process: async websocket handlers with decoupled recv/send loops
    - Worker process: batched policy inference
    - Lock-free shared memory: dict maps robot_id -> latest request
    - Handlers update atomically (single key write), worker snapshots and filters by timestamp
    - Stale requests are silently skipped (worker tracks last_processed_timestamp per robot)
    """

    def __init__(
        self,
        policy_factory: Callable,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        max_batch_size: int = 1,
        log_dir: str | None = None,
        scheduling_algorithm: SchedulingAlgorithm = SchedulingAlgorithm.GREEDY,
    ) -> None:
        self._policy_factory = policy_factory
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._max_batch_size = max_batch_size
        self._log_dir = log_dir
        self._scheduling_algorithm = scheduling_algorithm

        # Shared memory for latest requests per robot
        self._manager = mp.Manager()
        self._latest_requests = self._manager.dict()  # robot_id -> ArrivedRequest
        self._reset_robots = self._manager.list()  # robot_ids that need reset
        self._requests_lock = self._manager.Lock()  # Lock for latest_requests access

        # IPC endpoint for responses only
        socket_id = uuid.uuid4().hex[:8]
        self._response_endpoint = f"ipc:///tmp/openpi_resp_{socket_id}"

        # Connection tracking
        self._connections: dict[str, ConnectionState] = {}
        self._request_routing: dict[int, str] = {}  # request_id -> conn_id

        self._worker = mp.Process(
            target=_run_worker,
            args=(
                self._policy_factory,
                self._latest_requests,
                self._reset_robots,
                self._requests_lock,
                self._response_endpoint,
                self._max_batch_size,
                self._scheduling_algorithm,
            ),
        )

        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self._worker_identity: bytes | None = None  # Worker identity (learned from first message)
        self.last_request_id = 0

        self._metrics = MetricsCollector()

    def serve_forever(self) -> None:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(self._run())

    async def _run(self):
        zmq_ctx = zmq.asyncio.Context()

        # PULL socket for receiving responses from worker
        self._response_socket = zmq_ctx.socket(zmq.PULL)
        self._response_socket.bind(self._response_endpoint)

        # Start worker after socket is bound
        await asyncio.sleep(0.1)
        self._worker.start()

        # Wait for worker ready signal
        ready_msg = await self._response_socket.recv_string()
        if ready_msg != "ready":
            raise RuntimeError(f"Unexpected ready message: {ready_msg}")
        logger.info("Worker ready")

        # Start response router
        response_task = asyncio.create_task(self._route_responses())

        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Server interrupted, shutting down gracefully...")
        finally:
            # Close all active websocket connections
            logger.info("Closing active connections...")
            close_tasks = []
            for conn in list(self._connections.values()):
                try:
                    close_tasks.append(
                        conn.websocket.close(
                            code=websockets.frames.CloseCode.GOING_AWAY,
                            reason="Server shutting down",
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error closing connection: {e}")

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            # Cancel response router and wait for it to finish
            logger.info("Stopping response router...")
            response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await response_task

            # Close ZMQ socket
            logger.info("Closing ZMQ socket...")
            self._response_socket.close()
            zmq_ctx.term()

            # Gracefully stop worker
            logger.info("Stopping worker process...")
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=5)
                if self._worker.is_alive():
                    logger.warning("Worker did not terminate gracefully, killing...")
                    self._worker.kill()
                    self._worker.join()

            # Shutdown manager
            logger.info("Shutting down shared memory manager...")
            self._manager.shutdown()

            # Generate plots after everything is stopped
            if self._log_dir:
                logger.info("Generating metrics plots...")
                try:
                    plot_metrics(self._metrics, self._log_dir)
                except Exception as e:
                    logger.error(f"Error generating plots: {e}", exc_info=True)

            logger.info("Shutdown complete")

    async def _route_responses(self):
        """Routes responses from worker to appropriate websocket connections."""
        while True:
            try:
                message = await self._response_socket.recv_pyobj()

                # Handle both InferResponse and BatchMetrics
                if isinstance(message, BatchMetrics):
                    # Handle batch metrics
                    batch_metric = message
                    self._metrics.add_batch_metrics(batch_metric)

                    # Update processing start time for all requests in batch
                    self._metrics.add_batch_start(
                        batch_metric.request_ids,
                        batch_metric.processing_start_time,
                    )

                    # Log observability metrics
                    stats = self._metrics.get_recent_latency_stats()
                    logger.info(
                        f"Batch {batch_metric.batch_id} completed: "
                        f"batch_time={batch_metric.batch_processing_time * 1000:.1f}ms, "
                        f"avg_lat_1={stats['avg_1'] * 1000:.1f}ms, "
                        f"avg_lat_5={stats['avg_5'] * 1000:.1f}ms, "
                        f"avg_lat_10={stats['avg_10'] * 1000:.1f}ms"
                    )
                    continue

                # Handle InferResponse
                response: InferResponse = message
                conn_id = self._request_routing.pop(response.request_id, None)
                if conn_id is None:
                    logger.debug(f"No routing for request {response.request_id}")
                    continue

                conn = self._connections.get(conn_id)
                if conn is None:
                    logger.debug(f"Connection {conn_id} closed, dropping response")
                    continue

                # Track finished time
                finished_time = time.perf_counter()
                self._metrics.add_request_finished(response.request_id, finished_time)

                await conn.response_queue.put(response)  # type: ignore

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error routing response: {e}", exc_info=True)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")

        conn = ConnectionState(
            websocket=websocket,
            response_queue=asyncio.Queue(),
            conn_id=uuid.uuid4().hex,
        )
        self._connections[conn.conn_id] = conn

        # Send metadata
        await websocket.send(msgpack_numpy.packb(self._metadata))  # type: ignore

        try:
            await asyncio.gather(
                self._recv_loop(conn),
                self._send_loop(conn),
            )
        except websockets.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error.",
            )
            raise
        finally:
            del self._connections[conn.conn_id]

    async def _recv_loop(self, conn: ConnectionState):
        """Receives requests from websocket and updates shared dict."""
        while True:
            message = msgpack_numpy.unpackb(await conn.websocket.recv())
            # FIXME: hack for reset messages
            if "reset" in message:
                robot_id = message["robot_id"]
                with self._requests_lock:
                    if robot_id in self._latest_requests:
                        del self._latest_requests[robot_id]
                self._reset_robots.append(robot_id)
                continue

            infer_request = InferRequest(**message)

            # Track arrival time
            arrival_time = time.perf_counter()
            arrived_request = ArrivedRequest.receive(infer_request)
            self._metrics.add_request_arrival(arrived_request.request_id, arrival_time)

            self._request_routing[arrived_request.request_id] = conn.conn_id

            # Update shared dict (overwrites previous request from same robot)
            robot_id = arrived_request.infer_request.robot_id
            with self._requests_lock:
                self._latest_requests[robot_id] = arrived_request

            # FIXME: review this later, maybe not necessary
            # Track queued time (when written to shared dict)
            queued_time = time.perf_counter()
            self._metrics.add_request_queued(arrived_request.request_id, queued_time)

    async def _send_loop(self, conn: ConnectionState):
        """Sends responses from queue to websocket."""
        while True:
            response: InferResponse = await conn.response_queue.get()
            logger.info(
                f"Response for {response.robot_id} for step {response.start_step} with execution horizon {response.execution_horizon}"
            )
            await conn.websocket.send(msgpack_numpy.packb(asdict(response)))  # type: ignore


def _run_worker(
    policy_factory: Callable,
    latest_requests: dict,
    reset_robots: list,
    requests_lock,
    response_endpoint: str,
    max_batch_size: int,
    scheduling_algorithm: SchedulingAlgorithm = SchedulingAlgorithm.GREEDY,
):
    """Worker process: polls shared dict, runs batched inference, sends responses."""
    logger.info("Worker starting")

    # Graceful shutdown handling
    shutdown_requested = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"Worker received signal {signum}, shutting down...")
        shutdown_requested = True

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Initialize policy in worker to avoid CUDA fork issues
    policy = policy_factory()
    policy.warmup(max_batch_size)

    # Connect to main process
    zmq_ctx = zmq.Context()
    response_socket = zmq_ctx.socket(zmq.PUSH)
    response_socket.connect(response_endpoint)

    # Signal ready
    response_socket.send_string("ready")
    logger.info("Worker ready")

    scheduler = RequestScheduler(algorithm=scheduling_algorithm)
    logger.info(f"Using scheduling algorithm: {scheduling_algorithm.value}")

    # Batch counter for metrics
    batch_counter = 0

    try:
        while not shutdown_requested:
            # Process any pending resets
            while reset_robots:
                robot_id = reset_robots.pop(0)
                scheduler.reset_robot(robot_id)
                logger.info(f"Reset scheduler state for robot {robot_id}")

            with requests_lock:
                snapshot = dict(latest_requests)

            batch = scheduler.get_next_batch(snapshot, max_batch_size)
            if not batch:
                time.sleep(0.01)
                continue

            batch_start_time = time.perf_counter()

            # Run inference
            logger.info(f"Inferring batch of size {len(batch)}")
            try:
                actions = policy.infer_batch([req.infer_request for req in batch])
            except Exception as e:
                logger.error(f"Inference failed: {e}", exc_info=True)
                continue

            batch_end_time = time.perf_counter()

            # Build responses and collect per-robot info for metrics
            responses: list[InferResponse] = []
            batch_robot_ids: list[str] = []
            batch_start_steps: list[int] = []
            batch_execution_horizons: list[int] = []

            for req, action_dict in zip(batch, actions, strict=True):
                execution_horizon = scheduler.calculate_execution_horizon(
                    req.infer_request.robot_id, action_dict["actions"]
                )
                response = InferResponse(
                    robot_id=req.infer_request.robot_id,
                    request_id=req.request_id,
                    start_step=req.infer_request.start_step,
                    request_timestamp=req.infer_request.request_timestamp,
                    execution_horizon=execution_horizon,
                    actions=action_dict["actions"],
                    noise=action_dict["noise"],
                    server_compute_ms=(batch_end_time - batch_start_time) * 1e3,
                )
                responses.append(response)
                batch_robot_ids.append(req.infer_request.robot_id)
                batch_start_steps.append(req.infer_request.start_step)
                batch_execution_horizons.append(execution_horizon)

            # Send batch metrics (with scheduling data)
            batch_metric = BatchMetrics(
                batch_id=batch_counter,
                processing_start_time=batch_start_time,
                processing_end_time=batch_end_time,
                num_real_requests=len(batch),
                total_batch_size=len(batch),
                request_ids=[req.request_id for req in batch],
                robot_ids=batch_robot_ids,
                start_steps=batch_start_steps,
                execution_horizons=batch_execution_horizons,
            )
            response_socket.send_pyobj(batch_metric)

            # Send individual responses
            for response in responses:
                response_socket.send_pyobj(response)
                scheduler.update_last_response(response.robot_id, response)

            scheduler.update_deadlines(batch)

            batch_counter += 1

    finally:
        logger.info("Worker shutting down gracefully")

        # Set linger to allow pending messages to be sent (max 1 second)
        response_socket.setsockopt(zmq.LINGER, 1000)

        response_socket.close()
        zmq_ctx.term()
        logger.info("Worker shutdown complete")


def _health_check(connection: _server.ServerConnection, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
