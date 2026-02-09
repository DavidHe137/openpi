from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import http
import logging
import multiprocessing as mp
import signal
import time
import traceback
from typing import Any
import uuid

import numpy as np
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
from openpi.serving.schemas import ArrivedRequest
from openpi.serving.variable_execution import calculate_execution_horizon

logger = logging.getLogger(__name__)


@dataclass
class ConnectionState:
    """State for a single websocket connection."""

    websocket: _server.ServerConnection
    response_queue: asyncio.Queue[InferResponse]
    conn_id: str
    pending_requests: set[int] = field(default_factory=set)


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
    ) -> None:
        self._policy_factory = policy_factory
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._max_batch_size = max_batch_size
        self._log_dir = log_dir

        # Shared memory for latest requests per robot
        self._manager = mp.Manager()
        self._latest_requests = self._manager.dict()  # robot_id -> ArrivedRequest

        # IPC endpoint for responses only
        socket_id = uuid.uuid4().hex[:8]
        self._response_endpoint = f"ipc:///tmp/openpi_resp_{socket_id}"

        # Connection tracking
        self._connections: dict[str, ConnectionState] = {}
        self._request_routing: dict[int, str] = {}  # request_id -> conn_id

        self._worker = mp.Process(
            target=self._run_worker,
            args=(
                self._latest_requests,
                self._response_endpoint,
                self._max_batch_size,
            ),
        )

        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self.responses = dict[int, asyncio.futures.Future]()
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

                conn.pending_requests.discard(response.request_id)
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
            for req_id in conn.pending_requests:
                self._request_routing.pop(req_id, None)
            del self._connections[conn.conn_id]

    async def _recv_loop(self, conn: ConnectionState):
        """Receives requests from websocket and updates shared dict."""
        while True:
            message = msgpack_numpy.unpackb(await conn.websocket.recv())
            infer_request = InferRequest(**message)

            # Track arrival time
            arrival_time = time.perf_counter()
            arrived_request = ArrivedRequest.receive(infer_request)
            self._metrics.add_request_arrival(arrived_request.request_id, arrival_time)

            conn.pending_requests.add(arrived_request.request_id)
            self._request_routing[arrived_request.request_id] = conn.conn_id

            # Update shared dict (overwrites previous request from same robot)
            robot_id = arrived_request.infer_request.robot_id
            self._latest_requests[robot_id] = arrived_request

            # Track queued time (when written to shared dict)
            queued_time = time.perf_counter()
            self._metrics.add_request_queued(arrived_request.request_id, queued_time)

            logger.info(
                f"Request from {arrived_request.infer_request.robot_id} for step {arrived_request.infer_request.start_step}"
            )

    async def _send_loop(self, conn: ConnectionState):
        """Sends responses from queue to websocket."""
        while True:
            response: InferResponse = await conn.response_queue.get()
            logger.info(
                f"Response for {response.robot_id} for step {response.start_step} with execution horizon {response.execution_horizon}"
            )
            await conn.websocket.send(msgpack_numpy.packb(asdict(response)))  # type: ignore

    def _run_worker(
        self,
        latest_requests: dict,
        response_endpoint: str,
        max_batch_size: int,
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
        self._policy = self._policy_factory()
        self._policy.warmup(max_batch_size)

        # Connect to main process
        zmq_ctx = zmq.Context()
        response_socket = zmq_ctx.socket(zmq.PUSH)
        response_socket.connect(response_endpoint)

        # Signal ready
        response_socket.send_string("ready")
        logger.info("Worker ready")

        # Track last processed timestamp per robot for stale detection
        last_processed_timestamp: dict[str, float] = {}

        # NOTE: hacky data structure for now
        last_response: dict[str, InferResponse] = {}

        # Batch counter for metrics
        batch_counter = 0

        try:
            while not shutdown_requested:
                # Check if there are any requests available
                if not latest_requests:
                    time.sleep(0.01)
                    continue

                # Take snapshot of current requests (atomic read via Manager)
                snapshot = dict(latest_requests)

                # Filter non-stale candidates
                candidates: list[tuple[str, ArrivedRequest]] = []
                for robot_id, request in snapshot.items():
                    request_timestamp = request.infer_request.request_timestamp

                    # Check staleness - only process requests newer than last processed
                    if request_timestamp > last_processed_timestamp.get(robot_id, 0):
                        candidates.append((robot_id, request))

                # Sort by deadline (earliest first) and select batch
                candidates.sort(key=lambda x: x[1].infer_request.deadline)
                batch = [request for _, request in candidates[:max_batch_size]]

                if not batch:
                    time.sleep(0.01)
                    continue

                # Update last_processed_timestamp for selected requests
                for request in batch:
                    last_processed_timestamp[request.infer_request.robot_id] = request.infer_request.request_timestamp

                # Track batch timing
                batch_start_time = time.perf_counter()

                # Extract noise from requests if present
                batch_noise = None
                if batch[0].infer_request.noise is not None:
                    # Stack noise from all requests in the batch
                    batch_noise = np.stack([req.infer_request.noise for req in batch], axis=0)

                # Run inference
                logger.info(f"Inferring batch of size {len(batch)}")
                try:
                    actions = self._policy.infer_batch([req.infer_request for req in batch], noise=batch_noise)
                except Exception as e:
                    logger.error(f"Inference failed: {e}", exc_info=True)
                    continue

                batch_end_time = time.perf_counter()

                # Create and send batch metrics
                batch_metric = BatchMetrics(
                    batch_id=batch_counter,
                    processing_start_time=batch_start_time,
                    processing_end_time=batch_end_time,
                    num_real_requests=len(batch),
                    total_batch_size=len(batch),
                    request_ids=[req.request_id for req in batch],
                )
                response_socket.send_pyobj(batch_metric)

                # Send responses
                for req, action_dict in zip(batch, actions, strict=True):
                    actions = action_dict["actions"]
                    noise = action_dict.get("noise")

                    execution_horizon = len(actions)
                    if req.infer_request.robot_id in last_response:
                        execution_horizon = calculate_execution_horizon(
                            last_response[req.infer_request.robot_id],
                            req.infer_request.start_step,
                            actions,
                        )

                    response = InferResponse(
                        robot_id=req.infer_request.robot_id,
                        request_id=req.request_id,
                        start_step=req.infer_request.start_step,
                        request_timestamp=req.infer_request.request_timestamp,
                        execution_horizon=execution_horizon,
                        actions=actions,
                        noise=noise,
                    )
                    response_socket.send_pyobj(response)
                    last_response[req.infer_request.robot_id] = response

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
