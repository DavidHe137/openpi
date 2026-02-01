import asyncio
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import http
import logging
import multiprocessing as mp
import signal
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

from openpi.serving.request_queue import RequestQueue
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
    - PUSH/PULL sockets for simple unidirectional IPC
    - Stale requests are silently dropped
    """

    def __init__(
        self,
        policy_factory: Callable,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        max_batch_size: int = 1,
    ) -> None:
        self._policy_factory = policy_factory
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._max_batch_size = max_batch_size

        # IPC endpoints
        socket_id = uuid.uuid4().hex[:8]
        self._request_endpoint = f"ipc:///tmp/openpi_req_{socket_id}"
        self._response_endpoint = f"ipc:///tmp/openpi_resp_{socket_id}"

        # Connection tracking
        self._connections: dict[str, ConnectionState] = {}
        self._request_routing: dict[int, str] = {}  # request_id -> conn_id

        self._worker = mp.Process(
            target=self._run_worker,
            args=(
                self._request_endpoint,
                self._response_endpoint,
                self._max_batch_size,
            ),
        )

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self):
        zmq_ctx = zmq.asyncio.Context()

        # PUSH socket for sending requests to worker
        self._request_socket = zmq_ctx.socket(zmq.PUSH)
        self._request_socket.bind(self._request_endpoint)

        # PULL socket for receiving responses from worker
        self._response_socket = zmq_ctx.socket(zmq.PULL)
        self._response_socket.bind(self._response_endpoint)

        # Start worker after sockets are bound
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
        finally:
            response_task.cancel()
            self._request_socket.close()
            self._response_socket.close()
            zmq_ctx.term()

    async def _route_responses(self):
        """Routes responses from worker to appropriate websocket connections."""
        while True:
            try:
                response: InferResponse = await self._response_socket.recv_pyobj()

                conn_id = self._request_routing.pop(response.request_id, None)
                if conn_id is None:
                    logger.debug(f"No routing for request {response.request_id}")
                    continue

                conn = self._connections.get(conn_id)
                if conn is None:
                    logger.debug(f"Connection {conn_id} closed, dropping response")
                    continue

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
        """Receives requests from websocket and forwards to worker."""
        while True:
            message = msgpack_numpy.unpackb(await conn.websocket.recv())
            infer_request = InferRequest(**message)
            arrived_request = ArrivedRequest.receive(infer_request)

            conn.pending_requests.add(arrived_request.request_id)
            self._request_routing[arrived_request.request_id] = conn.conn_id

            logger.info(
                f"Request from {arrived_request.infer_request.robot_id} for step {arrived_request.infer_request.start_step}"
            )
            await self._request_socket.send_pyobj(arrived_request)

    async def _send_loop(self, conn: ConnectionState):
        """Sends responses from queue to websocket."""
        while True:
            response: InferResponse = await conn.response_queue.get()
            logger.info(
                f"Response for {response.robot_id} for step {response.start_step} with execution horizon {response.execution_horizon}"
            )
            await conn.websocket.send(msgpack_numpy.packb(asdict(response)))  # type: ignore

    def _run_worker(self, request_endpoint: str, response_endpoint: str, max_batch_size: int):
        """Worker process: receives requests, runs batched inference, sends responses."""
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
        request_socket = zmq_ctx.socket(zmq.PULL)
        request_socket.connect(request_endpoint)
        response_socket = zmq_ctx.socket(zmq.PUSH)
        response_socket.connect(response_endpoint)

        # Signal ready
        response_socket.send_string("ready")
        logger.info("Worker ready")

        request_queue = RequestQueue()

        # NOTE: hacky data structure for now
        last_response: dict[str, InferResponse] = {}

        try:
            while not shutdown_requested:
                # Block for at least one message
                request: ArrivedRequest = request_socket.recv_pyobj()
                request_queue.add(request)

                # Drain remaining messages without blocking
                while True:
                    try:
                        request = request_socket.recv_pyobj(zmq.NOBLOCK)
                        request_queue.add(request)
                    except zmq.Again:
                        break

                # Build batch from non-stale requests
                batch: list[ArrivedRequest] = []
                request_queue.clear_stale()
                while not request_queue.empty and len(batch) < max_batch_size:
                    batch.append(request_queue.pop())

                if not batch:
                    continue

                # Run inference
                logger.info(f"Inferring batch of size {len(batch)}")
                try:
                    actions = self._policy.infer_batch([req.infer_request for req in batch])
                except Exception as e:
                    logger.error(f"Inference failed: {e}", exc_info=True)
                    continue

                # Send responses
                for req, action in zip(batch, actions, strict=True):
                    actions = action["actions"]
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
                    )
                    response_socket.send_pyobj(response)
                    last_response[req.infer_request.robot_id] = response

        finally:
            logger.info("Worker shutting down")
            request_socket.close()
            response_socket.close()
            zmq_ctx.term()


def _health_check(connection: _server.ServerConnection, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
