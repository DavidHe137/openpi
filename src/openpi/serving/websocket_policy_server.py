import asyncio
from collections.abc import Callable
import http
import logging
import multiprocessing as mp
import time
import traceback
from typing import Any
import uuid

from openpi_client import msgpack_numpy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferType
from openpi_client.messages import RTCParams
import websockets.asyncio.server as _server
import websockets.frames
import zmq
import zmq.asyncio

from openpi.serving.metrics import BatchMetrics
from openpi.serving.metrics import MetricsCollector
from openpi.serving.metrics import plot_metrics

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy_factory: Callable,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        batch_size: int = 1,
        log_dir: str | None = None,
    ) -> None:
        self._policy_factory = policy_factory
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._batch_size = batch_size
        self._log_dir = log_dir

        # Create unique IPC endpoint for ZeroMQ ROUTER/DEALER socket
        socket_id = uuid.uuid4().hex[:8]
        self._endpoint = f"ipc:///tmp/openpi_{socket_id}"

        self._worker = mp.Process(
            target=self.worker,
            args=(self._endpoint, self._batch_size),
        )
        self.responses = dict[int, asyncio.futures.Future]()
        self._worker_identity: bytes | None = None  # Worker identity (learned from first message)
        self.last_request_id = 0

        self._metrics = MetricsCollector()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        # Create async ZeroMQ context and ROUTER socket
        zmq_ctx = zmq.asyncio.Context()

        # ROUTER socket for bidirectional communication with worker (binds so worker can connect)
        self._socket = zmq_ctx.socket(zmq.ROUTER)
        self._socket.bind(self._endpoint)

        # Wait a moment for socket to be ready, then start worker
        await asyncio.sleep(0.1)
        self._worker.start()

        # Start background task to process responses from worker
        # This will also learn the worker identity when worker sends its ready message
        response_task = asyncio.create_task(self._process_responses())

        # Wait for worker to connect and identify itself
        # The worker sends a "ready" message when it starts, allowing us to learn its identity
        while self._worker_identity is None:
            await asyncio.sleep(0.01)  # Small sleep to allow async task to process ready message
            if not self._worker.is_alive():
                raise RuntimeError("Worker process died before connecting")

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
        except KeyboardInterrupt:
            logger.info("Server interrupted, shutting down...")
        finally:
            # Generate plots before cleanup
            if self._log_dir:
                logger.info("Generating metrics plots...")
                plot_metrics(self._metrics, self._log_dir)

            # Cleanup
            response_task.cancel()
            self._socket.close()
            zmq_ctx.term()
            self._worker.terminate()
            self._worker.join(timeout=5)

    async def _process_responses(self):
        """Background task that reads from the ROUTER socket and completes futures.

        ROUTER messages consist of identity frame + message frame. The worker's DEALER
        socket automatically includes its identity when sending responses.
        """
        while True:
            try:
                # ROUTER receives: identity frame + message frame
                worker_identity = await self._socket.recv()

                # Learn worker identity from first message if not already known
                if self._worker_identity is None:
                    self._worker_identity = worker_identity
                    logger.info("Learned worker identity")

                # Receive the message frame (pickled Python object)
                message = await self._socket.recv_pyobj()

                # Handle "ready" message from worker (used to establish identity)
                if message == "ready":
                    logger.info("Received ready message from worker")
                    continue

                # Handle typed messages (message_type, payload)
                msg_type, payload = message

                if msg_type == "response":
                    # Handle normal request/response messages
                    request_id, action = payload

                    if request_id in self.responses:
                        self.responses[request_id].set_result(action)
                        logger.info(f"Set result for request {request_id}")
                    else:
                        logger.warning(f"Received response for unknown request {request_id}")

                elif msg_type == "batch_metrics":
                    # Handle batch metrics from worker
                    batch_metric = payload
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

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing response: {e}", exc_info=True)

    def worker(self, endpoint: str, batch_size: int):
        """Worker process that uses DEALER socket to communicate with ROUTER.

        DEALER automatically handles identity frames - it strips identity when receiving
        and adds its identity when sending, making bidirectional communication simple.
        """
        logger.info("Worker started")
        # Initialize policy in worker process to avoid CUDA fork issues
        logger.info("Initializing policy in worker process")
        self._policy = self._policy_factory()
        self._warmup(batch_size)
        logger.info("Worker warmed up")

        # Create blocking ZeroMQ context and DEALER socket
        zmq_ctx = zmq.Context()

        # DEALER socket for bidirectional communication with ROUTER (connects to main's ROUTER)
        socket = zmq_ctx.socket(zmq.DEALER)
        socket.connect(endpoint)

        # Send "ready" message so main process can learn our identity
        socket.send_pyobj("ready")
        logger.info("Sent ready message to main process")

        # Use poller for non-blocking receive
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        # Initialize batch counter and request buffer
        batch_counter = 0
        request_buffer = []  # Buffer for requests beyond batch_size

        try:
            while True:
                # Wait indefinitely for the first message (if buffer empty)
                if not request_buffer:
                    poller.poll()  # Block until at least one message arrives
                    request_id, obs = socket.recv_pyobj()
                    request_buffer.append((request_id, obs))

                # Drain the entire queue non-blockingly
                while True:
                    socks = dict(poller.poll(timeout=0))
                    if socket in socks and socks[socket] == zmq.POLLIN:
                        request_id, obs = socket.recv_pyobj()
                        request_buffer.append((request_id, obs))
                    else:
                        break  # Queue is empty

                # Track metrics
                queue_depth = len(request_buffer)
                batch_start_time = time.perf_counter()

                # Take up to batch_size requests from buffer
                request_ids = []
                batch = []
                num_real = min(batch_size, len(request_buffer))
                for _ in range(num_real):
                    request_id, obs = request_buffer.pop(0)
                    request_ids.append(request_id)
                    batch.append(obs)

                # # Pad batch to batch_size to avoid JIT recompilation
                # while len(batch) < batch_size:
                #     batch.append(batch[-1])

                if batch:
                    # TODO: can we support multiple infer_types in the same batch?
                    assert len({request.infer_type for request in batch}) == 1, (
                        "All requests must have the same infer_type"
                    )

                    # Enhanced logging with accurate queue depth
                    logger.info(
                        f"Processing batch {batch_counter}: "
                        f"queue_depth={queue_depth}, "
                        f"real_requests={num_real}/{len(batch)}, "
                        f"utilization={num_real / len(batch):.2%}"
                    )

                    # Execute inference
                    actions = self._policy.infer_batch(batch)
                    batch_end_time = time.perf_counter()

                    # Create batch metrics
                    batch_metric = BatchMetrics(
                        batch_id=batch_counter,
                        processing_start_time=batch_start_time,
                        processing_end_time=batch_end_time,
                        num_real_requests=num_real,
                        total_batch_size=len(batch),
                        request_ids=request_ids[:num_real],
                    )

                    # Send batch metrics to main process
                    socket.send_pyobj(("batch_metrics", batch_metric))

                    # Send individual responses with new protocol
                    for i in range(num_real):
                        socket.send_pyobj(("response", (request_ids[i], actions[i])))

                    batch_counter += 1
        finally:
            socket.close()
            zmq_ctx.term()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                message = msgpack_numpy.unpackb(await websocket.recv())
                request = InferRequest(**message)

                # Track arrival time
                arrival_time = time.perf_counter()
                request_id = self.last_request_id + 1
                self.last_request_id = request_id
                self._metrics.add_request_arrival(request_id, arrival_time)

                self.responses[request_id] = asyncio.Future()

                # Worker identity should already be learned from ready message
                if self._worker_identity is None:
                    raise RuntimeError("Worker identity not available - worker may not have connected")

                # Track queued time
                queued_time = time.perf_counter()
                self._metrics.add_request_queued(request_id, queued_time)

                # ROUTER sends: identity frame + message frame
                await self._socket.send(self._worker_identity, zmq.SNDMORE)
                await self._socket.send_pyobj((request_id, request))

                action = await self.responses[request_id]

                # Track finished time
                finished_time = time.perf_counter()
                self._metrics.add_request_finished(request_id, finished_time)

                await websocket.send(packer.pack(action))

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _warmup(self, max_batch_size: int) -> None:
        """Warm up policy by compiling for the fixed max_batch_size.

        Since we always pad batches to max_batch_size, we only need to compile once.
        This avoids JIT compilation delays during inference.
        """
        logger.info("Warming up policy...")
        observation = self._policy.make_example()

        requests: list[InferRequest] = []

        requests.append(InferRequest(observation=observation, infer_type=InferType.SYNC, params=None))
        requests.append(
            InferRequest(
                observation=observation,
                infer_type=InferType.INFERENCE_TIME_RTC,
                params=RTCParams(
                    prev_action=self._policy.make_example_actions(),
                    s_param=5,
                    d_param=3,
                ),
            )
        )
        for batch_size in range(1, max_batch_size + 1):
            for request in requests:
                logger.info(f"Warming up {request.infer_type} for batch_size={batch_size}")
                # Warm up with full batch_size (we always pad to this size)
                batch = [request] * batch_size
                self._policy.infer_batch(batch)


def _health_check(connection: _server.ServerConnection, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
