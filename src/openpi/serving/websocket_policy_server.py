import asyncio
from collections.abc import Callable
import http
import logging
import multiprocessing as mp
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
    ) -> None:
        self._policy_factory = policy_factory
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._batch_size = batch_size

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
        logging.getLogger("websockets.server").setLevel(logging.INFO)

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
        finally:
            # Cleanup
            response_task.cancel()
            self._socket.close()
            zmq_ctx.term()

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

                # Handle normal request/response messages
                request_id, action = message

                if request_id in self.responses:
                    self.responses[request_id].set_result(action)
                    logger.info(f"Set result for request {request_id}")
                else:
                    logger.warning(f"Received response for unknown request {request_id}")
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

        try:
            while True:
                request_ids = []
                batch = []

                # Wait indefinitely for the first message
                poller.poll()  # Block until at least one message arrives

                # Receive the first message
                request_id, obs = socket.recv_pyobj()
                request_ids.append(request_id)
                batch.append(obs)

                # Collect additional messages up to batch_size.
                while len(batch) < batch_size:
                    socks = dict(poller.poll(timeout=0))
                    if socket in socks and socks[socket] == zmq.POLLIN:
                        # DEALER receives messages without identity frame
                        request_id, obs = socket.recv_pyobj()
                        request_ids.append(request_id)
                        batch.append(obs)
                    else:
                        # No more messages immediately available, process what we have
                        break

                if batch:
                    num_real = len(batch)
                    # Pad batch to batch_size with to avoid JIT recompilation
                    while len(batch) < batch_size:
                        batch.append(batch[-1])

                    # TODO: can we support multiple infer_types in the same batch?
                    assert len({request.infer_type for request in batch}) == 1, (
                        "All requests must have the same infer_type"
                    )

                    logger.info(f"Inferring batch of size {batch_size} (padded from {num_real} real requests)")
                    actions = self._policy.infer_batch(batch)

                    # Only send back results for real requests (not padding)
                    for i in range(num_real):
                        # DEALER automatically adds identity frame when sending
                        socket.send_pyobj((request_ids[i], actions[i]))
                        logger.info(f"Sent result for request {request_ids[i]} via ZeroMQ")
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

                request_id = self.last_request_id + 1
                self.last_request_id = request_id
                self.responses[request_id] = asyncio.Future()

                # Worker identity should already be learned from ready message
                if self._worker_identity is None:
                    raise RuntimeError("Worker identity not available - worker may not have connected")

                # ROUTER sends: identity frame + message frame
                await self._socket.send(self._worker_identity, zmq.SNDMORE)
                await self._socket.send_pyobj((request_id, request))
                logger.info(f"Sent request {request_id} via ZeroMQ")

                action = await self.responses[request_id]
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

    def _warmup(self, batch_size: int) -> None:
        """Warm up policy by compiling for the fixed batch_size.

        Since we always pad batches to batch_size, we only need to compile once.
        This avoids JIT compilation delays during inference.
        """
        logger.info("Warming up policy...")
        observation = self._policy.make_example()

        requests = []

        requests.append(
            InferRequest(robot_id="test_robot", observation=observation, infer_type=InferType.SYNC, params=None)
        )
        requests.append(
            InferRequest(
                robot_id="test_robot",
                observation=observation,
                infer_type=InferType.INFERENCE_TIME_RTC,
                params=RTCParams(
                    prev_action=self._policy.make_example_actions(),
                    s_param=5,
                    d_param=3,
                ),
            )
        )
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
