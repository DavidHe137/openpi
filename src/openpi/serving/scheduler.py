import enum
import logging
import random

import numpy as np
from openpi_client.messages import InferResponse
from openpi_client.messages import ResetRequest
import zmq

from openpi.serving.schemas import ArrivedRequest

DEFAULT_EXECUTION_HORIZON = 10
DISTANCE_THRESHOLD = 0.5
MINIMUM_EXECUTION_HORIZON = 5

logger = logging.getLogger(__name__)


# TODO: took out metrics for now, will add back later
class SchedulingAlgorithm(str, enum.Enum):
    """Scheduling algorithm for batching requests."""

    GREEDY = "greedy"
    ROUND_ROBIN = "round_robin"
    RANDOM_BATCH = "random"


class RequestScheduler:
    def __init__(self, algorithm: SchedulingAlgorithm = SchedulingAlgorithm.GREEDY):
        self._algorithm = algorithm
        self._deadlines: dict[str, float] = {}
        self._last_response: dict[str, InferResponse] = {}

        # Round-robin state
        self._rr_index: int = 0
        self._rr_robot_order: list[str] = []

    def _can_infer(self, request: ArrivedRequest) -> bool:
        robot_id = request.infer_request.robot_id
        stale: bool = robot_id in self._deadlines and request.infer_request.deadline < self._deadlines[robot_id]
        inside_execution_minimum: bool = (
            robot_id in self._last_response
            and request.infer_request.start_step < self._last_response[robot_id].start_step + MINIMUM_EXECUTION_HORIZON
        )

        return not stale and not inside_execution_minimum

    def _register_robot(self, robot_id: str) -> None:
        if robot_id not in self._rr_robot_order:
            self._rr_robot_order.append(robot_id)

    def get_next_batch(self, snapshot: dict[str, ArrivedRequest], max_batch_size: int) -> list[ArrivedRequest]:
        for robot_id in snapshot:
            self._register_robot(robot_id)

        candidates: list[ArrivedRequest] = [request for request in snapshot.values() if self._can_infer(request)]
        if not candidates:
            return []

        if self._algorithm == SchedulingAlgorithm.GREEDY:
            return self._greedy_batch(candidates, max_batch_size)
        if self._algorithm == SchedulingAlgorithm.ROUND_ROBIN:
            return self._round_robin_batch(candidates, max_batch_size)
        return self._random_batch(candidates, max_batch_size)

    def _greedy_batch(self, candidates: list[ArrivedRequest], max_batch_size: int) -> list[ArrivedRequest]:
        """Greedy (earliest deadline first): sort by deadline, take up to max_batch_size."""
        return sorted(candidates, key=lambda x: x.infer_request.deadline)[:max_batch_size]

    def _round_robin_batch(self, candidates: list[ArrivedRequest], max_batch_size: int) -> list[ArrivedRequest]:
        """Round Robin: cycle through robots starting from current pointer, fill to max_batch_size."""
        candidate_by_robot = {req.infer_request.robot_id: req for req in candidates}
        n_robots = len(self._rr_robot_order)
        if n_robots == 0:
            return []

        batch: list[ArrivedRequest] = []
        checked = 0
        idx = self._rr_index % n_robots

        while len(batch) < max_batch_size and checked < n_robots:
            robot_id = self._rr_robot_order[idx]
            if robot_id in candidate_by_robot:
                batch.append(candidate_by_robot[robot_id])
            idx = (idx + 1) % n_robots
            checked += 1

        self._rr_index = idx
        return batch

    def _random_batch(self, candidates: list[ArrivedRequest], max_batch_size: int) -> list[ArrivedRequest]:
        """Random: randomly select up to max_batch_size from candidates."""
        k = min(max_batch_size, len(candidates))
        return random.sample(candidates, k)

    def update_deadlines(self, batch: list[ArrivedRequest]) -> None:
        for request in batch:
            self._deadlines[request.infer_request.robot_id] = request.infer_request.deadline

    def calculate_execution_horizon(self, robot_id: str, actions: np.ndarray) -> int:
        return len(actions)

    def update_last_response(self, robot_id: str, response: InferResponse) -> None:
        self._last_response[robot_id] = response

    def reset_robot(self, robot_id: str) -> None:
        self._last_response.pop(robot_id, None)
        self._deadlines.pop(robot_id, None)


def _run_scheduler(
    sched_in_ep: str,
    sched_out_ep: str,
    gpu_in_ep: str,
    gpu_out_ep: str,
    max_batch_size: int,
    algorithm: SchedulingAlgorithm,
) -> None:
    """Owns all robot state; dispatches batches to GPU; routes responses to WS."""
    logger.info("Scheduler starting (algorithm=%s)", algorithm.value)

    scheduler = RequestScheduler(algorithm=algorithm)

    # TODO: set up sockets
    ctx = zmq.Context()

    # Bind input from WS (ArrivedRequest / ResetMessage)
    req_sock = ctx.socket(zmq.PULL)
    req_sock.bind(sched_in_ep)

    # Bind output to WS (InferResponse / BatchMetrics)
    resp_sock = ctx.socket(zmq.PUSH)
    resp_sock.bind(sched_out_ep)

    # Connect to GPU (send batches)
    batch_sock = ctx.socket(zmq.PUSH)
    batch_sock.connect(gpu_in_ep)

    # Connect to GPU (receive results)
    result_sock = ctx.socket(zmq.PULL)
    result_sock.connect(gpu_out_ep)

    logger.info("Scheduler ready")

    try:
        while True:
            # --- Drain incoming requests / resets from WS (non-blocking) ---
            while True:
                try:
                    msg = req_sock.recv_pyobj(zmq.NOBLOCK)
                    if isinstance(msg, ResetRequest):
                        scheduler.reset_robot(msg.robot_id)
                        logger.info("Reset robot %s", msg.robot_id)
                    else:
                        arrived: ArrivedRequest = msg
                        scheduler.update(arrived)
                except zmq.Again:
                    break

            # --- Drain inference results from GPU (non-blocking) ---
            while True:
                try:
                    batch, actions, batch_start, batch_end = result_sock.recv_pyobj(zmq.NOBLOCK)

                    batch_robot_ids: list[str] = []
                    batch_start_steps: list[int] = []
                    batch_execution_horizons: list[int] = []

                    for req, action_dict in zip(batch, actions, strict=True):
                        robot_id = req.infer_request.robot_id
                        execution_horizon = scheduler.calculate_execution_horizon(robot_id, action_dict["actions"])
                        response = InferResponse(
                            robot_id=robot_id,
                            request_id=req.request_id,
                            start_step=req.infer_request.start_step,
                            request_timestamp=req.infer_request.request_timestamp,
                            execution_horizon=execution_horizon,
                            actions=action_dict["actions"],
                            noise=action_dict["noise"],
                            server_compute_ms=(batch_end - batch_start) * 1e3,
                        )
                        scheduler.update_last_response(robot_id, response)
                        batch_robot_ids.append(robot_id)
                        batch_start_steps.append(req.infer_request.start_step)
                        batch_execution_horizons.append(execution_horizon)
                        resp_sock.send_pyobj(response)

                    scheduler.update_deadlines(batch)

                except zmq.Again:
                    break

    finally:
        logger.info("Scheduler shutting down")
        for sock in (req_sock, batch_sock, result_sock):
            sock.setsockopt(zmq.LINGER, 0)
            sock.close()
        resp_sock.setsockopt(zmq.LINGER, 1000)
        resp_sock.close()
        ctx.term()


# TODO: how to have scheduler send to zmq?
# async def _router_task(
#     resp_sock: zmq.Socket,
# ) -> None:
#     while True:
#         try:
#             message = await resp_sock.recv_pyobj()

#             response: InferResponse = message
#             # conn_id = routing.pop(response.request_id, None)
#             if conn_id is None:
#                 logger.debug("No routing for request %d", response.request_id)
#                 continue

#             conn = connections.get(conn_id)
#             if conn is None:
#                 logger.debug("Connection %s closed, dropping response", conn_id)
#                 continue

#             await conn.response_queue.put(response)

#         except asyncio.CancelledError:
#             break
#         except Exception:
#             logger.exception("Error in router task")
