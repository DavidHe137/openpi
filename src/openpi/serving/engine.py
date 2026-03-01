from collections.abc import Callable
import logging
import multiprocessing as mp
import time

from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
import zmq

from openpi.serving.schemas import CompletionNotification
from openpi.serving.schemas import SlotRequest
from openpi.serving.slots import RobotSlots

logger = logging.getLogger(__name__)


def _run_gpu_worker(
    policy_factory: Callable,
    max_batch_size: int,
    slots: RobotSlots,
    batch_queue: mp.Queue,
    gpu_out_ep: str,
    result_ep: str,
    ready_event: mp.Event,
) -> None:
    """Loads model, then loops: recv batch → read obs from shared memory → infer → send results.

    Sends InferResponse objects directly to WS (gpu_out_ep) and small CompletionNotifications
    to the scheduler (result_ep) for state updates. These are decoupled so ILP solving in the
    scheduler cannot delay response delivery to clients.
    """
    logger.info("GPU worker starting")

    policy = policy_factory()
    policy.warmup(max_batch_size)

    ctx = zmq.Context()

    # Direct path to WS _router_task (WS process binds)
    response_sock = ctx.socket(zmq.PUSH)
    response_sock.connect(gpu_out_ep)

    # State-update path to scheduler (scheduler binds)
    notify_sock = ctx.socket(zmq.PUSH)
    notify_sock.connect(result_ep)

    ready_event.set()
    logger.info("GPU worker ready")

    while True:
        slot_reqs: list[SlotRequest] = batch_queue.get()  # blocking

        infer_requests = [
            InferRequest(
                robot_id=sr.robot_id,
                observation=slots.read_obs(sr.slot_index),
                start_step=sr.start_step,
                request_timestamp=sr.request_timestamp,
                deadline=sr.deadline,
                infer_type=sr.infer_type,
                params=sr.params,
                noise=sr.noise,
            )
            for sr in slot_reqs
        ]

        logger.info("GPU worker inferring batch of %d", len(infer_requests))
        t0 = time.perf_counter()
        actions = policy.infer_batch(infer_requests)
        t1 = time.perf_counter()

        responses = [
            InferResponse(
                robot_id=sr.robot_id,
                request_id=sr.request_id,
                start_step=sr.start_step,
                request_timestamp=sr.request_timestamp,
                execution_horizon=len(action_dict["actions"]),
                actions=action_dict["actions"],
                noise=action_dict["noise"],
                server_compute_ms=(t1 - t0) * 1e3,
            )
            for sr, action_dict in zip(slot_reqs, actions, strict=True)
        ]

        # Send responses directly to WS — not via scheduler, so ILP latency doesn't affect clients
        response_sock.send_pyobj(responses)

        # Notify scheduler of completions for state updates (can be delayed by ILP, that's fine)
        notifications = [CompletionNotification(robot_id=r.robot_id, start_step=r.start_step) for r in responses]
        notify_sock.send_pyobj(notifications)
