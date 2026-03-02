from __future__ import annotations

from collections.abc import Callable
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event
import signal
import time

from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
import zmq

from openpi.serving.schemas import BatchProfile
from openpi.serving.schemas import SlotRequest
from openpi.serving.slots import RobotSlots
from openpi.shared import logging_config

logger = logging.getLogger(__name__)


def _profile_and_send(policy, max_batch_size: int, notify_sock: zmq.Socket) -> None:
    """Profile inference latency for each batch size and send a BatchProfile to the scheduler."""
    logger.info("Profiling batch latency for sizes 1..%d", max_batch_size)
    profile: dict[int, float] = {}

    request = policy.make_infer_request()
    for batch_size in range(1, max_batch_size + 1):
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            policy.infer_batch([request] * batch_size)
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1e3
            latencies.append(latency)
        profile[batch_size] = sum(latencies) / len(latencies)
        logger.info("  batch_size=%d → %.1f ms", batch_size, profile[batch_size])
    notify_sock.send_pyobj(BatchProfile(latency_ms=profile))
    logger.info("Sent batch profile to scheduler")


def _run_gpu_worker(
    policy_factory: Callable,
    max_batch_size: int,
    slots: RobotSlots,
    batch_queue: mp.Queue,
    gpu_out_ep: str,
    result_ep: str,
    ready_event: Event,
    log_queue: mp.Queue | None = None,
) -> None:
    """Loads model, then loops: recv batch → read obs from shared memory → infer → send results.

    Sends InferResponse objects directly to WS (gpu_out_ep) and small CompletionNotifications
    to the scheduler (result_ep) for state updates. These are decoupled so ILP solving in the
    scheduler cannot delay response delivery to clients.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if log_queue is not None:
        logging_config.setup_worker_logging(log_queue, process_name="gpu-worker")

    logger.info("GPU worker starting")

    # FIXME: really doesn't belong here
    # GPU devices may not be immediately visible after container snapshot restore.
    # Retry until CUDA becomes available (or we exhaust retries).
    cuda_max_wait_s = 120
    cuda_retry_s = 5
    policy = None
    for attempt in range(cuda_max_wait_s // cuda_retry_s):
        try:
            policy = policy_factory()
            break
        except RuntimeError as e:
            if "No visible GPU devices" in str(e) and attempt < cuda_max_wait_s // cuda_retry_s - 1:
                logger.warning(
                    "CUDA not yet available (attempt %d/%d), retrying in %ds…",
                    attempt + 1,
                    cuda_max_wait_s // cuda_retry_s,
                    cuda_retry_s,
                )
                time.sleep(cuda_retry_s)
            else:
                raise
    assert policy is not None
    policy.warmup(max_batch_size)

    ctx = zmq.Context()

    # Direct path to WS _router_task (WS process binds)
    response_sock = ctx.socket(zmq.PUSH)
    response_sock.connect(gpu_out_ep)

    # State-update path to scheduler (scheduler binds)
    notify_sock = ctx.socket(zmq.PUSH)
    notify_sock.connect(result_ep)

    _profile_and_send(policy, max_batch_size, notify_sock)

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

        logger.debug("Inferring batch of %d", len(infer_requests))
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

        # FIXME: might not be needed
        # Notify scheduler of completions for state updates (can be delayed by ILP, that's fine)
        # notifications = [CompletionNotification(robot_id=r.robot_id, start_step=r.start_step) for r in responses]
        # notify_sock.send_pyobj(notifications)
