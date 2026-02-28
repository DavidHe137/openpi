from collections.abc import Callable
import logging
import time

import zmq

from openpi.serving.schemas import ArrivedRequest

logger = logging.getLogger(__name__)

# TODO: load model
# TODO: should just consume a shared queue


def _run_gpu_worker(
    policy_factory: Callable,
    max_batch_size: int,
    gpu_in_ep: str,
    gpu_out_ep: str,
    gpu_ready_ep: str,
) -> None:
    """Loads model, signals ready, then loops: recv batch → infer → send results."""

    logger.info("GPU worker starting")

    policy = policy_factory()
    policy.warmup(max_batch_size)

    ctx = zmq.Context()

    # Bind input (receives batches from scheduler)
    batch_sock = ctx.socket(zmq.PULL)
    batch_sock.bind(gpu_in_ep)

    # Bind output (sends results to scheduler)
    result_sock = ctx.socket(zmq.PUSH)
    result_sock.bind(gpu_out_ep)

    # Signal ready to main process, then close the ready socket
    ready_sock = ctx.socket(zmq.PUSH)
    ready_sock.connect(gpu_ready_ep)
    ready_sock.send_string("ready")
    ready_sock.setsockopt(zmq.LINGER, 1000)
    ready_sock.close()

    logger.info("GPU worker ready")

    try:
        while True:
            batch: list[ArrivedRequest] = batch_sock.recv_pyobj()
            logger.info("GPU worker inferring batch of %d", len(batch))
            batch_start = time.perf_counter()
            actions = policy.infer_batch([r.infer_request for r in batch])
            batch_end = time.perf_counter()
            result_sock.send_pyobj((batch, actions, batch_start, batch_end))
    finally:
        logger.info("GPU worker shutting down")
        batch_sock.setsockopt(zmq.LINGER, 0)
        result_sock.setsockopt(zmq.LINGER, 1000)
        batch_sock.close()
        result_sock.close()
        ctx.term()
