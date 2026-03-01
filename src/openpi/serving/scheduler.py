import logging
import multiprocessing as mp

from openpi_client.messages import ResetRequest
import zmq

from openpi.scheduling import RequestScheduler
from openpi.scheduling.baselines import GreedyScheduler
from openpi.scheduling.baselines import RandomBatchScheduler
from openpi.scheduling.baselines import RoundRobinScheduler
from openpi.serving.schemas import CompletionNotification
from openpi.serving.schemas import SlotRequest

logger = logging.getLogger(__name__)


_SCHEDULER_REGISTRY: dict[str, type[RequestScheduler]] = {
    "greedy": GreedyScheduler,
    "round_robin": RoundRobinScheduler,
    "random": RandomBatchScheduler,
}


def _run_scheduler(
    sched_in_ep: str,
    result_ep: str,
    batch_queue: mp.Queue,
    max_batch_size: int,
    algorithm: str,
    ready_event: mp.Event,
) -> None:
    """Owns all robot state; dispatches batches to GPU via mp.Queue.

    GPU sends InferResponses directly to WS (not via this process), so ILP solving here
    cannot delay client response delivery. This process only receives small CompletionNotifications
    from GPU for state bookkeeping.
    """
    logger.info("Scheduler starting (algorithm=%s)", algorithm)

    cls = _SCHEDULER_REGISTRY.get(algorithm)
    if cls is None:
        raise ValueError(f"Unknown scheduling algorithm {algorithm!r}, expected one of: {list(_SCHEDULER_REGISTRY)}")
    scheduler = cls(max_batch_size=max_batch_size)

    ctx = zmq.Context()

    req_sock = ctx.socket(zmq.PULL)
    req_sock.bind(sched_in_ep)  # WS main connects

    result_sock = ctx.socket(zmq.PULL)
    result_sock.bind(result_ep)  # GPU connects

    poller = zmq.Poller()
    poller.register(req_sock, zmq.POLLIN)
    poller.register(result_sock, zmq.POLLIN)

    ready_event.set()
    logger.info("Scheduler ready")

    while True:
        poller.poll(timeout=1)  # 1ms — ensures dispatch runs regularly

        # Phase 1: drain WS messages
        while req_sock.poll(0):
            msg = req_sock.recv_pyobj(zmq.NOBLOCK)
            if isinstance(msg, ResetRequest):
                scheduler.reset_robot(msg.robot_id)
            elif isinstance(msg, SlotRequest):
                scheduler.update(msg)

        # Phase 2: drain GPU completion notifications (small — no arrays)
        while result_sock.poll(0):
            notifications: list[CompletionNotification] = result_sock.recv_pyobj(zmq.NOBLOCK)
            for n in notifications:
                scheduler.notify_complete(n)

        # Phase 3: dispatch next batch (only if queue has room)
        # NOTE: this is where ILP solving will go — it may block for a long time,
        # but that's fine because response delivery to clients bypasses this process entirely.
        if not batch_queue.full():
            batch = scheduler.schedule()
            if batch:
                scheduler.update_deadlines(batch)
                batch_queue.put_nowait(batch)
