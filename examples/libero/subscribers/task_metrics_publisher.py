from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from openpi_client.client import BidirectionalWebsocket
from openpi_client.runtime import subscriber as _subscriber
from openpi_client.schemas import Action
from openpi_client.schemas import Observation
from typing_extensions import override

if TYPE_CHECKING:
    from libero.libero import benchmark

    from examples.libero.env import LiberoSimEnvironment

logger = logging.getLogger(__name__)


class TaskMetricsPublisher(_subscriber.Subscriber):
    """Publishes per-episode task outcomes to the policy server dashboard."""

    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        environment: LiberoSimEnvironment,
        task_suite_name: str,
        task_id: int,
        task: benchmark.Task,
    ) -> None:
        self._ws_client = ws_client
        self._environment = environment
        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._task_language = task.language
        self._episode_start_perf: float | None = None
        self._steps_taken = 0

    @override
    def on_episode_start(self) -> None:
        self._episode_start_perf = time.perf_counter()
        self._steps_taken = 0

    @override
    def on_step(self, observation: Observation, action: Action) -> None:
        self._steps_taken += 1

    @override
    def on_episode_end(self) -> None:
        if self._episode_start_perf is None:
            return

        duration_s = max(0.0, time.perf_counter() - self._episode_start_perf)
        try:
            self._ws_client.send_task_result(
                task_suite_name=self._task_suite_name,
                task_id=self._task_id,
                episode_idx=self._environment.episode_idx,
                success=self._environment.current_success,
                duration_s=duration_s,
                steps_taken=self._steps_taken,
                task_language=self._task_language,
            )
        except Exception:
            logger.warning(
                "Failed to publish task_result for task %s/%s",
                self._task_suite_name,
                self._task_id,
            )
