import logging
import pathlib
import time
import imageio
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict
from dataclasses import dataclass
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.schemas import (
    Timestamp,
    JSONDataclass,
    ActionChunk,
    Observation,
    Action,
)
from libero.libero import benchmark
from examples.libero.env import LiberoSimEnvironment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result(JSONDataclass):
    robot_idx: int
    success: bool
    steps_taken: int
    task_suite_name: str
    task_id: int
    task_language: str
    episode_idx: int


class Saver(_subscriber.Subscriber):
    """Saves episode data."""

    def __init__(
        self,
        out_dir: pathlib.Path,
        environment: LiberoSimEnvironment,
        action_chunk_broker: ActionChunkBroker,
        task_suite_name: str,
        task_id: int,
        task: benchmark.Task,
        robot_idx: int,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._task = task
        self._robot_idx = robot_idx
        self._environment = environment
        self._action_chunk_broker = action_chunk_broker
        self._timestamps: List[Timestamp] = []
        self._control_hz = environment.control_hz
        self._observations_buffer: Dict[int, Observation] = {}

    @override
    def on_episode_start(self) -> None:
        self._timestamps = []
        self._action_chunk_indices = []
        self._observations_buffer = {}
        self._actions_left_snapshot: list[int] = []
        self._cost_history: list[float] = []

    @override
    def on_step(self, observation: Observation, action: Action) -> None:
        # Store observation for debug data and video reconstruction
        self._observations_buffer[observation.step] = observation

        self._timestamps.append(
            Timestamp(
                timestamp=time.perf_counter(),
                action_chunk_index=action.action_chunk_index,
                action_index=action.index_in_chunk,
                env_step=observation.step,
            )
        )

        # Snapshot broker queue length after this step's action was consumed.
        # broker.infer() already recorded into _actions_left_history; mirror it here
        # by reading the latest entry (avoids a second lock acquisition).
        history = self._action_chunk_broker.actions_left_history
        self._actions_left_snapshot.append(history[-1] if history else 0)

        # Cost: elapsed time from when the observation was sent for inference
        # until this step executes that chunk's action.
        if action.action_chunk_index is not None:
            chunk = self._action_chunk_broker.action_chunks[action.action_chunk_index]
            cost = time.time() - chunk.request_timestamp
        else:
            cost = float("nan")
        self._cost_history.append(cost)

    @override
    def on_episode_end(self) -> None:
        out_folder = self._get_out_folder()

        self._save_metadata(out_folder)
        self._save_timestamps(out_folder)
        self._save_action_chunks(out_folder)
        self._save_video(out_folder)
        self._save_debug_data(out_folder)
        self._save_actions_left(out_folder)
        self._save_cost_history(out_folder)

    def _get_out_folder(self) -> pathlib.Path:
        robot_folder = self._out_dir / str(self._robot_idx)
        pathlib.Path(robot_folder).mkdir(parents=True, exist_ok=True)

        existing = list(robot_folder.iterdir())
        next_idx = (
            max([int(p.name.split("_")[0]) for p in existing if p.is_dir()], default=-1)
            + 1
        )
        success_str = "success" if self._environment.current_success else "failure"
        out_folder = (
            robot_folder
            / f"{next_idx}_{self._task_suite_name}_{self._task_id}_{success_str}"
        )
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(out_folder)

    def _save_metadata(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving metadata to {out_folder / 'metadata.json'}")
        result = Result(
            success=self._environment.current_success,
            robot_idx=self._robot_idx,
            steps_taken=len(self._timestamps),
            task_suite_name=self._task_suite_name,
            task_id=self._task_id,
            task_language=self._task.language,
            episode_idx=self._environment.episode_idx,
        )
        result.to_json(out_folder / "metadata.json")

    def _save_timestamps(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving timestamps to {out_folder / 'timestamps.csv'}")
        Timestamp.to_csv(self._timestamps, out_folder / "timestamps.csv")

    def _save_action_chunks(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving action chunks to {out_folder}")
        action_chunks = self._action_chunk_broker.action_chunks
        ActionChunk.to_csv(action_chunks, out_folder / "action_chunks.csv")
        ActionChunk.to_parquet(action_chunks, out_folder / "action_chunks.parquet")

    def _save_video(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving video to {out_folder / 'out.mp4'}")
        # Reconstruct images from observations buffer
        images = [obs.image for obs in self._observations_buffer.values()]
        imageio.mimwrite(
            out_folder / "out.mp4",
            [np.asarray(x) for x in images],
            fps=self._control_hz,  # NOTE: saving in control hz fps for now
        )

    def _save_debug_data(self, out_folder: pathlib.Path) -> None:
        """Save debug data as a single .npz file with observations, noise, and actions."""
        action_chunks = self._action_chunk_broker.action_chunks

        # Check if we have noise data
        has_noise = any(chunk.noise is not None for chunk in action_chunks)
        if not has_noise:
            logger.debug("No debug data to save (no noise present)")
            return

        debug_data_file = out_folder / "debug_data.npz"
        logger.info(f"Saving debug data to {debug_data_file}")

        # Build data dict for .npz file
        data_to_save = {}

        # Save the initial state used for this episode
        data_to_save["initial_state"] = self._environment.current_initial_state

        for i, chunk in enumerate(action_chunks):
            prefix = f"chunk_{i:04d}"

            # Save observation that triggered this inference
            obs = self._observations_buffer.get(chunk.observation_step)
            if obs is not None:
                data_to_save[f"{prefix}/observation/state"] = obs.state
                data_to_save[f"{prefix}/observation/image"] = obs.image
                data_to_save[f"{prefix}/observation/wrist_image"] = obs.wrist_image
                if hasattr(obs, "prompt"):
                    data_to_save[f"{prefix}/observation/prompt"] = obs.prompt
            else:
                logger.warning(
                    f"No observation found for chunk {i} at step {chunk.observation_step}"
                )

            # Save noise (should always be present)
            if chunk.noise is not None:
                data_to_save[f"{prefix}/noise"] = chunk.noise

            # Save actions (final robot-ready actions)
            data_to_save[f"{prefix}/actions"] = chunk.actions

            # Save metadata
            # TODO: fix this
            data_to_save[f"{prefix}/start_step"] = chunk.observation_step
            data_to_save[f"{prefix}/execution_horizon"] = chunk.execution_horizon

        # Save as compressed npz
        np.savez_compressed(debug_data_file, **data_to_save)
        logger.info(f"Saved {len(action_chunks)} chunks to {debug_data_file}")

    def _save_actions_left(self, out_folder: pathlib.Path) -> None:
        path = out_folder / "actions_left.npy"
        np.save(path, np.array(self._actions_left_snapshot, dtype=np.int32))
        logger.info(f"Saved actions_left to {path}")

    def _save_cost_history(self, out_folder: pathlib.Path) -> None:
        costs = np.array(self._cost_history, dtype=np.float64)
        npy_path = out_folder / "cost_history.npy"
        np.save(npy_path, costs)
        logger.info(f"Saved cost_history to {npy_path}")

        plot_path = out_folder / "cost_history.png"
        steps = np.arange(len(costs))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, costs, linewidth=0.8, color="steelblue")
        ax.set_xlabel("Environment step")
        ax.set_ylabel("Cost (s)")
        ax.set_title(
            f"Cost per step — robot {self._robot_idx} | "
            f"{self._task_suite_name} task {self._task_id}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved cost_history plot to {plot_path}")
