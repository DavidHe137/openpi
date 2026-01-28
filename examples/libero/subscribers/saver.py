import logging
import pathlib
import time
import imageio
import numpy as np

from typing import List, Dict, Any
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
from examples.libero.env import LiberoSimEnvironment

logger = logging.getLogger(__name__)


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@dataclass(frozen=True)
class Result(JSONDataclass):
    success: bool
    robot_idx: int
    task_suite_name: str
    task_id: int
    episode_idx: int


class Saver(_subscriber.Subscriber):
    """Saves episode data."""

    # TODO: probably pass metadata with dataclass
    def __init__(
        self,
        out_dir: pathlib.Path,
        environment: LiberoSimEnvironment,
        action_chunk_broker: ActionChunkBroker,
        task_suite_name: str,
        task_id: int,
        robot_idx: int,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._robot_idx = robot_idx
        self._environment = environment
        self._action_chunk_broker = action_chunk_broker
        self._timestamps: List[Timestamp] = []
        self._images: List[np.ndarray] = []
        self._control_hz = environment.control_hz

    @override
    def on_episode_start(self) -> None:
        self._timestamps = []
        self._action_chunk_indices = []
        self._images = []

    @override
    def on_step(self, observation: Observation, action: Action) -> None:
        self._timestamps.append(
            Timestamp(
                timestamp=time.perf_counter(),
                action_chunk_index=action.action_chunk_index,
                action_index=action.index_in_chunk,
                env_step=observation.step,
            )
        )
        self._images.append(observation.image)

    @override
    def on_episode_end(self) -> None:
        out_folder = self._get_out_folder()

        self._save_metadata(out_folder)
        self._save_timestamps(out_folder)
        self._save_action_chunks(out_folder)
        self._save_video(out_folder)
        self._save_debug_data(out_folder)

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
            task_suite_name=self._task_suite_name,
            task_id=self._task_id,
            episode_idx=self._environment.episode_idx,
        )
        result.to_json(out_folder / "metadata.json")

    def _save_timestamps(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving timestamps to {out_folder / 'timestamps.csv'}")
        Timestamp.to_csv(self._timestamps, out_folder / "timestamps.csv")

    def _save_action_chunks(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving action chunks to {out_folder / 'action_chunks.csv'}")
        ActionChunk.to_csv(
            self._action_chunk_broker.action_chunks,
            out_folder / "action_chunks.csv",
        )

    def _save_video(self, out_folder: pathlib.Path) -> None:
        logger.info(f"Saving video to {out_folder / 'out.mp4'}")
        imageio.mimwrite(
            out_folder / "out.mp4",
            [np.asarray(x) for x in self._images],
            fps=self._control_hz,  # NOTE: saving in control hz fps for now
        )

    def _save_debug_data(self, out_folder: pathlib.Path) -> None:
        action_chunks = self._action_chunk_broker.action_chunks
        debug_data_dir = out_folder / "debug_data"

        has_debug_data = any(chunk.debug_data for chunk in action_chunks)
        if not has_debug_data:
            logger.debug("No debug data to save")
            return

        debug_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving debug data to {debug_data_dir}")

        for i, chunk in enumerate(action_chunks):
            if not chunk.debug_data:
                continue

            flat_data = _flatten_dict(chunk.debug_data)

            chunk_file = debug_data_dir / f"chunk_{i:04d}.npy"
            np.save(chunk_file, flat_data, allow_pickle=True)
            logger.debug(f"Saved debug data for chunk {i} to {chunk_file}")
