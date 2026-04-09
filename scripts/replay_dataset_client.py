import dataclasses
import datetime
import json
import logging
import math
import pathlib
import time
from typing import Any, Literal

from datasets import DownloadConfig
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import matplotlib
import numpy as np
from openpi_client.client import BidirectionalWebsocket
from openpi_client.messages import InferType
from openpi_client.schemas import LiberoObservation
import pandas as pd
from PIL import Image
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

STATE_KEYS = ("state", "observation/state", "observation.state")
IMAGE_KEYS = ("image", "observation/image", "observation.image")
WRIST_IMAGE_KEYS = ("wrist_image", "observation/wrist_image", "observation.wrist_image")
ACTION_KEYS = ("actions", "action")
PROMPT_KEYS = ("prompt", "task", "language_instruction", "language", "instruction")


@dataclasses.dataclass
class Args:
    # Server connection.
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str | None = None
    robot_id: str = "dataset_replay"
    control_hz: float = 10.0

    # Dataset selection.
    repo_id: str = "solace222/sort-and-throw-the-legos-20260409"
    dataset_root: pathlib.Path | None = None
    episode_index: int = 0
    start_step: int = 0
    max_steps: int | None = None
    prompt: str | None = None
    local_files_only: bool = False

    # Output.
    output_dir: pathlib.Path = pathlib.Path("data/replay_dataset_client")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


def _as_scalar(value: Any) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar value, got shape {array.shape}")
    return int(array.reshape(()).item())


def _find_first_key(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of {keys} found in row keys: {sorted(data.keys())}")


def _maybe_find_first_key(data: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _to_numpy_vector(value: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _to_numpy_image(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        array = np.asarray(value)
    elif isinstance(value, dict):
        if "array" in value:
            array = np.asarray(value["array"])
        elif "bytes" in value and value["bytes"] is not None:
            from io import BytesIO

            array = np.asarray(Image.open(BytesIO(value["bytes"])))
        elif "path" in value and value["path"]:
            array = np.asarray(Image.open(value["path"]))
        else:
            raise ValueError(f"Unsupported image dictionary keys: {sorted(value.keys())}")
    else:
        array = np.asarray(value)

    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {array.shape}")

    # Handle CHW arrays from some dataset decoders.
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)

    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.5:
            array = 255.0 * array
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def _normalize_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    normalized = tasks.copy()
    if "task" in normalized.columns and normalized.index.name != "task":
        normalized = normalized.set_index("task")
    return normalized


def _load_tasks(args: Args) -> pd.DataFrame | None:
    task_path: pathlib.Path | None = None

    if args.dataset_root is not None:
        candidate = args.dataset_root / "meta" / "tasks.parquet"
        if candidate.exists():
            task_path = candidate
    else:
        try:
            task_path = pathlib.Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename="meta/tasks.parquet",
                    local_files_only=args.local_files_only,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on local cache/network
            logger.warning("Unable to load tasks.parquet for %s: %s", args.repo_id, exc)

    if task_path is None:
        return None

    return _normalize_tasks(pd.read_parquet(task_path))


def _load_dataset_rows(args: Args) -> tuple[list[dict[str, Any]], pd.DataFrame | None, str]:
    download_config = DownloadConfig(local_files_only=args.local_files_only)

    if args.dataset_root is not None:
        data_dir = args.dataset_root / "data"
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")
        logger.info("Loading %d parquet shard(s) from %s", len(parquet_files), data_dir)
        dataset = load_dataset(
            "parquet",
            data_files=[str(path) for path in parquet_files],
            split="train",
            download_config=download_config,
        )
        dataset_source = str(args.dataset_root)
    else:
        logger.info("Loading dataset %s", args.repo_id)
        dataset = load_dataset(args.repo_id, split="train", download_config=download_config)
        dataset_source = args.repo_id

    if "episode_index" not in dataset.column_names:
        raise KeyError(f"Dataset is missing 'episode_index'. Columns: {dataset.column_names}")

    episode_dataset = dataset.filter(
        lambda episode_index: int(episode_index) == args.episode_index,
        input_columns=["episode_index"],
        desc=f"Filtering episode {args.episode_index}",
    )
    if len(episode_dataset) == 0:
        raise ValueError(f"Episode {args.episode_index} not found in dataset {dataset_source}")

    for sort_key in ("frame_index", "index", "timestamp"):
        if sort_key in episode_dataset.column_names:
            episode_dataset = episode_dataset.sort(sort_key)
            break

    if args.start_step < 0:
        raise ValueError("--start-step must be >= 0")
    if args.start_step >= len(episode_dataset):
        raise ValueError(
            f"--start-step={args.start_step} is past the end of episode {args.episode_index} with {len(episode_dataset)} frames"
        )

    end_step = len(episode_dataset)
    if args.max_steps is not None:
        end_step = min(end_step, args.start_step + args.max_steps)

    episode_dataset = episode_dataset.select(range(args.start_step, end_step))
    rows = [episode_dataset[i] for i in range(len(episode_dataset))]
    return rows, _load_tasks(args), dataset_source


def _prompt_from_row(row: dict[str, Any], tasks: pd.DataFrame | None, prompt_override: str | None) -> str:
    if prompt_override is not None:
        return prompt_override

    prompt_value = _maybe_find_first_key(row, PROMPT_KEYS)
    if prompt_value is not None:
        return str(prompt_value)

    if tasks is not None and "task_index" in row:
        task_index = _as_scalar(row["task_index"])
        if "task_index" in tasks.columns:
            matches = tasks[tasks["task_index"] == task_index]
            if not matches.empty:
                return str(matches.index[0])

    raise ValueError(
        "Could not determine a prompt from the dataset row. Pass --prompt or provide a dataset with prompt/task metadata."
    )


def _build_observation(row: dict[str, Any], tasks: pd.DataFrame | None, prompt_override: str | None) -> LiberoObservation:
    step = 0
    for key in ("frame_index", "index", "timestamp"):
        if key in row:
            step = _as_scalar(row[key]) if key != "timestamp" else int(round(float(np.asarray(row[key]).item())))
            break

    return LiberoObservation(
        state=_to_numpy_vector(_find_first_key(row, STATE_KEYS)),
        image=_to_numpy_image(_find_first_key(row, IMAGE_KEYS)),
        wrist_image=_to_numpy_image(_find_first_key(row, WRIST_IMAGE_KEYS)),
        prompt=_prompt_from_row(row, tasks, prompt_override),
        step=step,
    )


def _extract_actions(rows: list[dict[str, Any]]) -> np.ndarray:
    actions = [_to_numpy_vector(_find_first_key(row, ACTION_KEYS)) for row in rows]
    action_dims = {action.shape[0] for action in actions}
    if len(action_dims) != 1:
        raise ValueError(f"Inconsistent action dimensions in dataset rows: {sorted(action_dims)}")
    return np.stack(actions, axis=0).astype(np.float32)


def _run_dir(base_dir: pathlib.Path, episode_index: int) -> pathlib.Path:
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    path = base_dir / f"episode_{episode_index:04d}_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _plot_action_overlay(frame_indices: np.ndarray, predicted: np.ndarray, ground_truth: np.ndarray, out_path: pathlib.Path) -> None:
    action_dim = ground_truth.shape[1]
    cols = 2
    rows = math.ceil(action_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for dim in range(rows * cols):
        ax = axes.flat[dim]
        if dim >= action_dim:
            ax.axis("off")
            continue
        ax.plot(frame_indices, ground_truth[:, dim], label="ground truth", linewidth=1.5)
        ax.plot(frame_indices, predicted[:, dim], label="predicted", linewidth=1.2)
        ax.set_title(f"Action dim {dim}")
        ax.grid(alpha=0.3)
        if dim % cols == 0:
            ax.set_ylabel("value")

    axes.flat[0].legend(loc="best")
    axes.flat[-1].set_xlabel("frame index")
    fig.suptitle("First Action vs Dataset Action", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_offset_mae(offset_mae: np.ndarray, out_path: pathlib.Path) -> None:
    offsets = np.arange(offset_mae.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    overall = np.nanmean(offset_mae, axis=1)
    ax.plot(offsets, overall, marker="o", linewidth=2, label="mean over dims")
    for dim in range(offset_mae.shape[1]):
        ax.plot(offsets, offset_mae[:, dim], linewidth=1, alpha=0.5, label=f"dim {dim}")
    ax.set_xlabel("prediction offset in chunk")
    ax.set_ylabel("mean absolute error")
    ax.set_title("Chunk Error vs Forecast Offset")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_chunk_mae(frame_indices: np.ndarray, chunk_mae: np.ndarray, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame_indices, chunk_mae, linewidth=1.5)
    ax.set_xlabel("frame index")
    ax.set_ylabel("chunk MAE")
    ax.set_title("Per-Step Chunk MAE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(args: Args) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rows, tasks, dataset_source = _load_dataset_rows(args)
    gt_actions = _extract_actions(rows)
    run_dir = _run_dir(args.output_dir, args.episode_index)

    logger.info("Loaded %d frame(s) for episode %d from %s", len(rows), args.episode_index, dataset_source)
    logger.info("Saving outputs to %s", run_dir)

    ws_client = BidirectionalWebsocket(
        robot_id=args.robot_id,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        control_hz=args.control_hz,
    )
    server_metadata = ws_client.server_metadata
    action_horizon = int(server_metadata.action_horizon)

    frame_indices: list[int] = []
    prompts: list[str] = []
    first_pred_actions: list[np.ndarray] = []
    first_gt_actions: list[np.ndarray] = []
    chunk_mae_per_step: list[float] = []
    server_ms: list[float] = []

    compare_dim = gt_actions.shape[1]
    pred_chunks = np.full((len(rows), action_horizon, compare_dim), np.nan, dtype=np.float32)
    gt_chunks = np.full((len(rows), action_horizon, compare_dim), np.nan, dtype=np.float32)
    offset_errors: list[list[np.ndarray]] = [[] for _ in range(action_horizon)]

    try:
        for row_index, row in enumerate(rows):
            observation = _build_observation(row, tasks, args.prompt)
            frame_indices.append(observation.step)
            prompts.append(observation.prompt)

            ws_client.send(
                obs=observation,
                deadline=time.time(),
                action_start_step=observation.step,
                infer_type=InferType.SYNC,
                execution_horizon=action_horizon,
            )
            response = ws_client.receive()
            ws_client.send_ack(
                request_id=response.request_id,
                receive_time=time.time(),
                execution_start_step=observation.step,
                first_executed_index=0,
            )

            predicted_chunk = np.asarray(response.actions, dtype=np.float32)
            if predicted_chunk.ndim == 3 and predicted_chunk.shape[0] == 1:
                predicted_chunk = predicted_chunk[0]
            if predicted_chunk.ndim != 2:
                raise ValueError(f"Unexpected predicted action chunk shape: {predicted_chunk.shape}")

            if predicted_chunk.shape[1] < compare_dim:
                raise ValueError(
                    f"Predicted action dim {predicted_chunk.shape[1]} is smaller than dataset action dim {compare_dim}"
                )

            predicted_chunk = predicted_chunk[:, :compare_dim]
            overlap = min(action_horizon, len(rows) - row_index)
            gt_chunk = gt_actions[row_index : row_index + overlap]
            pred_overlap = predicted_chunk[:overlap]

            pred_chunks[row_index, :overlap] = pred_overlap
            gt_chunks[row_index, :overlap] = gt_chunk

            first_pred_actions.append(pred_overlap[0].copy())
            first_gt_actions.append(gt_chunk[0].copy())

            abs_error = np.abs(pred_overlap - gt_chunk)
            chunk_mae_per_step.append(float(np.mean(abs_error)))
            for offset in range(overlap):
                offset_errors[offset].append(abs_error[offset])

            if response.inference_end_time > 0 and response.inference_start_time > 0:
                server_ms.append(1000.0 * (response.inference_end_time - response.inference_start_time))

            if (row_index + 1) % 10 == 0 or row_index + 1 == len(rows):
                logger.info("Processed %d/%d frames", row_index + 1, len(rows))
    finally:
        ws_client.close()

    frame_indices_array = np.asarray(frame_indices, dtype=np.int64)
    first_pred_array = np.stack(first_pred_actions, axis=0)
    first_gt_array = np.stack(first_gt_actions, axis=0)
    chunk_mae_array = np.asarray(chunk_mae_per_step, dtype=np.float32)
    offset_mae = np.full((action_horizon, compare_dim), np.nan, dtype=np.float32)
    for offset, errors in enumerate(offset_errors):
        if errors:
            offset_mae[offset] = np.mean(np.stack(errors, axis=0), axis=0)

    summary = {
        "dataset_source": dataset_source,
        "episode_index": args.episode_index,
        "num_steps": len(rows),
        "start_step": args.start_step,
        "server": {
            "host": args.host,
            "port": args.port,
            "config_name": server_metadata.config_name,
            "checkpoint_dir": server_metadata.checkpoint_dir,
            "env": server_metadata.env,
            "action_horizon": action_horizon,
            "action_dim": server_metadata.action_dim,
        },
        "metrics": {
            "first_action_mae": float(np.mean(np.abs(first_pred_array - first_gt_array))),
            "first_action_rmse": float(np.sqrt(np.mean((first_pred_array - first_gt_array) ** 2))),
            "first_action_mae_per_dim": np.mean(np.abs(first_pred_array - first_gt_array), axis=0).tolist(),
            "chunk_mae_mean": float(np.mean(chunk_mae_array)),
            "chunk_mae_std": float(np.std(chunk_mae_array)),
            "forecast_offset_mae_mean": np.nanmean(offset_mae, axis=1).tolist(),
            "server_inference_ms_mean": float(np.mean(server_ms)) if server_ms else None,
            "server_inference_ms_p95": float(np.percentile(server_ms, 95)) if server_ms else None,
        },
        "prompt_example": prompts[0] if prompts else None,
    }

    _plot_action_overlay(frame_indices_array, first_pred_array, first_gt_array, run_dir / "first_action_overlay.png")
    _plot_offset_mae(offset_mae, run_dir / "forecast_offset_mae.png")
    _plot_chunk_mae(frame_indices_array, chunk_mae_array, run_dir / "chunk_mae.png")

    np.savez_compressed(
        run_dir / "replay_arrays.npz",
        frame_indices=frame_indices_array,
        predicted_first_actions=first_pred_array,
        ground_truth_first_actions=first_gt_array,
        predicted_chunks=pred_chunks,
        ground_truth_chunks=gt_chunks,
        chunk_mae=chunk_mae_array,
        forecast_offset_mae=offset_mae,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Replay finished. First-action MAE: %.6f", summary["metrics"]["first_action_mae"])
    logger.info("Artifacts written to %s", run_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
