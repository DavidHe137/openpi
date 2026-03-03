from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.libero.metrics import calculate_metrics
from examples.libero.metrics import load_planner_starvation_metrics
from openpi_client.schemas import RuntimeMetadata


def test_load_planner_starvation_metrics_counts_null_action_steps(tmp_path) -> None:
    runtime_metadata = RuntimeMetadata(
        task_suite_name="suite",
        num_steps_wait=0,
        num_trials_per_robot=1,
        max_steps=10,
        seed=0,
        resize_size=224,
        num_robots=1,
        control_hz=20,
        broker_type="sync",
        latency_ms=[],
    )
    runtime_metadata.to_json(tmp_path / "runtime_metadata.json")

    episode_dir = tmp_path / "0" / "0_suite_0_success"
    episode_dir.mkdir(parents=True)
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "robot_idx": 0,
                "success": True,
                "steps_taken": 4,
                "task_suite_name": "suite",
                "task_id": 0,
                "task_language": "task",
                "episode_idx": 0,
            },
            f,
        )
    np.save(episode_dir / "cost_history.npy", np.array([0.1, np.nan, 0.2, np.nan]))

    df = load_planner_starvation_metrics(tmp_path)

    assert len(df) == 1
    assert int(df.loc[0, "planner_starvation_steps"]) == 2
    assert float(df.loc[0, "planner_starvation_rate"]) == 0.5
    assert float(df.loc[0, "planner_starvation_seconds"]) == 0.1


def test_calculate_metrics_writes_planner_starvation_columns(tmp_path) -> None:
    runtime_metadata = RuntimeMetadata(
        task_suite_name="suite",
        num_steps_wait=0,
        num_trials_per_robot=1,
        max_steps=10,
        seed=0,
        resize_size=224,
        num_robots=1,
        control_hz=20,
        broker_type="sync",
        latency_ms=[],
    )
    runtime_metadata.to_json(tmp_path / "runtime_metadata.json")

    episode_dir = tmp_path / "0" / "0_suite_0_success"
    episode_dir.mkdir(parents=True)
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "robot_idx": 0,
                "success": True,
                "steps_taken": 4,
                "task_suite_name": "suite",
                "task_id": 0,
                "task_language": "task",
                "episode_idx": 0,
            },
            f,
        )
    np.save(episode_dir / "cost_history.npy", np.array([0.1, np.nan, 0.2, np.nan]))

    calculate_metrics(tmp_path)

    results_df = pd.read_csv(tmp_path / "results.csv")
    summary_df = pd.read_csv(tmp_path / "summary.csv")

    assert "planner_starvation_steps" in results_df.columns
    assert "planner_starvation_rate" in results_df.columns
    assert "planner_starvation_steps" in summary_df.columns
    assert float(summary_df.loc[0, "planner_starvation_steps"]) == 2.0
