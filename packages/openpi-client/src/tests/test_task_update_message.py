from dataclasses import asdict

from openpi_client import msgpack_numpy
from openpi_client.messages import TaskUpdate


def test_task_update_progress_round_trip() -> None:
    payload = TaskUpdate(
        task_suite_name="libero_10",
        task_id=3,
        episode_idx=1,
        current_step=12,
        max_episode_steps=600,
        phase="progress",
        task_language="pick up the bowl",
        total_episodes=2,
    )

    unpacked = msgpack_numpy.unpackb(msgpack_numpy.packb(asdict(payload)))
    assert TaskUpdate(**unpacked) == payload


def test_task_update_result_round_trip() -> None:
    payload = TaskUpdate(
        task_suite_name="libero_10",
        task_id=3,
        episode_idx=1,
        current_step=244,
        max_episode_steps=600,
        phase="result",
        task_language="pick up the bowl",
        total_episodes=2,
        success=True,
        duration_s=12.3,
        steps_taken=244,
        max_duration_s=30.0,
    )

    unpacked = msgpack_numpy.unpackb(msgpack_numpy.packb(asdict(payload)))
    assert TaskUpdate(**unpacked) == payload
