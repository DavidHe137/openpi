"""
Run a short multi-robot workload for nsys profiling, then shut down the server.

Usage:
  source scripts/libero_client.sh
  nsys profile --trace=cuda,nvtx,osrt --process-scope=process-tree -o profile \\
      uv run python examples/libero/profile_nsys.py \\
          --host 0.0.0.0 --port 8080 --num_robots 4 --max_steps 75

Or start the server separately, wait for "GPU worker ready", then:
  nsys profile --attach $(pgrep -P $(pgrep -f serve.py) | head -1) -o profile
  uv run python examples/libero/profile_nsys.py --host 0.0.0.0 --port 8080
"""

import sys
import os

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys.path[0] != _repo_root:
    sys.path.insert(0, _repo_root)

import dataclasses  # noqa: E402
import logging  # noqa: E402
import multiprocessing  # noqa: E402

import requests  # noqa: E402
import tyro  # noqa: E402

from examples.libero.main_multi_robot_runtime import Args, main  # noqa: E402

logger = logging.getLogger(__name__)

_PROFILE_DEFAULTS = dict(
    num_robots=4,
    num_trials_per_task=1,  # one episode per task
    max_steps=75,  # ~3-4 s at 20 Hz — enough for steady-state inference
    task_suite_name="libero_10",
    output_dir="/tmp/profile_nsys_out",
    overwrite=True,
    # progress_type=None,  # no rich progress bar during profiling
)


def _shutdown_server(host: str, port: int) -> None:
    url = f"http://{host}:{port}/shutdown"
    try:
        requests.post(url, timeout=5.0)
        logger.info("Sent shutdown request to server at %s", url)
    except Exception as e:
        logger.warning("Could not reach server for shutdown: %s", e)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # Parse CLI, injecting profiling-friendly defaults.
    # Any explicit CLI flag overrides the defaults below.
    base_args = Args(
        **{
            **dataclasses.asdict(Args()),
            **_PROFILE_DEFAULTS,
        }
    )
    args = tyro.cli(Args, default=base_args)

    try:
        main(args)
    finally:
        _shutdown_server(args.host, args.port)
