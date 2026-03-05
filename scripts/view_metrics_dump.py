"""Standalone dashboard viewer for server metrics dumps.

Point it at an experiment output directory (or the JSON file directly) and get
the same Dash dashboard you'd see on the live server.

Usage:
    uv run python -m scripts.view_metrics_dump path/to/output_dir
    uv run python -m scripts.view_metrics_dump path/to/server_metrics_history.json
    uv run python -m scripts.view_metrics_dump path/to/output_dir --port 8050
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
import sys
from typing import Any

import numpy as np
from openpi_client.schemas import ServerMetadata
import tyro

from openpi.serving.metrics.dash_app import create_dash_app


@dataclass
class Args:
    path: str  # experiment output dir or path to server_metrics_history.json
    port: int = 8050
    host: str = "127.0.0.1"


class MockMetricsStore:
    """Duck-typed MetricsStore backed by a saved history JSON dump."""

    def __init__(self, hist: dict) -> None:
        self._hist = hist

    def _filter_batches(self, window_s: float | None) -> list[dict]:
        batches = self._hist["batches"]
        if not batches or window_s is None:
            return batches
        max_t = batches[-1]["t"]
        return [b for b in batches if b["t"] >= max_t - window_s]

    def history(self, window_s: float | None = None) -> dict[str, Any]:
        batches = self._filter_batches(window_s)
        return {
            "server_start_time": self._hist["server_start_time"],
            "batches": batches,
            "outbound_delays_ms": self._hist.get("outbound_delays_ms", {}),
            "scheduler_timings_ms": self._hist.get("scheduler_timings_ms", {}),
        }

    def snapshot(self, window_s: float | None = None) -> dict[str, Any]:
        batches = self._filter_batches(window_s)

        total_requests = sum(b["batch_size"] for b in batches)
        gpu_times = [b["gpu_time_ms"] for b in batches]
        avg_gpu_time_ms = float(np.mean(gpu_times)) if gpu_times else 0.0

        if len(batches) >= 2:
            wall_s = batches[-1]["inference_end_t"] - batches[0]["inference_start_t"]
            gpu_busy_pct = min(100.0, sum(gpu_times) / (wall_s * 1000) * 100) if wall_s > 0 else 0.0
        else:
            wall_s = 0.0
            gpu_busy_pct = 0.0

        uptime_s = batches[-1]["t"] if batches else 0.0

        # e2e latency ≈ inbound + queue + infer (from client send to inference complete)
        e2e_ms = [req["inbound_ms"] + req["queue_ms"] + req["infer_ms"] for b in batches for req in b["per_request"]]
        p50 = float(np.percentile(e2e_ms, 50)) if e2e_ms else 0.0
        p99 = float(np.percentile(e2e_ms, 99)) if e2e_ms else 0.0

        queue_delays = [req["queue_ms"] for b in batches for req in b["per_request"]]
        avg_queue_delay_ms = float(np.mean(queue_delays)) if queue_delays else 0.0
        requests_per_second = total_requests / wall_s if wall_s > 0 else 0.0

        outbound = self._hist.get("outbound_delays_ms", {})
        per_robot: dict[str, Any] = {
            rid: {
                "total_starvations": 0,  # not stored in history dumps
                "avg_network_delay_ms": float(np.mean(delays)) if delays else 0.0,
            }
            for rid, delays in outbound.items()
        }

        return {
            "uptime_s": uptime_s,
            "total_batches": len(batches),
            "total_requests": total_requests,
            "avg_gpu_time_ms": avg_gpu_time_ms,
            "gpu_busy_pct": round(gpu_busy_pct, 1),
            "p50_latency_ms": p50,
            "p99_latency_ms": p99,
            "avg_queue_delay_ms": avg_queue_delay_ms,
            "requests_per_second": requests_per_second,
            "per_robot": per_robot,
        }

    def reset(self) -> None:
        pass  # no-op for the viewer


def main(args: Args) -> None:
    path = pathlib.Path(args.path)

    if path.is_dir():
        hist_path = path / "server_metrics_history.json"
        meta_path = path / "server_metadata.json"
    else:
        hist_path = path
        meta_path = path.parent / "server_metadata.json"

    if not hist_path.exists():
        print(f"Error: metrics history not found: {hist_path}", file=sys.stderr)
        sys.exit(1)

    hist = json.loads(hist_path.read_text())

    if meta_path.exists():
        metadata = ServerMetadata.from_json(meta_path)
    else:
        metadata = ServerMetadata(
            config_name="unknown",
            checkpoint_dir="unknown",
            action_horizon=0,
            action_dim=0,
            num_steps=0,
            max_batch_size=0,
            env="unknown",
            scheduling_algorithm="unknown",
        )

    mock_store = MockMetricsStore(hist)
    app = create_dash_app(metadata, mock_store)
    url = f"http://{args.host}:{args.port}/dashboard/"
    print(f"Dashboard: {url}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main(tyro.cli(Args))
