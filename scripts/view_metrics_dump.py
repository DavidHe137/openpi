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

from openpi_client.schemas import ServerMetadata
import tyro

from openpi.serving.metrics.dash_app import create_dash_app
from openpi.serving.metrics.store import MetricsStore


@dataclass
class Args:
    path: str  # experiment output dir or path to server_metrics_history.json
    port: int = 8050
    host: str = "127.0.0.1"


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

    mock_store = MetricsStore.from_dump(hist)
    app = create_dash_app(metadata, mock_store)
    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard: {url}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main(tyro.cli(Args))
