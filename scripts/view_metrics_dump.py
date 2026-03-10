"""Standalone dashboard viewer for server metrics dumps.

Point it at an experiment output directory (or the JSON file directly) and get
the same Dash dashboard you'd see on the live server.

Usage:
    uv run python -m scripts.view_metrics_dump path/to/output_dir
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib

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

    hist_path = path / "server_metrics_history.json"
    meta_path = path / "server_metadata.json"

    hist = json.loads(hist_path.read_text())
    metadata = ServerMetadata.from_json(meta_path)

    mock_store = MetricsStore.from_dump(hist)
    app = create_dash_app(metadata, mock_store)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main(tyro.cli(Args))
