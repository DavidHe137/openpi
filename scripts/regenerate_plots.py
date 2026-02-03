#!/usr/bin/env python3
"""Script to regenerate plots from existing experiment data."""

import pathlib

import tyro

from examples.libero.metrics import calculate_metrics
from examples.libero.metrics import generate_all_plots


def main(data_dir: str) -> None:
    """Regenerate all plots from an existing data directory.

    Args:
        data_dir: Path to the data directory (e.g., data/libero/sync)
    """
    output_path = pathlib.Path(data_dir)

    if not output_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")

    print(f"Regenerating plots for {data_dir}...")

    # Regenerate metrics CSV files
    calculate_metrics(output_path)

    # Generate all plots
    generate_all_plots(output_path)

    print(f"\nDone! Plots saved to {output_path / 'plots'}")


if __name__ == "__main__":
    tyro.cli(main)
