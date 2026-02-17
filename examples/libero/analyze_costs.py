"""
Standalone script to analyze costs from existing episode data.

This script can be used to retroactively analyze costs for episodes
that were run without cost analysis enabled.
"""

import argparse
import pathlib
import logging
import numpy as np
import pandas as pd

from openpi_client.schemas import ActionChunk, Timestamp
from examples.libero.cost_analyzer import CostAnalyzer, CostConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_episode_data(episode_dir: pathlib.Path):
    """Load action chunks and timestamps from an episode directory."""
    timestamps_file = episode_dir / "timestamps.csv"
    action_chunks_parquet = episode_dir / "action_chunks.parquet"
    action_chunks_csv = episode_dir / "action_chunks.csv"

    if not timestamps_file.exists():
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")

    # Prefer parquet as it contains full action data
    if action_chunks_parquet.exists():
        action_chunks_file = action_chunks_parquet
        use_parquet = True
    elif action_chunks_csv.exists():
        action_chunks_file = action_chunks_csv
        use_parquet = False
    else:
        raise FileNotFoundError(f"Action chunks file not found in {episode_dir}")

    logger.info(f"Loading data from {episode_dir}")

    # Load timestamps - handle both old and new formats
    df = pd.read_csv(timestamps_file)
    timestamps = []

    for _, row in df.iterrows():
        # Check if the new format has action_chunk_index column
        action_chunk_index = row.get("action_chunk_index", None)
        action_index = row.get("action_index", None)

        # Handle NaN values (pandas reads None as NaN)
        if pd.isna(action_chunk_index):
            action_chunk_index = None
        if pd.isna(action_index):
            action_index = None

        timestamps.append(
            Timestamp(
                timestamp=row["timestamp"],
                env_step=row["env_step"],
                action_chunk_index=action_chunk_index,
                action_index=action_index,
            )
        )

    # Load action chunks
    if use_parquet:
        action_chunks = ActionChunk.from_parquet(action_chunks_file)
    else:
        # For CSV, we need to handle the missing 'actions' field
        # Since we only need timing and horizon info for cost analysis, we can create dummy actions
        df_chunks = pd.read_csv(action_chunks_csv)
        action_chunks = []

        for _, row in df_chunks.iterrows():
            # Create a minimal ActionChunk with dummy actions array
            chunk = ActionChunk(
                start_step=int(row["start_step"]),
                actions=np.zeros((int(row["execution_horizon"]), 7)),  # Dummy actions
                execution_horizon=int(row["execution_horizon"]),
                request_timestamp=float(row["request_timestamp"]),
                response_timestamp=float(row["response_timestamp"]),
                noise=None,
            )
            action_chunks.append(chunk)

    logger.info(
        f"Loaded {len(timestamps)} timestamps and {len(action_chunks)} action chunks"
    )
    return timestamps, action_chunks


def analyze_episode(
    episode_dir: pathlib.Path,
    cost_config: CostConfig,
    output_prefix: str = "cost_analysis",
):
    """Analyze costs for a single episode."""
    timestamps, action_chunks = load_episode_data(episode_dir)

    analyzer = CostAnalyzer(cost_config)
    chunk_costs, total_cost = analyzer.analyze(action_chunks, timestamps)

    # Save outputs
    analyzer.plot_costs(episode_dir / f"{output_prefix}.png")
    analyzer.save_summary(episode_dir / f"{output_prefix}_summary.txt")

    logger.info(f"Analysis complete. Total cost: {total_cost:.2f}")
    return chunk_costs, total_cost


def analyze_batch(
    root_dir: pathlib.Path, cost_config: CostConfig, recursive: bool = True
):
    """Analyze costs for all episodes in a directory."""
    # Find all episode directories (those containing timestamps.csv)
    if recursive:
        episode_dirs = [p.parent for p in root_dir.rglob("timestamps.csv")]
    else:
        episode_dirs = [p.parent for p in root_dir.glob("*/timestamps.csv")]

    logger.info(f"Found {len(episode_dirs)} episodes to analyze")

    total_costs = []
    for episode_dir in episode_dirs:
        try:
            _, total_cost = analyze_episode(episode_dir, cost_config)
            total_costs.append(total_cost)
        except Exception as e:
            logger.error(f"Failed to analyze {episode_dir}: {e}")

    if total_costs:
        logger.info("\nBatch Analysis Summary:")
        logger.info(f"  Episodes analyzed: {len(total_costs)}")
        logger.info(f"  Average cost: {sum(total_costs) / len(total_costs):.2f}")
        logger.info(f"  Min cost: {min(total_costs):.2f}")
        logger.info(f"  Max cost: {max(total_costs):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scheduling costs from episode data"
    )
    parser.add_argument(
        "path",
        type=pathlib.Path,
        help="Path to episode directory or root directory containing episodes",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Analyze all episodes in the directory (recursively)",
    )
    parser.add_argument(
        "--execution-cost-rate",
        type=float,
        default=1.0,
        help="Linear cost rate per second during execution (default: 1.0)",
    )
    parser.add_argument(
        "--pause-cost",
        type=float,
        default=10.0,
        help="One-time cost for pauses between chunks (default: 10.0)",
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=0.05,
        help="Threshold in seconds to consider a gap as a pause (default: 0.05)",
    )

    args = parser.parse_args()

    # Create cost config
    cost_config = CostConfig(
        execution_cost_rate=args.execution_cost_rate,
        pause_cost=args.pause_cost,
        pause_threshold=args.pause_threshold,
    )

    logger.info("Cost Configuration:")
    logger.info(f"  Execution cost rate: {cost_config.execution_cost_rate}")
    logger.info(f"  Pause cost: {cost_config.pause_cost}")
    logger.info(f"  Pause threshold: {cost_config.pause_threshold}s")
    logger.info("")

    # Analyze
    if args.batch:
        analyze_batch(args.path, cost_config)
    else:
        analyze_episode(args.path, cost_config)


if __name__ == "__main__":
    main()
