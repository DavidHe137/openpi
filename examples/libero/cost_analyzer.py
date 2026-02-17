"""
Cost analysis module for robot action scheduling.

This module tracks and analyzes the cost of executing action chunks,
including linear execution costs and pause penalties.
"""

import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
from openpi_client.schemas import ActionChunk, Timestamp

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """Configuration for cost calculation."""

    # Linear cost per timestep during execution (cost = c * time)
    execution_cost_rate: float = 1.0

    # One-time cost incurred when there's a pause between action chunks
    pause_cost: float = 10.0

    # Threshold (in seconds) to consider a gap as a pause
    pause_threshold: float = 0.05


@dataclass
class ChunkCostInfo:
    """Cost information for a single action chunk."""

    chunk_index: int
    start_step: int
    execution_horizon: int
    actual_steps_executed: int
    replanned_early: bool

    # Timing information
    start_time: float
    end_time: float
    duration: float

    # Cost breakdown
    execution_cost: float
    pause_cost: float
    total_cost: float

    # Gap to next chunk (if any)
    gap_to_next: float = 0.0


class CostAnalyzer:
    """Analyzes and tracks costs for action chunk execution."""

    def __init__(self, config: CostConfig = None):
        """
        Initialize the cost analyzer.

        Args:
            config: Cost configuration. If None, uses default values.
        """
        self.config = config or CostConfig()
        self._chunk_costs: List[ChunkCostInfo] = []
        self._total_cost: float = 0.0

    def analyze(
        self, action_chunks: List[ActionChunk], timestamps: List[Timestamp]
    ) -> Tuple[List[ChunkCostInfo], float]:
        """
        Analyze costs for all action chunks based on execution timestamps.

        Args:
            action_chunks: List of action chunks that were executed
            timestamps: List of timestamps from execution

        Returns:
            Tuple of (list of chunk cost info, total cost)
        """
        if not action_chunks or not timestamps:
            logger.warning("No action chunks or timestamps to analyze")
            return [], 0.0

        self._chunk_costs = []
        self._total_cost = 0.0

        # Group timestamps by action chunk index
        chunk_timestamps: Dict[int, List[Timestamp]] = {}

        # Check if timestamps have action_chunk_index information
        has_chunk_index = any(ts.action_chunk_index is not None for ts in timestamps)

        if has_chunk_index:
            # Use the action_chunk_index from timestamps
            for ts in timestamps:
                if ts.action_chunk_index is not None:
                    if ts.action_chunk_index not in chunk_timestamps:
                        chunk_timestamps[ts.action_chunk_index] = []
                    chunk_timestamps[ts.action_chunk_index].append(ts)
        else:
            # Infer chunk assignment from start_step and execution_horizon
            logger.info(
                "Timestamps don't have action_chunk_index, inferring from env_step"
            )
            for ts in timestamps:
                # Find which chunk this timestamp belongs to
                for chunk_idx, chunk in enumerate(action_chunks):
                    end_step = chunk.start_step + chunk.execution_horizon
                    if chunk.start_step <= ts.env_step < end_step:
                        if chunk_idx not in chunk_timestamps:
                            chunk_timestamps[chunk_idx] = []
                        chunk_timestamps[chunk_idx].append(ts)
                        break

        # Analyze each chunk
        for chunk_idx, chunk in enumerate(action_chunks):
            chunk_ts = chunk_timestamps.get(chunk_idx, [])

            if not chunk_ts:
                logger.warning(f"No timestamps found for chunk {chunk_idx}")
                continue

            # Sort timestamps by time
            chunk_ts.sort(key=lambda x: x.timestamp)

            # Get timing info
            start_time = chunk_ts[0].timestamp
            end_time = chunk_ts[-1].timestamp
            duration = end_time - start_time

            # Calculate actual steps executed
            actual_steps = len(chunk_ts)
            replanned_early = actual_steps < chunk.execution_horizon

            # Calculate execution cost (linear with time)
            execution_cost = self.config.execution_cost_rate * duration

            # Check for pause before this chunk
            pause_cost = 0.0
            gap_to_next = 0.0

            if chunk_idx > 0 and self._chunk_costs:
                prev_chunk = self._chunk_costs[-1]
                gap = start_time - prev_chunk.end_time
                prev_chunk.gap_to_next = gap

                # If gap exceeds threshold, incur pause cost
                if gap > self.config.pause_threshold:
                    pause_cost = self.config.pause_cost
                    logger.debug(
                        f"Pause detected before chunk {chunk_idx}: "
                        f"gap={gap:.4f}s, cost={pause_cost}"
                    )

            total_cost = execution_cost + pause_cost

            chunk_cost_info = ChunkCostInfo(
                chunk_index=chunk_idx,
                start_step=chunk.start_step,
                execution_horizon=chunk.execution_horizon,
                actual_steps_executed=actual_steps,
                replanned_early=replanned_early,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                execution_cost=execution_cost,
                pause_cost=pause_cost,
                total_cost=total_cost,
                gap_to_next=gap_to_next,
            )

            self._chunk_costs.append(chunk_cost_info)
            self._total_cost += total_cost

        logger.info(
            f"Cost analysis complete: {len(self._chunk_costs)} chunks, "
            f"total cost={self._total_cost:.2f}"
        )

        return self._chunk_costs, self._total_cost

    def plot_costs(self, out_path: pathlib.Path, show_breakdown: bool = True) -> None:
        """
        Generate cost visualization plots.

        Args:
            out_path: Path to save the plot
            show_breakdown: Whether to show cost breakdown by type
        """
        if not self._chunk_costs:
            logger.warning("No cost data to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 13))

        chunk_indices = [c.chunk_index for c in self._chunk_costs]

        # Plot 1: Total cost per chunk
        ax1 = axes[0]
        total_costs = [c.total_cost for c in self._chunk_costs]
        cumulative_costs = np.cumsum(total_costs)

        ax1.bar(chunk_indices, total_costs, alpha=0.7, label="Per-chunk cost")
        ax1.plot(
            chunk_indices,
            cumulative_costs,
            "r-",
            linewidth=2,
            label="Cumulative cost",
            marker="o",
        )
        ax1.set_xlabel("Chunk Index")
        ax1.set_ylabel("Cost")
        ax1.set_title("Total Cost per Action Chunk")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cost breakdown (execution vs pause)
        ax2 = axes[1]
        if show_breakdown:
            execution_costs = [c.execution_cost for c in self._chunk_costs]
            pause_costs = [c.pause_cost for c in self._chunk_costs]

            x = np.arange(len(chunk_indices))
            width = 0.35

            ax2.bar(x, execution_costs, width, label="Execution cost", alpha=0.8)
            ax2.bar(
                x,
                pause_costs,
                width,
                bottom=execution_costs,
                label="Pause cost",
                alpha=0.8,
            )
            ax2.set_xlabel("Chunk Index")
            ax2.set_ylabel("Cost")
            ax2.set_title("Cost Breakdown: Execution vs Pause")
            ax2.set_xticks(x)
            ax2.set_xticklabels(chunk_indices)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Timing information
        ax3 = axes[2]
        durations = [c.duration for c in self._chunk_costs]
        gaps = [c.gap_to_next for c in self._chunk_costs[:-1]] + [
            0
        ]  # Last chunk has no gap

        ax3.bar(chunk_indices, durations, alpha=0.7, label="Execution duration")
        ax3.scatter(
            chunk_indices,
            gaps,
            color="red",
            s=100,
            marker="x",
            label="Gap to next chunk",
            zorder=5,
        )

        # Highlight pauses
        pause_indices = [c.chunk_index for c in self._chunk_costs if c.pause_cost > 0]
        if pause_indices:
            ax3.axhline(
                y=self.config.pause_threshold,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Pause threshold ({self.config.pause_threshold}s)",
            )
            for idx in pause_indices:
                ax3.axvspan(idx - 0.4, idx + 0.4, alpha=0.2, color="red")

        ax3.set_xlabel("Chunk Index")
        ax3.set_ylabel("Time (seconds)")
        ax3.set_title("Execution Timing and Gaps")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Cost over time (resets to 0 at each chunk start)
        ax4 = axes[3]

        time_points = []
        cost_values = []

        for c in self._chunk_costs:
            # Pause cost spike (if any)
            if c.pause_cost > 0:
                time_points.append(c.start_time - 0.001)
                cost_values.append(0)
                time_points.append(c.start_time - 0.001)
                cost_values.append(c.pause_cost)

            # Chunk execution: linear ramp from 0 to execution_cost
            time_points.append(c.start_time)
            cost_values.append(0)
            time_points.append(c.end_time)
            cost_values.append(c.execution_cost)

        # Normalize time
        if time_points:
            start_offset = self._chunk_costs[0].start_time
            time_points = [t - start_offset for t in time_points]

        ax4.plot(time_points, cost_values, "b-", linewidth=2)
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("Cost")
        ax4.set_title("Cost Over Time (Resets to 0 at Each Chunk Start)")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Cost visualization saved to {out_path}")

    def save_summary(self, out_path: pathlib.Path) -> None:
        """
        Save a text summary of the cost analysis.

        Args:
            out_path: Path to save the summary text file
        """
        if not self._chunk_costs:
            logger.warning("No cost data to save")
            return

        with open(out_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COST ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Configuration
            f.write("Configuration:\n")
            f.write(f"  Execution cost rate: {self.config.execution_cost_rate}\n")
            f.write(f"  Pause cost: {self.config.pause_cost}\n")
            f.write(f"  Pause threshold: {self.config.pause_threshold}s\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"  Total chunks: {len(self._chunk_costs)}\n")
            f.write(f"  Total cost: {self._total_cost:.2f}\n")

            total_execution_cost = sum(c.execution_cost for c in self._chunk_costs)
            total_pause_cost = sum(c.pause_cost for c in self._chunk_costs)
            f.write(f"  Total execution cost: {total_execution_cost:.2f}\n")
            f.write(f"  Total pause cost: {total_pause_cost:.2f}\n")

            num_pauses = sum(1 for c in self._chunk_costs if c.pause_cost > 0)
            f.write(f"  Number of pauses: {num_pauses}\n")

            num_replanned = sum(1 for c in self._chunk_costs if c.replanned_early)
            f.write(f"  Chunks replanned early: {num_replanned}\n\n")

            # Per-chunk details
            f.write("Per-Chunk Details:\n")
            f.write("-" * 80 + "\n")

            for c in self._chunk_costs:
                f.write(f"\nChunk {c.chunk_index}:\n")
                f.write(f"  Start step: {c.start_step}\n")
                f.write(f"  Execution horizon: {c.execution_horizon}\n")
                f.write(f"  Actual steps executed: {c.actual_steps_executed}\n")
                f.write(f"  Replanned early: {c.replanned_early}\n")
                f.write(f"  Duration: {c.duration:.4f}s\n")
                f.write(f"  Gap to next: {c.gap_to_next:.4f}s\n")
                f.write(f"  Execution cost: {c.execution_cost:.2f}\n")
                f.write(f"  Pause cost: {c.pause_cost:.2f}\n")
                f.write(f"  Total cost: {c.total_cost:.2f}\n")

        logger.info(f"Cost summary saved to {out_path}")
