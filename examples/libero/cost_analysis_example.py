"""
Example demonstrating programmatic use of the cost analyzer.

This shows how to:
1. Load episode data
2. Configure and run cost analysis
3. Access cost information programmatically
"""

import pathlib
from openpi_client.schemas import ActionChunk, Timestamp
from examples.libero.cost_analyzer import CostAnalyzer, CostConfig


def basic_example():
    """Basic cost analysis example."""
    print("=" * 80)
    print("Basic Cost Analysis Example")
    print("=" * 80)

    # Configure cost parameters
    config = CostConfig(execution_cost_rate=1.0, pause_cost=10.0, pause_threshold=0.05)

    # Create analyzer
    analyzer = CostAnalyzer(config)

    # Load episode data (replace with your actual path)
    episode_dir = pathlib.Path("data/libero/sync_5/0/0_libero_10_0_success")

    if not episode_dir.exists():
        print(f"Episode directory not found: {episode_dir}")
        print("Please update the path to a valid episode directory.")
        return

    timestamps = Timestamp.from_csv(episode_dir / "timestamps.csv")
    action_chunks = ActionChunk.from_csv(episode_dir / "action_chunks.csv")

    print(
        f"\nLoaded {len(timestamps)} timestamps and {len(action_chunks)} action chunks"
    )

    # Run analysis
    chunk_costs, total_cost = analyzer.analyze(action_chunks, timestamps)

    # Print results
    print(
        f"\n{'Chunk':<8} {'Duration':<12} {'Exec Cost':<12} {'Pause Cost':<12} {'Total':<12}"
    )
    print("-" * 60)

    for chunk in chunk_costs[:10]:  # Show first 10 chunks
        print(
            f"{chunk.chunk_index:<8} "
            f"{chunk.duration:<12.4f} "
            f"{chunk.execution_cost:<12.2f} "
            f"{chunk.pause_cost:<12.2f} "
            f"{chunk.total_cost:<12.2f}"
        )

    if len(chunk_costs) > 10:
        print(f"... ({len(chunk_costs) - 10} more chunks)")

    print(f"\nTotal cost: {total_cost:.2f}")

    # Generate visualizations
    output_dir = pathlib.Path("cost_analysis_output")
    output_dir.mkdir(exist_ok=True)

    analyzer.plot_costs(output_dir / "example_cost_plot.png")
    analyzer.save_summary(output_dir / "example_summary.txt")

    print(f"\nVisualizations saved to: {output_dir}")


def custom_analysis_example():
    """Example showing custom analysis of cost data."""
    print("\n" + "=" * 80)
    print("Custom Analysis Example")
    print("=" * 80)

    config = CostConfig(
        execution_cost_rate=2.0,  # Double the execution cost
        pause_cost=5.0,  # Lower pause penalty
        pause_threshold=0.1,  # Higher threshold
    )

    analyzer = CostAnalyzer(config)

    # Load data
    episode_dir = pathlib.Path("data/libero/sync_5/0/0_libero_10_0_success")

    if not episode_dir.exists():
        print(f"Episode directory not found: {episode_dir}")
        return

    timestamps = Timestamp.from_csv(episode_dir / "timestamps.csv")
    action_chunks = ActionChunk.from_csv(episode_dir / "action_chunks.csv")

    # Analyze
    chunk_costs, total_cost = analyzer.analyze(action_chunks, timestamps)

    # Custom analysis: find most expensive chunks
    sorted_chunks = sorted(chunk_costs, key=lambda c: c.total_cost, reverse=True)

    print("\nTop 5 Most Expensive Chunks:")
    print(f"{'Rank':<6} {'Chunk':<8} {'Total Cost':<12} {'Reason'}")
    print("-" * 50)

    for rank, chunk in enumerate(sorted_chunks[:5], 1):
        reason = []
        if chunk.pause_cost > 0:
            reason.append("pause penalty")
        if chunk.duration > 0.5:
            reason.append("long duration")
        if chunk.replanned_early:
            reason.append("replanned early")

        reason_str = ", ".join(reason) if reason else "normal execution"
        print(
            f"{rank:<6} {chunk.chunk_index:<8} {chunk.total_cost:<12.2f} {reason_str}"
        )

    # Count chunks with pauses
    chunks_with_pauses = sum(1 for c in chunk_costs if c.pause_cost > 0)
    print(f"\nChunks with pause penalties: {chunks_with_pauses}/{len(chunk_costs)}")

    # Average execution time
    avg_duration = sum(c.duration for c in chunk_costs) / len(chunk_costs)
    print(f"Average chunk duration: {avg_duration:.4f}s")

    # Replanning statistics
    replanned_count = sum(1 for c in chunk_costs if c.replanned_early)
    print(f"Chunks replanned early: {replanned_count}/{len(chunk_costs)}")


def comparison_example():
    """Example comparing different cost configurations."""
    print("\n" + "=" * 80)
    print("Configuration Comparison Example")
    print("=" * 80)

    # Load data once
    episode_dir = pathlib.Path("data/libero/sync_5/0/0_libero_10_0_success")

    if not episode_dir.exists():
        print(f"Episode directory not found: {episode_dir}")
        return

    timestamps = Timestamp.from_csv(episode_dir / "timestamps.csv")
    action_chunks = ActionChunk.from_csv(episode_dir / "action_chunks.csv")

    # Define configurations to compare
    configs = [
        ("Balanced", CostConfig(execution_cost_rate=1.0, pause_cost=10.0)),
        ("Execution-heavy", CostConfig(execution_cost_rate=5.0, pause_cost=1.0)),
        ("Pause-heavy", CostConfig(execution_cost_rate=0.5, pause_cost=50.0)),
    ]

    print(f"\nComparing {len(configs)} cost configurations:\n")
    print(
        f"{'Configuration':<20} {'Total Cost':<15} {'Exec Cost':<15} {'Pause Cost':<15}"
    )
    print("-" * 70)

    for name, config in configs:
        analyzer = CostAnalyzer(config)
        chunk_costs, total_cost = analyzer.analyze(action_chunks, timestamps)

        total_exec = sum(c.execution_cost for c in chunk_costs)
        total_pause = sum(c.pause_cost for c in chunk_costs)

        print(
            f"{name:<20} {total_cost:<15.2f} {total_exec:<15.2f} {total_pause:<15.2f}"
        )

    print(
        "\nThis comparison helps understand how different cost models affect optimization goals."
    )


if __name__ == "__main__":
    # Run examples
    basic_example()
    custom_analysis_example()
    comparison_example()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
