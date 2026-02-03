# ruff: noqa
import os
import time

import matplotlib.pyplot as plt
import numpy as np


def busy_wait_timing(target_interval: float, num_steps: int, use_affinity: bool) -> list[float]:
    """Test timing using busy waiting."""
    if use_affinity:
        # Pin to first CPU
        os.sched_setaffinity(0, {list(os.sched_getaffinity(0))[0]})

    intervals = []
    last_time = time.perf_counter()

    for i in range(num_steps):
        # Busy wait until target interval has passed
        while time.perf_counter() - last_time < target_interval:
            pass

        current_time = time.perf_counter()
        actual_interval = current_time - last_time
        intervals.append(actual_interval)
        last_time = current_time

    return intervals


def sleep_timing(target_interval: float, num_steps: int, use_affinity: bool) -> list[float]:
    """Test timing using time.sleep()."""
    if use_affinity:
        # Pin to first CPU
        os.sched_setaffinity(0, {list(os.sched_getaffinity(0))[0]})

    intervals = []
    last_time = time.perf_counter()

    for i in range(num_steps):
        # Sleep until target interval
        sleep_duration = target_interval - (time.perf_counter() - last_time)
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        current_time = time.perf_counter()
        actual_interval = current_time - last_time
        intervals.append(actual_interval)
        last_time = current_time

    return intervals


def analyze_intervals(intervals: list[float], target: float) -> tuple[float, float, float]:
    """Calculate statistics for intervals."""
    intervals_array = np.array(intervals)
    mean = np.mean(intervals_array)
    std = np.std(intervals_array)
    max_error = np.max(np.abs(intervals_array - target))
    return mean, std, max_error


def run_experiments():
    """Run all timing experiments."""
    target_interval = 0.01  # 10ms target interval (100 Hz)
    num_steps = 1000

    print("Running timing experiments...")
    print(f"Target interval: {target_interval * 1000:.2f}ms")
    print(f"Number of steps: {num_steps}")
    print(f"Available CPUs: {os.sched_getaffinity(0)}\n")

    # Run all combinations
    results = {}

    print("1. Sleep timing WITHOUT affinity...")
    results["sleep_no_affinity"] = sleep_timing(target_interval, num_steps, use_affinity=False)
    mean, std, max_err = analyze_intervals(results["sleep_no_affinity"], target_interval)
    print(f"   Mean: {mean * 1000:.4f}ms, Std: {std * 1000:.4f}ms, Max error: {max_err * 1000:.4f}ms")

    print("2. Sleep timing WITH affinity...")
    results["sleep_with_affinity"] = sleep_timing(target_interval, num_steps, use_affinity=True)
    mean, std, max_err = analyze_intervals(results["sleep_with_affinity"], target_interval)
    print(f"   Mean: {mean * 1000:.4f}ms, Std: {std * 1000:.4f}ms, Max error: {max_err * 1000:.4f}ms")

    # Reset affinity for next tests
    os.sched_setaffinity(0, os.sched_getaffinity(0))

    print("3. Busy wait timing WITHOUT affinity...")
    results["busy_no_affinity"] = busy_wait_timing(target_interval, num_steps, use_affinity=False)
    mean, std, max_err = analyze_intervals(results["busy_no_affinity"], target_interval)
    print(f"   Mean: {mean * 1000:.4f}ms, Std: {std * 1000:.4f}ms, Max error: {max_err * 1000:.4f}ms")

    print("4. Busy wait timing WITH affinity...")
    results["busy_with_affinity"] = busy_wait_timing(target_interval, num_steps, use_affinity=True)
    mean, std, max_err = analyze_intervals(results["busy_with_affinity"], target_interval)
    print(f"   Mean: {mean * 1000:.4f}ms, Std: {std * 1000:.4f}ms, Max error: {max_err * 1000:.4f}ms")

    return results, target_interval


def plot_results(results: dict, target_interval: float):
    """Create plots comparing all timing methods."""

    # Convert to milliseconds for plotting
    target_ms = target_interval * 1000

    # Plot 1: Sleep vs Busy Wait (without affinity)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Sleep without affinity
    ax = axes[0, 0]
    intervals_ms = np.array(results["sleep_no_affinity"]) * 1000
    ax.plot(intervals_ms, alpha=0.7, linewidth=0.5)
    ax.axhline(y=target_ms, color="r", linestyle="--", label=f"Target ({target_ms}ms)")
    ax.set_title("Sleep Timing (No Affinity)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Interval (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: Sleep with affinity
    ax = axes[0, 1]
    intervals_ms = np.array(results["sleep_with_affinity"]) * 1000
    ax.plot(intervals_ms, alpha=0.7, linewidth=0.5)
    ax.axhline(y=target_ms, color="r", linestyle="--", label=f"Target ({target_ms}ms)")
    ax.set_title("Sleep Timing (With Affinity)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Interval (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: Busy wait without affinity
    ax = axes[1, 0]
    intervals_ms = np.array(results["busy_no_affinity"]) * 1000
    ax.plot(intervals_ms, alpha=0.7, linewidth=0.5)
    ax.axhline(y=target_ms, color="r", linestyle="--", label=f"Target ({target_ms}ms)")
    ax.set_title("Busy Wait Timing (No Affinity)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Interval (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Busy wait with affinity
    ax = axes[1, 1]
    intervals_ms = np.array(results["busy_with_affinity"]) * 1000
    ax.plot(intervals_ms, alpha=0.7, linewidth=0.5)
    ax.axhline(y=target_ms, color="r", linestyle="--", label=f"Target ({target_ms}ms)")
    ax.set_title("Busy Wait Timing (With Affinity)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Interval (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("timing_comparison_all.png", dpi=150)
    print("\nSaved: timing_comparison_all.png")

    # Plot 2: Histograms comparing distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (key, title) in enumerate(
        [
            ("sleep_no_affinity", "Sleep (No Affinity)"),
            ("sleep_with_affinity", "Sleep (With Affinity)"),
            ("busy_no_affinity", "Busy Wait (No Affinity)"),
            ("busy_with_affinity", "Busy Wait (With Affinity)"),
        ]
    ):
        ax = axes[idx // 2, idx % 2]
        intervals_ms = np.array(results[key]) * 1000
        ax.hist(intervals_ms, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(x=target_ms, color="r", linestyle="--", label=f"Target ({target_ms}ms)")
        ax.axvline(x=np.mean(intervals_ms), color="g", linestyle="--", label=f"Mean ({np.mean(intervals_ms):.4f}ms)")
        ax.set_title(title)
        ax.set_xlabel("Interval (ms)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("timing_histograms.png", dpi=150)
    print("Saved: timing_histograms.png")

    # Plot 3: Direct comparison of errors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sleep comparison
    ax = axes[0]
    errors_no_aff = (np.array(results["sleep_no_affinity"]) - target_interval) * 1000
    errors_with_aff = (np.array(results["sleep_with_affinity"]) - target_interval) * 1000
    ax.plot(errors_no_aff, alpha=0.6, label="No Affinity", linewidth=0.5)
    ax.plot(errors_with_aff, alpha=0.6, label="With Affinity", linewidth=0.5)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_title("Sleep Timing: Error Comparison")
    ax.set_xlabel("Step")
    ax.set_ylabel("Error from Target (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Busy wait comparison
    ax = axes[1]
    errors_no_aff = (np.array(results["busy_no_affinity"]) - target_interval) * 1000
    errors_with_aff = (np.array(results["busy_with_affinity"]) - target_interval) * 1000
    ax.plot(errors_no_aff, alpha=0.6, label="No Affinity", linewidth=0.5)
    ax.plot(errors_with_aff, alpha=0.6, label="With Affinity", linewidth=0.5)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_title("Busy Wait Timing: Error Comparison")
    ax.set_xlabel("Step")
    ax.set_ylabel("Error from Target (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("timing_error_comparison.png", dpi=150)
    print("Saved: timing_error_comparison.png")


if __name__ == "__main__":
    results, target_interval = run_experiments()
    plot_results(results, target_interval)
    print("\nExperiments complete!")
