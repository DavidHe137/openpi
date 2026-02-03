# ruff: noqa
#!/usr/bin/env python3
"""Offline benchmark for policy inference latency and throughput.

Benchmarks policy.infer_batch() directly without websocket overhead.
Tests multiple batch sizes and computes detailed latency/throughput metrics.

Example usage:
    # Single batch size
    uv run scripts/benchmark_offline.py --model LIBERO_PI0 --batch-sizes 1 --num-iterations 10

    # Multiple batch sizes
    uv run scripts/benchmark_offline.py --model LIBERO_PI0 --batch-sizes 1,2,4,8

    # All three models
    for model in LIBERO_PI0 LIBERO_PYTORCH LIBERO_REALTIME; do
        uv run scripts/benchmark_offline.py --model $model --batch-sizes 1,2,4,8
    done

    # With JSON output
    uv run scripts/benchmark_offline.py --model LIBERO_PI0 --save-result --save-result-dir benchmarks/offline
"""

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import gc
import json
import os
import subprocess
import time
from typing import Any

import numpy as np

# Conditional torch import for CUDA synchronization
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from openpi_client.messages import InferRequest
from openpi_client.messages import InferType

from openpi.policies import policy_config as _policy_config
from openpi.policies.libero_policy import make_libero_example
from openpi.policies.policy import EnvMode
from openpi.training import config as _config

# Model name to EnvMode mapping
MODEL_MAP = {
    "LIBERO_PI0": EnvMode.LIBERO_PI0,
    "LIBERO_PYTORCH": EnvMode.LIBERO_PYTORCH,
    "LIBERO_REALTIME": EnvMode.LIBERO_REALTIME,
}

# Default checkpoints for each environment
DEFAULT_CHECKPOINT = {
    EnvMode.LIBERO_PI0: {
        "config": "pi0_libero",
        "dir": "gs://openpi-assets/checkpoints/pi0_libero",
    },
    EnvMode.LIBERO_PYTORCH: {
        "config": "pi0_libero",
        "dir": "/coc/flash8/rbansal66/openpi_rollout/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_openpi",
    },
    EnvMode.LIBERO_REALTIME: {
        "config": "pi0_libero",
        "dir": "/coc/flash8/rbansal66/openpi_rollout/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_dexmal_mokapots",
    },
}


@dataclass
class BatchMetrics:
    """Metrics for a single batch size."""

    batch_size: int
    num_iterations: int
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    percentiles_ms: dict[str, float]
    throughput_samples_per_sec: float
    latency_per_sample_ms: float


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        gpu_info = result.stdout.strip().split(", ")
        return {
            "gpu_available": True,
            "gpu_name": gpu_info[0],
            "driver_version": gpu_info[1],
            "memory_total": gpu_info[2],
        }
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return {"gpu_available": False}


def create_policy(model_name: str, batch_size: int, num_steps: int):
    """Create a policy for the given model name.

    Args:
        model_name: One of LIBERO_PI0, LIBERO_PYTORCH, LIBERO_REALTIME
        batch_size: Batch size for inference
        num_steps: Number of diffusion sampling steps

    Returns:
        Policy instance
    """
    env_mode = MODEL_MAP[model_name]
    if checkpoint := DEFAULT_CHECKPOINT.get(env_mode):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint["config"]),
            checkpoint["dir"],
            default_prompt=None,
            sample_kwargs={"num_steps": num_steps},
            use_triton_optimized=(env_mode == EnvMode.LIBERO_REALTIME),
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported environment mode: {env_mode}")


def create_batch_requests(batch_size: int) -> list[InferRequest]:
    """Create a batch of InferRequest objects with random observations.

    Args:
        batch_size: Number of requests to create

    Returns:
        List of InferRequest objects
    """
    return [
        InferRequest(observation=make_libero_example(), infer_type=InferType.SYNC, params=None)
        for _ in range(batch_size)
    ]


def warmup_policy(policy, batch_size: int, num_warmup: int) -> None:
    """Warm up policy to trigger JIT compilation.

    The first call to infer_batch triggers JIT compilation for JAX/PyTorch/Triton.
    Additional warmup iterations ensure stable timing.

    Args:
        policy: Policy instance to warm up
        batch_size: Batch size for warmup
        num_warmup: Number of warmup iterations
    """
    print(f"Running {num_warmup} warmup iterations (batch_size={batch_size})...")

    for i in range(num_warmup):
        requests = create_batch_requests(batch_size)
        _ = policy.infer_batch(requests)

        if i == 0:
            print("  First warmup complete (JIT compilation triggered)")

    # Synchronize GPU if PyTorch
    if hasattr(policy, "_is_pytorch_model") and policy._is_pytorch_model and TORCH_AVAILABLE:
        torch.cuda.synchronize()

    print("Warmup complete.")


def benchmark_batch_size(policy, batch_size: int, num_iterations: int) -> list[float]:
    """Benchmark inference for a specific batch size.

    Args:
        policy: Policy instance to benchmark
        batch_size: Batch size to test
        num_iterations: Number of benchmark iterations

    Returns:
        List of latencies in milliseconds
    """
    latencies = []
    is_pytorch = hasattr(policy, "_is_pytorch_model") and policy._is_pytorch_model  # noqa: SLF001

    for _ in range(num_iterations):
        requests = create_batch_requests(batch_size)

        # CRITICAL: Synchronize GPU before timing for accuracy
        if is_pytorch and TORCH_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = policy.infer_batch(requests)

        # CRITICAL: Synchronize GPU after inference
        if is_pytorch and TORCH_AVAILABLE:
            torch.cuda.synchronize()

        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    return latencies


def calculate_metrics(batch_size: int, latencies_ms: list[float], percentiles: list[float]) -> BatchMetrics:
    """Calculate benchmark metrics from latency measurements.

    Args:
        batch_size: Batch size for this benchmark
        latencies_ms: List of latency measurements in milliseconds
        percentiles: List of percentiles to compute (e.g., [50, 95, 99])

    Returns:
        BatchMetrics with computed statistics
    """
    mean_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    median_ms = float(np.median(latencies_ms))

    # Compute percentiles
    percentiles_dict = {f"p{int(p) if int(p) == p else p}": float(np.percentile(latencies_ms, p)) for p in percentiles}

    # Throughput: samples per second
    mean_latency_s = mean_ms / 1000.0
    throughput = batch_size / mean_latency_s

    # Latency per sample
    latency_per_sample = mean_ms / batch_size

    return BatchMetrics(
        batch_size=batch_size,
        num_iterations=len(latencies_ms),
        mean_latency_ms=mean_ms,
        std_latency_ms=std_ms,
        min_latency_ms=min_ms,
        max_latency_ms=max_ms,
        median_latency_ms=median_ms,
        percentiles_ms=percentiles_dict,
        throughput_samples_per_sec=throughput,
        latency_per_sample_ms=latency_per_sample,
    )


def print_gpu_info(gpu_info: dict[str, Any]) -> None:
    """Print GPU information."""
    if gpu_info.get("gpu_available"):
        print(f"GPU:               {gpu_info['gpu_name']}")
        print(f"Driver:            {gpu_info['driver_version']}")
        print(f"Memory:            {gpu_info['memory_total']}")
    else:
        print("GPU:               Not available")


def print_batch_metrics(metrics: BatchMetrics) -> None:
    """Print metrics for a single batch size."""
    print(f"\nBatch Size: {metrics.batch_size}")
    print("-" * 80)
    print(f"  Iterations:                 {metrics.num_iterations}")
    print(f"  Mean Latency:               {metrics.mean_latency_ms:.2f} ± {metrics.std_latency_ms:.2f} ms")
    print(f"  Min/Max:                    {metrics.min_latency_ms:.2f} / {metrics.max_latency_ms:.2f} ms")
    print(f"  Median:                     {metrics.median_latency_ms:.2f} ms")

    # Print percentiles
    for p_name, p_value in sorted(metrics.percentiles_ms.items()):
        print(f"  {p_name.upper()}:                          {p_value:.2f} ms")

    print(f"  Throughput:                 {metrics.throughput_samples_per_sec:.2f} samples/sec")
    print(f"  Latency per sample:         {metrics.latency_per_sample_ms:.2f} ms")


def print_summary_table(all_results: dict[int, BatchMetrics]) -> None:
    """Print summary comparison table across all batch sizes."""
    print("\n" + "=" * 80)
    print("Summary Comparison")
    print("=" * 80)

    # Table header
    print(f"{'Batch':<8} {'Latency (ms)':<16} {'Throughput':<18} {'Latency/Sample':<18} {'Speedup':<10}")
    print("-" * 80)

    # Baseline is batch_size=1 for speedup calculation
    batch_sizes = sorted(all_results.keys())
    baseline_throughput = all_results[batch_sizes[0]].throughput_samples_per_sec if batch_sizes else 1.0

    for batch_size in batch_sizes:
        metrics = all_results[batch_size]
        speedup = metrics.throughput_samples_per_sec / baseline_throughput

        print(
            f"{batch_size:<8} "
            f"{metrics.mean_latency_ms:<16.2f} "
            f"{metrics.throughput_samples_per_sec:<18.2f} "
            f"{metrics.latency_per_sample_ms:<18.2f} "
            f"{speedup:<10.2f}x"
        )

    print("=" * 80)


def save_results_json(args: argparse.Namespace, gpu_info: dict, all_results: dict[int, BatchMetrics]) -> None:
    """Save benchmark results to JSON file.

    Args:
        args: Command-line arguments
        gpu_info: GPU information dictionary
        all_results: Dictionary of batch_size -> BatchMetrics
    """
    current_dt = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    filename = f"{args.save_result_dir}/benchmark-offline-{args.model}-{current_dt}.json"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Build JSON structure
    results_dict = {}
    for batch_size, metrics in all_results.items():
        metrics_dict = asdict(metrics)
        results_dict[str(batch_size)] = metrics_dict

    output = {
        "date": datetime.now(tz=UTC).isoformat(),
        "model": args.model,
        "num_steps": args.num_steps,
        "num_warmup": args.num_warmup,
        "num_iterations": args.num_iterations,
        "gpu_info": gpu_info,
        "results": results_dict,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {filename}")


def main(args: argparse.Namespace) -> None:
    """Main entry point for offline benchmark."""
    print("=" * 80)
    print(f"Offline Benchmark: {args.model}")
    print("=" * 80)
    print(f"Config: num_steps={args.num_steps}, num_warmup={args.num_warmup}, num_iterations={args.num_iterations}")

    # Set random seed
    np.random.seed(args.seed)

    # Get GPU info
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)

    # Parse batch sizes and percentiles
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    percentiles = [float(p) for p in args.metric_percentiles.split(",")]

    print(f"Batch sizes: {batch_sizes}")
    print(f"Percentiles: {percentiles}")

    all_results: dict[int, BatchMetrics] = {}

    # Benchmark each batch size
    for batch_size in batch_sizes:
        print("\n" + "=" * 80)
        print(f"Creating policy for batch_size={batch_size}")

        try:
            # Create policy (recreate for each batch size, especially important for Triton)
            policy = create_policy(args.model, batch_size, args.num_steps)

            # Warmup (triggers JIT compilation)
            warmup_policy(policy, batch_size, args.num_warmup)

            # Benchmark
            print(f"Running {args.num_iterations} benchmark iterations...")
            latencies = benchmark_batch_size(policy, batch_size, args.num_iterations)

            # Calculate metrics
            metrics = calculate_metrics(batch_size, latencies, percentiles)
            all_results[batch_size] = metrics

            # Print results
            print_batch_metrics(metrics)

            # Cleanup GPU memory and torch.compile caches
            del policy
            gc.collect()
            if TORCH_AVAILABLE:
                # Reset torch.compile/dynamo state to free compiled model caches
                torch._dynamo.reset()  # noqa: SLF001
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nERROR: Out of memory for batch_size={batch_size}")
                print("Skipping this batch size and continuing...")
                continue
            raise

    # Print summary table
    if all_results:
        print_summary_table(all_results)

        # Save JSON if requested
        if args.save_result:
            save_results_json(args, gpu_info, all_results)
    else:
        print("\nNo results collected (all batch sizes failed or were skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline benchmark for policy inference latency and throughput.")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Model to benchmark",
    )

    # Batch configuration
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated list of batch sizes to test (default: 1,2,4,8,16,32,64)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations to trigger JIT compilation (default: 10)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations per batch size (default: 100)",
    )

    # Model parameters
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of diffusion sampling steps (default: 10)",
    )

    # Metrics
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,95,99",
        help="Comma-separated list of percentiles to compute (default: 50,95,99)",
    )

    # Output options
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save benchmark results to a JSON file",
    )
    parser.add_argument(
        "--save-result-dir",
        type=str,
        default="benchmarks",
        help="Directory to save benchmark results (default: benchmarks)",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    args = parser.parse_args()
    main(args)
