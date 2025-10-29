# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/benchmarks/serve.py
r"""Benchmark online serving throughput for openpi policy server.

On the server side, run:
    uv run scripts/serve_policy.py --env=[ALOHA | DROID | LIBERO] --port=8000

On the client side, run:
    uv run scripts/benchmark.py \
        --host localhost \
        --port 8000 \
        --env aloha_sim \
        --num-requests 100 \
        --request-rate 10
"""

import argparse
import asyncio
from collections.abc import AsyncGenerator
import contextlib
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import enum
import json
import os
import time
from typing import Any
import warnings

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm.asyncio import tqdm

from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    DROID = "droid"
    LIBERO = "libero"


@dataclass
class RequestFuncOutput:
    """Output from a single request."""

    success: bool
    latency: float  # seconds
    start_time: float
    error: str | None = None


@dataclass
class BenchmarkMetrics:
    """Metrics for the benchmark."""

    completed: int
    total_requests: int
    request_throughput: float
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    percentiles_latency_ms: list[tuple[float, float]]


def get_observation(env: EnvMode) -> dict:
    """Generate a random observation for the given environment."""
    if env == EnvMode.ALOHA:
        return make_aloha_example()
    if env == EnvMode.DROID:
        return make_droid_example()
    if env == EnvMode.LIBERO:
        return make_libero_example()

    raise ValueError(f"Unknown environment: {env}")


async def get_request(
    num_requests: int,
    request_rate: float,
) -> AsyncGenerator[int, None]:
    """
    Asynchronously generates requests at a specified rate

    Args:
        num_requests:
            The number of requests to generate.
        request_rate:
            The rate at which requests are generated (requests/s).
    """
    assert num_requests > 0, "No requests provided."

    # Precompute delays among requests to minimize request send laggings
    delay_ts = []
    for _ in range(num_requests):
        if request_rate == float("inf"):
            delay_ts.append(0)
        else:
            theta = 1.0 / (request_rate)
            delay_ts.append(np.random.exponential(scale=theta))

    # Calculate the cumulative delay time from the first sent out requests.
    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if delay_ts[-1] != 0:
        # When ramp_up_strategy is not set, we assume the request rate is fixed
        # and all requests should be sent in target_total_delay_s, the following
        # logic would re-scale delay time to ensure the final delay_ts
        # align with target_total_delay_s.
        #
        # NOTE: If we simply accumulate the random delta values
        # from the gamma distribution, their sum would have 1-2% gap
        # from target_total_delay_s. The purpose of the following logic is to
        # close the gap for stabilizing the throughput data
        # from different random seeds.
        target_total_delay_s = num_requests / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    start_ts = time.time()
    for request_index in range(num_requests):
        if delay_ts[request_index] > 0:
            current_ts = time.time()
            sleep_interval_s = start_ts + delay_ts[request_index] - current_ts
            if sleep_interval_s > 0:
                await asyncio.sleep(sleep_interval_s)
        yield request_index


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    duration: float,
    num_requests: int,
    selected_percentiles: list[float],
) -> BenchmarkMetrics:
    """Calculate benchmark metrics from outputs."""
    completed = sum(1 for o in outputs if o.success)
    latencies = [o.latency for o in outputs if o.success]

    if completed == 0:
        warnings.warn("All requests failed.", stacklevel=2)
        return BenchmarkMetrics(
            completed=0,
            total_requests=num_requests,
            request_throughput=0.0,
            mean_latency_ms=0.0,
            median_latency_ms=0.0,
            std_latency_ms=0.0,
            percentiles_latency_ms=[(p, 0.0) for p in selected_percentiles],
        )

    return BenchmarkMetrics(
        completed=completed,
        total_requests=num_requests,
        request_throughput=completed / duration,
        mean_latency_ms=float(np.mean(latencies)) * 1000,
        median_latency_ms=float(np.median(latencies)) * 1000,
        std_latency_ms=float(np.std(latencies)) * 1000,
        percentiles_latency_ms=[(p, float(np.percentile(latencies, p)) * 1000) for p in selected_percentiles],
    )


async def send_request(
    policy: _websocket_client_policy.WebsocketClientPolicy, obs: dict, pbar: tqdm | None
) -> RequestFuncOutput:
    """Send a request to the server."""
    output = policy.infer(obs)
    if pbar is not None:
        pbar.update(1)
    latency = output["server_timing"]["infer_ms"]
    return RequestFuncOutput(
        success=True,
        latency=latency,
        start_time=time.time(),
        error=None,
    )


async def benchmark(
    host: str,
    port: int,
    env: EnvMode,
    num_requests: int,
    request_rate: float,
    max_concurrency: int | None,
    selected_percentiles: list[float],
    *,
    disable_tqdm: bool = False,
) -> dict[str, Any]:
    """Run the benchmark."""
    uri = f"ws://{host}:{port}"
    print(f"Connecting to server at {uri}...")

    # Test connection
    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        metadata = policy.get_server_metadata()
        print(f"Server metadata: {metadata}")

        # Warm-up request
        print("Running warm-up request...")
        test_obs = get_observation(env)
        _ = policy.infer(test_obs)
        print("Warm-up completed.")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to server: {e}") from e

    # Generate observations
    print(f"Generating {num_requests} observations...")
    observations = [get_observation(env) for _ in range(num_requests)]

    print("Starting benchmark...")
    print(f"Request rate: {request_rate if request_rate != float('inf') else 'unlimited'} req/s")
    print(f"Max concurrency: {max_concurrency if max_concurrency else 'unlimited'}")

    pbar = None if disable_tqdm else tqdm(total=num_requests)

    # Semaphore for max concurrency
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()

    async def limited_request(obs):
        async with semaphore:
            return await send_request(policy, obs, pbar)

    benchmark_start_time = time.perf_counter()
    tasks = []

    async for request_idx in get_request(num_requests, request_rate):
        task = asyncio.create_task(limited_request(observations[request_idx]))
        tasks.append(task)

    outputs = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # Calculate metrics
    metrics = calculate_metrics(outputs, benchmark_duration, num_requests, selected_percentiles)

    # Print results
    print("{s:{c}^{n}}".format(s=" Benchmark Results ", n=50, c="="))
    print("{:<40} {:<10}".format("Total requests:", metrics.total_requests))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.total_requests - metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{s:{c}^{n}}".format(s=" Latency Statistics ", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Median latency (ms):", metrics.median_latency_ms))
    print("{:<40} {:<10.2f}".format("Std latency (ms):", metrics.std_latency_ms))
    for p, value in metrics.percentiles_latency_ms:
        p_word = str(int(p)) if int(p) == p else str(p)
        print("{:<40} {:<10.2f}".format(f"P{p_word} latency (ms):", value))
    print("=" * 50)

    # Return results
    return {
        "date": datetime.now(tz=UTC).isoformat(),
        "host": host,
        "port": port,
        "env": env.value,
        "num_requests": num_requests,
        "request_rate": request_rate if request_rate != float("inf") else "inf",
        "max_concurrency": max_concurrency,
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "failed": metrics.total_requests - metrics.completed,
        "request_throughput": metrics.request_throughput,
        "mean_latency_ms": metrics.mean_latency_ms,
        "median_latency_ms": metrics.median_latency_ms,
        "std_latency_ms": metrics.std_latency_ms,
        **{f"p{int(p) if int(p) == p else p}_latency_ms": value for p, value in metrics.percentiles_latency_ms},
        "latencies": [o.latency * 1000 for o in outputs if o.success],
        "errors": [o.error for o in outputs if not o.success],
    }


def main(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    """Async main entry point."""
    print(args)
    np.random.seed(args.seed)

    result = await benchmark(
        host=args.host,
        port=args.port,
        env=EnvMode(args.env),
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        disable_tqdm=args.disable_tqdm,
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
    )

    # Save results if requested
    if args.save_result:
        current_dt = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")

        if args.result_filename:
            filename = args.result_filename
        else:
            rate_str = f"{args.request_rate}qps" if args.request_rate != float("inf") else "unlimited"
            concurrency_str = f"-concurrency{args.max_concurrency}" if args.max_concurrency else ""
            filename = f"benchmark-{args.env}-{rate_str}{concurrency_str}-{current_dt}.json"

        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            filename = os.path.join(args.result_dir, filename)

        # Remove detailed data if not requested
        if not args.save_detailed:
            result_to_save = {k: v for k, v in result.items() if k not in ["latencies", "errors"]}
        else:
            result_to_save = result

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_to_save, f, indent=2)

        print(f"\nResults saved to: {filename}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark openpi policy server throughput and latency.")

    # Server connection
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host address (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )

    # Environment
    parser.add_argument(
        "--env",
        type=str,
        default="libero",
        choices=["aloha", "droid", "libero"],
        help="Environment type (default: libero)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to send (default: 100)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate in requests/second. Use 'inf' for unlimited (default: inf)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests (default: unlimited)",
    )

    # Display options
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bar",
    )

    # Metrics
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="95,99",
        help="Comma-separated list of percentiles (default: 95,99)",
    )

    # Output options
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save benchmark results to a JSON file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Include detailed per-request data (latencies, errors) in saved results",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save results (default: current directory)",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Custom filename for results (default: auto-generated)",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    # Parse arguments
    args = parser.parse_args()
    main(args)
