#!/usr/bin/env python3
"""
Benchmark script for comparing single-sample vs batched Triton inference.
Tests Pi0Inference vs Pi0InferenceBatched performance.
"""

import argparse
import pickle
import time
from typing import Any

import numpy as np
import torch


def create_mock_checkpoint(prompt_len: int = 32) -> dict[str, torch.Tensor]:
    """Create mock checkpoint weights for testing."""
    return {
        "language_embeds": torch.randn(prompt_len, 2048, dtype=torch.bfloat16, device="cuda"),
        # Vision encoder weights
        "vision_patch_embedding_w": torch.randn(14, 14, 3, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_patch_embedding_b": torch.randn(1152, dtype=torch.bfloat16, device="cuda"),
        "vision_position_embedding": torch.randn(256, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_attn_qkv_w": torch.randn(27, 1152, 3456, dtype=torch.bfloat16, device="cuda"),
        "vision_attn_qkv_b": torch.randn(27, 3456, dtype=torch.bfloat16, device="cuda"),
        "vision_attn_o_w": torch.randn(27, 1152, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_attn_o_b": torch.randn(27, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_ffn_up_w": torch.randn(27, 1152, 4304, dtype=torch.bfloat16, device="cuda"),
        "vision_ffn_up_b": torch.randn(27, 4304, dtype=torch.bfloat16, device="cuda"),
        "vision_ffn_down_w": torch.randn(27, 4304, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_ffn_down_b": torch.randn(27, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_pre_attn_norm_w": torch.randn(27, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_pre_attn_norm_b": torch.randn(27, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_pre_ffn_norm_w": torch.randn(27, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_pre_ffn_norm_b": torch.randn(27, 1152, dtype=torch.bfloat16, device="cuda"),
        "vision_final_norm_w": torch.randn(1152, dtype=torch.bfloat16, device="cuda"),
        "vision_final_norm_b": torch.randn(1152, dtype=torch.bfloat16, device="cuda"),
        # Encoder weights
        "encoder_multi_modal_projector_w": torch.randn(1152, 2048, dtype=torch.bfloat16, device="cuda"),
        "encoder_multi_modal_projector_b": torch.randn(2048, dtype=torch.bfloat16, device="cuda"),
        "encoder_attn_qkv_w": torch.randn(18, 2048, 2560, dtype=torch.bfloat16, device="cuda"),
        "encoder_attn_o_w": torch.randn(18, 2048, 2048, dtype=torch.bfloat16, device="cuda"),
        "encoder_ffn_gate_w": torch.randn(18, 2048, 16384, dtype=torch.bfloat16, device="cuda"),
        "encoder_ffn_up_w": torch.randn(18, 2048, 16384, dtype=torch.bfloat16, device="cuda"),
        "encoder_ffn_down_w": torch.randn(18, 16384, 2048, dtype=torch.bfloat16, device="cuda"),
        # Decoder weights
        "decoder_state_in_proj_w": torch.randn(32, 1024, dtype=torch.bfloat16, device="cuda"),
        "decoder_state_in_proj_b": torch.randn(1024, dtype=torch.bfloat16, device="cuda"),
        "decoder_action_fused_in_proj_w": torch.randn(32, 1024, dtype=torch.bfloat16, device="cuda"),
        "decoder_action_fused_time_biases": torch.randn(10, 1024, dtype=torch.bfloat16, device="cuda"),
        "decoder_action_mlp_w": torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda"),
        "decoder_action_mlp_b": torch.randn(1024, dtype=torch.bfloat16, device="cuda"),
        "decoder_attn_qkv_w": torch.randn(18, 1024, 2560, dtype=torch.bfloat16, device="cuda"),
        "decoder_attn_o_w": torch.randn(18, 2048, 1024, dtype=torch.bfloat16, device="cuda"),
        "decoder_ffn_gate_w": torch.randn(18, 1024, 4096, dtype=torch.bfloat16, device="cuda"),
        "decoder_ffn_up_w": torch.randn(18, 1024, 4096, dtype=torch.bfloat16, device="cuda"),
        "decoder_ffn_down_w": torch.randn(18, 4096, 1024, dtype=torch.bfloat16, device="cuda"),
        "decoder_action_fused_out_proj_w": torch.randn(1024, 32, dtype=torch.bfloat16, device="cuda"),
        "decoder_action_fused_out_proj_b": torch.randn(32, dtype=torch.bfloat16, device="cuda"),
    }


def create_single_input(num_views: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create single sample inputs."""
    images = torch.randn(num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    state = torch.randn(32, dtype=torch.bfloat16, device="cuda")
    noise = torch.randn(chunk_size, 32, dtype=torch.bfloat16, device="cuda")
    return images, state, noise


def create_batch_input(
    batch_size: int, num_views: int, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create batched inputs."""
    images = torch.randn(batch_size, num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    state = torch.randn(batch_size, 32, dtype=torch.bfloat16, device="cuda")
    noise = torch.randn(batch_size, chunk_size, 32, dtype=torch.bfloat16, device="cuda")
    return images, state, noise


def benchmark_single(
    model, num_views: int, chunk_size: int, num_warmup: int = 5, num_iters: int = 20
) -> dict[str, Any]:
    """Benchmark single-sample inference."""
    images, state, noise = create_single_input(num_views, chunk_size)

    # Warmup
    for _ in range(num_warmup):
        _ = model.forward(images, state, noise)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model.forward(images, state, noise)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "throughput": 1.0 / np.mean(times),
    }


def benchmark_single_loop(
    model, batch_size: int, num_views: int, chunk_size: int, num_warmup: int = 5, num_iters: int = 20
) -> dict[str, Any]:
    """Benchmark single-sample inference in a loop (simulating batch processing)."""
    inputs = [create_single_input(num_views, chunk_size) for _ in range(batch_size)]

    # Warmup
    for _ in range(num_warmup):
        for images, state, noise in inputs:
            _ = model.forward(images, state, noise)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for images, state, noise in inputs:
            _ = model.forward(images, state, noise)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "throughput": batch_size / np.mean(times),
    }


def benchmark_batched(
    model, batch_size: int, num_views: int, chunk_size: int, num_warmup: int = 5, num_iters: int = 20
) -> dict[str, Any]:
    """Benchmark batched inference."""
    images, state, noise = create_batch_input(batch_size, num_views, chunk_size)

    # Warmup
    for _ in range(num_warmup):
        _ = model.forward(images, state, noise)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model.forward(images, state, noise)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "throughput": batch_size / np.mean(times),
    }


def print_results(name: str, results: dict[str, Any], batch_size: int = 1):
    """Print benchmark results."""
    print(f"\n{name}:")
    print(f"  Latency: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
    print(f"  Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput']:.2f} samples/sec")
    if batch_size > 1:
        print(f"  Per-sample: {results['mean_ms'] / batch_size:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton Pi0 inference")
    parser.add_argument("--num-views", type=int, default=2, help="Number of camera views")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action chunk size")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes to test")
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num-iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--prompt-len", type=int, default=32, help="Prompt token length")
    args = parser.parse_args()

    print("=" * 70)
    print("Triton Pi0 Inference Benchmark")
    print("=" * 70)
    print(f"Config: num_views={args.num_views}, chunk_size={args.chunk_size}, prompt_len={args.prompt_len}")
    print(f"Benchmark: {args.num_warmup} warmup, {args.num_iters} iterations")
    print(f"Batch sizes: {args.batch_sizes}")

    # Import here to avoid issues if not installed
    from openpi.shared.pi0_infer_batched import Pi0Inference
    from openpi.shared.pi0_infer_batched import Pi0InferenceBatched

    with open(
        "/coc/flash8/rbansal66/openpi_rollout/openpi/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_dexmal_mokapots/converted_checkpoint.pkl",
        "rb",
    ) as f:
        weights = pickle.load(f)
    # Create single-sample model
    print("Initializing single-sample model (Pi0Inference)...")
    single_model = Pi0Inference(weights, args.num_views, args.chunk_size)

    # Benchmark single sample
    print("\n" + "-" * 70)
    print("Single Sample Baseline")
    print("-" * 70)
    single_results = benchmark_single(single_model, args.num_views, args.chunk_size, args.num_warmup, args.num_iters)
    print_results("Single Sample (Pi0Inference)", single_results)

    # Benchmark each batch size
    all_results = {"single": single_results}

    for batch_size in args.batch_sizes:
        print("\n" + "-" * 70)
        print(f"Batch Size: {batch_size}")
        print("-" * 70)

        # Single model in loop (baseline for comparison)
        loop_results = benchmark_single_loop(
            single_model, batch_size, args.num_views, args.chunk_size, args.num_warmup, args.num_iters
        )
        print_results(f"Single Model Loop (batch={batch_size})", loop_results, batch_size)
        all_results[f"loop_{batch_size}"] = loop_results

        # Batched model
        print(f"\nInitializing batched model (Pi0InferenceBatched, batch_size={batch_size})...")
        try:
            batched_model = Pi0InferenceBatched(weights, args.num_views, args.chunk_size, batch_size)

            batched_results = benchmark_batched(
                batched_model, batch_size, args.num_views, args.chunk_size, args.num_warmup, args.num_iters
            )
            print_results(f"Batched Model (batch={batch_size})", batched_results, batch_size)
            all_results[f"batched_{batch_size}"] = batched_results

            # Calculate speedup
            speedup = loop_results["mean_ms"] / batched_results["mean_ms"]
            throughput_gain = batched_results["throughput"] / loop_results["throughput"]
            print(f"\n  Speedup vs loop: {speedup:.2f}x")
            print(f"  Throughput gain: {throughput_gain:.2f}x")

            del batched_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<30} {'Latency (ms)':<15} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 70)

    baseline_throughput = all_results["single"]["throughput"]
    print(
        f"{'Single (baseline)':<30} {all_results['single']['mean_ms']:<15.2f} {all_results['single']['throughput']:<15.2f} {'1.00x':<10}"
    )

    for batch_size in args.batch_sizes:
        loop_key = f"loop_{batch_size}"
        batched_key = f"batched_{batch_size}"

        if loop_key in all_results:
            speedup = all_results[loop_key]["throughput"] / baseline_throughput
            print(
                f"{f'Loop (batch={batch_size})':<30} {all_results[loop_key]['mean_ms']:<15.2f} {all_results[loop_key]['throughput']:<15.2f} {f'{speedup:.2f}x':<10}"
            )

        if batched_key in all_results:
            speedup = all_results[batched_key]["throughput"] / baseline_throughput
            print(
                f"{f'Batched (batch={batch_size})':<30} {all_results[batched_key]['mean_ms']:<15.2f} {all_results[batched_key]['throughput']:<15.2f} {f'{speedup:.2f}x':<10}"
            )

    print("=" * 70)
    print("Benchmark complete!")

    # Cleanup
    del single_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
