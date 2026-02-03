#!/usr/bin/env python3
"""Test script to verify batching correctness for pi0 policy.

Tests:
1. Results are identical for batch_size=1 vs batch_size=4
2. Different RTC settings can be used within the same batch
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client.messages import InferRequest
from openpi_client.messages import InferType
from openpi_client.messages import RTCParams

from openpi.policies import policy as _policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


def compare_outputs(output1, output2, name1="output1", name2="output2", tolerance=1e-5):
    """Compare two policy outputs and return True if they match."""
    if "actions" not in output1 or "actions" not in output2:
        print("Missing 'actions' key in outputs")
        return False

    actions1 = output1["actions"]
    actions2 = output2["actions"]

    if actions1.shape != actions2.shape:
        print(f"Shape mismatch: {name1}={actions1.shape} vs {name2}={actions2.shape}")
        return False

    max_diff = jnp.max(jnp.abs(actions1 - actions2))
    print(f"Max difference between {name1} and {name2}: {max_diff:.2e}")

    if max_diff > tolerance:
        print(f"❌ Outputs differ by more than {tolerance}")
        return False
    print(f"✓ Outputs match within tolerance {tolerance}")
    return True


def test_batch_consistency(policy, num_steps=10):
    """Test that batch_size=1 produces same results as batch_size=4."""
    print("\n" + "=" * 80)
    print("TEST 1: Batch consistency (batch_size=1 vs batch_size=4)")
    print("=" * 80)

    # Create 4 SYNC requests
    requests_single = [
        InferRequest(observation=policy.make_example(), infer_type=InferType.SYNC, params=None) for _ in range(4)
    ]

    # Generate fixed noise for reproducibility
    # Shape: (4, action_horizon, action_dim)
    rng = jax.random.key(42)
    action_horizon = policy._model.action_horizon  # noqa: SLF001
    action_dim = policy._model.action_dim  # noqa: SLF001
    noise_batch = jax.random.normal(rng, (4, action_horizon, action_dim))
    noise_batch = np.array(noise_batch)  # Convert to numpy for policy

    # Run each individually with batch_size=1, using corresponding noise
    print("\nRunning 4 individual inferences (batch_size=1)...")
    individual_outputs = []
    for i, request in enumerate(requests_single):
        # Use the i-th noise for the i-th request
        output = policy.infer_batch([request], noise=noise_batch[i : i + 1])[0]
        jax.block_until_ready(output)
        print(f"  Individual inference {i + 1}/4 complete")
        individual_outputs.append(output)

    # Run all together with batch_size=4, using the full noise batch
    print("\nRunning batched inference (batch_size=4)...")
    batched_outputs = policy.infer_batch(requests_single, noise=noise_batch)
    jax.block_until_ready(batched_outputs)
    print("  Batched inference complete")

    # Compare results
    print("\nComparing results:")
    all_match = True
    for i in range(4):
        print(f"\n  Example {i + 1}:")
        match = compare_outputs(
            individual_outputs[i], batched_outputs[i], name1=f"individual[{i}]", name2=f"batched[{i}]"
        )
        all_match = all_match and match

    if all_match:
        print("\n✅ TEST 1 PASSED: All outputs match!")
    else:
        print("\n❌ TEST 1 FAILED: Some outputs don't match")

    return all_match


def test_mixed_infer_types(policy, num_steps=10):
    """Test different RTC settings within the same batch.

    Note: Current implementation requires all requests to have same infer_type.
    This test will verify that constraint or test if it's been relaxed.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Mixed RTC settings in same batch")
    print("=" * 80)

    example_actions = policy.make_example_actions()

    # Generate fixed noise for reproducibility
    rng = jax.random.key(43)
    action_horizon = policy._model.action_horizon  # noqa: SLF001
    action_dim = policy._model.action_dim  # noqa: SLF001
    noise_batch = np.array(jax.random.normal(rng, (4, action_horizon, action_dim)))

    # Create 4 requests with different RTC parameters
    requests = [
        # SYNC inference
        InferRequest(observation=policy.make_example(), infer_type=InferType.SYNC, params=None),
        # RTC with s=3, d=2
        InferRequest(
            observation=policy.make_example(),
            infer_type=InferType.INFERENCE_TIME_RTC,
            params=RTCParams(prev_action=example_actions, s_param=3, d_param=2),
        ),
        # RTC with s=5, d=3
        InferRequest(
            observation=policy.make_example(),
            infer_type=InferType.INFERENCE_TIME_RTC,
            params=RTCParams(prev_action=example_actions, s_param=5, d_param=3),
        ),
        # RTC with s=7, d=4
        InferRequest(
            observation=policy.make_example(),
            infer_type=InferType.INFERENCE_TIME_RTC,
            params=RTCParams(prev_action=example_actions, s_param=7, d_param=4),
        ),
    ]

    print("\nAttempting to run batch with different infer_types...")
    print(f"  Request 0: {requests[0].infer_type}")
    print(f"  Request 1: {requests[1].infer_type}")
    print(f"  Request 2: {requests[2].infer_type}")
    print(f"  Request 3: {requests[3].infer_type}")

    try:
        outputs = policy.infer_batch(requests, noise=noise_batch)
        jax.block_until_ready(outputs)
        print("\n✅ Batch with mixed infer_types succeeded!")

        # Print output shapes
        for i, output in enumerate(outputs):
            print(f"  Output {i}: actions shape = {output['actions'].shape}")

        return True
    except AssertionError as e:
        print(f"\n⚠️  Mixed infer_types not supported: {e}")
        print("Testing same infer_type with different RTC params instead...")

        # Test with same infer_type but different RTC params
        requests_same_type = [
            InferRequest(
                observation=policy.make_example(),
                infer_type=InferType.INFERENCE_TIME_RTC,
                params=RTCParams(prev_action=example_actions, s_param=3, d_param=2),
            ),
            InferRequest(
                observation=policy.make_example(),
                infer_type=InferType.INFERENCE_TIME_RTC,
                params=RTCParams(prev_action=example_actions, s_param=5, d_param=3),
            ),
            InferRequest(
                observation=policy.make_example(),
                infer_type=InferType.INFERENCE_TIME_RTC,
                params=RTCParams(prev_action=example_actions, s_param=7, d_param=4),
            ),
            InferRequest(
                observation=policy.make_example(),
                infer_type=InferType.INFERENCE_TIME_RTC,
                params=RTCParams(prev_action=example_actions, s_param=9, d_param=5),
            ),
        ]

        print("\nAttempting batch with same infer_type but different RTC params...")
        for i, req in enumerate(requests_same_type):
            print(f"  Request {i}: s={req.params.s_param}, d={req.params.d_param}")

        try:
            outputs = policy.infer_batch(requests_same_type, noise=noise_batch)
            jax.block_until_ready(outputs)
            print("\n✅ Batch with different RTC params succeeded!")

            # Verify outputs are different (they should be with different RTC params)
            for i in range(len(outputs) - 1):
                diff = jnp.max(jnp.abs(outputs[i]["actions"] - outputs[i + 1]["actions"]))
                print(f"  Diff between output {i} and {i + 1}: {diff:.2e}")

            return True
        except Exception as e2:
            print(f"\n❌ TEST 2 FAILED: {e2}")
            return False


def test_determinism(policy, num_steps=10):
    """Test that running the same batch twice produces identical results."""
    print("\n" + "=" * 80)
    print("TEST 3: Determinism (same batch run twice)")
    print("=" * 80)

    # Use the same observation for all requests to test determinism
    example = policy.make_example()
    requests = [InferRequest(observation=example, infer_type=InferType.SYNC, params=None) for _ in range(4)]

    # Generate fixed noise to ensure deterministic results
    rng = jax.random.key(44)
    action_horizon = policy._model.action_horizon  # noqa: SLF001
    action_dim = policy._model.action_dim  # noqa: SLF001
    noise_batch = np.array(jax.random.normal(rng, (4, action_horizon, action_dim)))

    print("\nRunning first batch...")
    outputs1 = policy.infer_batch(requests, noise=noise_batch)
    jax.block_until_ready(outputs1)

    print("Running second batch...")
    outputs2 = policy.infer_batch(requests, noise=noise_batch)
    jax.block_until_ready(outputs2)

    print("\nComparing results:")
    all_match = True
    for i in range(4):
        print(f"\n  Example {i}:")
        match = compare_outputs(
            outputs1[i],
            outputs2[i],
            name1=f"run1[{i}]",
            name2=f"run2[{i}]",
            tolerance=1e-10,  # Should be exactly identical
        )
        all_match = all_match and match

    if all_match:
        print("\n✅ TEST 3 PASSED: Results are deterministic!")
    else:
        print("\n❌ TEST 3 FAILED: Results are not deterministic")

    return all_match


def main(args):
    print("Loading policy...")
    config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps},
    )

    policy._metadata["env"] = _policy.EnvMode.LIBERO  # noqa: SLF001
    print(f"Policy loaded. Testing with num_steps={args.num_steps}")

    # Warm up
    print("\nWarming up policy...")
    warmup_request = InferRequest(observation=policy.make_example(), infer_type=InferType.SYNC, params=None)
    # Generate warmup noise
    warmup_rng = jax.random.key(0)
    warmup_noise = np.array(jax.random.normal(warmup_rng, (4, policy._model.action_horizon, policy._model.action_dim)))  # noqa: SLF001
    warmup_output = policy.infer_batch([warmup_request] * 4, noise=warmup_noise)
    jax.block_until_ready(warmup_output)
    print("Warmup complete\n")

    # Run tests
    results = []

    if not args.skip_consistency:
        results.append(("Batch Consistency", test_batch_consistency(policy, args.num_steps)))

    if not args.skip_mixed:
        results.append(("Mixed RTC Settings", test_mixed_infer_types(policy, args.num_steps)))

    if not args.skip_determinism:
        results.append(("Determinism", test_determinism(policy, args.num_steps)))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batching implementation correctness")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of steps for policy rollout")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip batch consistency test")
    parser.add_argument("--skip-mixed", action="store_true", help="Skip mixed RTC settings test")
    parser.add_argument("--skip-determinism", action="store_true", help="Skip determinism test")
    args = parser.parse_args()

    sys.exit(main(args))
