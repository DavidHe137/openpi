"""Compare action expert speed: with vs without prefix in KV cache.

Simple comparison following the exact pattern from pi0.py sample_actions.
"""

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import tyro

from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


@dataclass
class Args:
    config_name: str = "pi05_libero_lora"
    """Config to load"""

    checkpoint_dir: str = "./checkpoints/pi05_libero_lora/my_experiment/1000"
    """Path to checkpoint directory"""

    num_warmup: int = 5
    """Number of warmup iterations"""

    num_iterations: int = 50
    """Number of benchmark iterations"""


def compare_speeds(args: Args):
    print("=" * 80)
    print("ACTION EXPERT SPEED COMPARISON")
    print("=" * 80)
    print("Comparing action expert runtime:")
    print("  1. WITHOUT prefix (empty KV cache)")
    print("  2. WITH prefix cached (images + prompt in KV cache)")
    print("=" * 80 + "\n")

    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.checkpoint_dir}\n")

    # Load model exactly like infer.py does
    print("Loading model...")
    config = _config.get_config(args.config_name)
    checkpoint_dir = download.maybe_download(args.checkpoint_dir)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    # Create fake example exactly like infer.py does
    print("Creating fake example...")
    example = libero_policy.make_libero_example()

    # Use policy.infer to get a proper observation (this handles all the transforms)
    print("Preparing observation via policy transforms...")
    _ = policy.infer(example)  # This warms up and validates everything works

    # Now manually create observation using the policy's internal transform
    inputs = policy._input_transform(example)  # noqa: SLF001
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[jnp.newaxis, ...], inputs)

    from openpi.models import model as _model

    observation = _model.Observation.from_dict(inputs)
    observation = _model.preprocess_observation(None, observation, train=False)

    model = policy._model  # noqa: SLF001

    print("\nModel info:")
    print(f"  Action dim: {model.action_dim}")
    print(f"  Action horizon: {model.action_horizon}")
    print(f"  Pi0.5: {model.pi05}")

    # Prepare for benchmarking
    rng = jax.random.PRNGKey(0)
    batch_size = 1

    # =========================================================================
    # SCENARIO 1: Action expert WITHOUT prefix (empty KV cache)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 1: ACTION EXPERT WITHOUT PREFIX")
    print("=" * 80)
    print("Action expert runs standalone with NO prefix in KV cache.\n")

    # Prepare action expert inputs (same as in sample_actions)
    x_t = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))
    time_val = jnp.array([0.5])

    # Embed suffix (action tokens)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time_val)

    from openpi.models.pi0 import make_attn_mask

    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    positions_standalone = jnp.cumsum(suffix_mask, axis=-1) - 1

    def action_expert_no_prefix():
        """Run action expert with NO prefix (empty KV cache)."""
        (pali_out, action_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=suffix_attn_mask,
            positions=positions_standalone,
            kv_cache=None,
            adarms_cond=[None, adarms_cond],
        )
        return model.action_out_proj(action_out[:, -model.action_horizon :])

    # JIT compile
    print("JIT compiling...")
    action_expert_no_prefix_jit = jax.jit(action_expert_no_prefix)

    # First call (compilation)
    compile_start = time.perf_counter()
    v_t = action_expert_no_prefix_jit()
    v_t.block_until_ready()
    compile_time = time.perf_counter() - compile_start
    print(f"First call (compilation): {compile_time:.2f}s\n")

    # Warmup
    print(f"Warming up ({args.num_warmup} iterations)...")
    for _ in range(args.num_warmup):
        v_t = action_expert_no_prefix_jit()
        v_t.block_until_ready()

    # Benchmark
    print(f"Benchmarking ({args.num_iterations} iterations)...")
    times_no_prefix = []
    for i in range(args.num_iterations):
        start = time.perf_counter()
        v_t = action_expert_no_prefix_jit()
        v_t.block_until_ready()
        elapsed = time.perf_counter() - start
        times_no_prefix.append(elapsed)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.num_iterations}")

    times_no_prefix = jnp.array(times_no_prefix)
    mean_no_prefix = jnp.mean(times_no_prefix)
    std_no_prefix = jnp.std(times_no_prefix)

    print("\nResults:")
    print(f"  Mean: {mean_no_prefix * 1000:.2f} ms (± {std_no_prefix * 1000:.2f} ms)")
    print(f"  Median: {jnp.median(times_no_prefix) * 1000:.2f} ms")
    print(f"  Min: {jnp.min(times_no_prefix) * 1000:.2f} ms")
    print(f"  Max: {jnp.max(times_no_prefix) * 1000:.2f} ms")

    # =========================================================================
    # SCENARIO 2: Action expert WITH prefix cached
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 2: ACTION EXPERT WITH PREFIX CACHED")
    print("=" * 80)
    print("Action expert attends to cached prefix (images + prompt).\n")

    # This follows the exact pattern from pi0.py lines 233-237
    print("Building KV cache from prefix...")
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions_prefix)

    print(f"Prefix length: {prefix_tokens.shape[1]} tokens")
    print(f"KV cache shape: K={kv_cache[0].shape}, V={kv_cache[1].shape}")

    # This follows the exact pattern from pi0.py lines 241-269
    import einops

    suffix_attn_mask_with_prefix = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask_expanded = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask_expanded, suffix_attn_mask_with_prefix], axis=-1)
    positions_with_prefix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    def action_expert_with_prefix():
        """Run action expert WITH prefix cached."""
        (pali_out, action_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_with_prefix,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return model.action_out_proj(action_out[:, -model.action_horizon :])

    # JIT compile
    print("\nJIT compiling...")
    action_expert_with_prefix_jit = jax.jit(action_expert_with_prefix)

    # First call (compilation)
    compile_start = time.perf_counter()
    v_t = action_expert_with_prefix_jit()
    v_t.block_until_ready()
    compile_time = time.perf_counter() - compile_start
    print(f"First call (compilation): {compile_time:.2f}s\n")

    # Warmup
    print(f"Warming up ({args.num_warmup} iterations)...")
    for _ in range(args.num_warmup):
        v_t = action_expert_with_prefix_jit()
        v_t.block_until_ready()

    # Benchmark
    print(f"Benchmarking ({args.num_iterations} iterations)...")
    times_with_prefix = []
    for i in range(args.num_iterations):
        start = time.perf_counter()
        v_t = action_expert_with_prefix_jit()
        v_t.block_until_ready()
        elapsed = time.perf_counter() - start
        times_with_prefix.append(elapsed)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.num_iterations}")

    times_with_prefix = jnp.array(times_with_prefix)
    mean_with_prefix = jnp.mean(times_with_prefix)
    std_with_prefix = jnp.std(times_with_prefix)

    print("\nResults:")
    print(f"  Mean: {mean_with_prefix * 1000:.2f} ms (± {std_with_prefix * 1000:.2f} ms)")
    print(f"  Median: {jnp.median(times_with_prefix) * 1000:.2f} ms")
    print(f"  Min: {jnp.min(times_with_prefix) * 1000:.2f} ms")
    print(f"  Max: {jnp.max(times_with_prefix) * 1000:.2f} ms")

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("\n1. Action expert WITHOUT prefix (empty KV cache):")
    print(f"   {mean_no_prefix * 1000:.2f} ms per forward pass")

    print("\n2. Action expert WITH prefix cached (images + prompt):")
    print(f"   {mean_with_prefix * 1000:.2f} ms per forward pass")

    overhead = mean_with_prefix - mean_no_prefix
    overhead_pct = (overhead / mean_no_prefix) * 100

    print("\n📊 OVERHEAD ANALYSIS:")
    print(f"   • Overhead from attending to cached prefix: {overhead * 1000:.2f} ms ({overhead_pct:.1f}%)")

    if overhead_pct > 5:
        print(f"   • The cached prefix adds {overhead_pct:.1f}% overhead to action expert runtime")
        print(f"   • Prefix tokens in cache: {prefix_tokens.shape[1]}")
        print("   • This overhead comes from attention to the longer key/value sequence")
    else:
        print(f"   • Negligible overhead ({overhead_pct:.1f}%) from cached prefix")

    slowdown_factor = mean_with_prefix / mean_no_prefix
    print(f"\n   • Slowdown factor: {slowdown_factor:.2f}x")

    print("=" * 80)


if __name__ == "__main__":
    tyro.cli(compare_speeds)
