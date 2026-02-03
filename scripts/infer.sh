#!/bin/bash
export JAX_ENABLE_PGLE=true

# For JAX version <= 0.5.0 make sure to include:
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"

export JAX_ENABLE_COMPILATION_CACHE=yes          # not strictly needed, on by default
export JAX_COMPILATION_CACHE_DIR=./jax_cache/
mkdir -p ./jax_cache/
JAX_ENABLE_PGLE=yes uv run -- scripts/infer.py --batch-size 4

JAX_COMPILATION_CACHE_EXPECT_PGLE=yes nsys profile --trace=cuda,nvtx,osrt --force-overwrite true -o nvtx_annotated_pgle_4 uv run -m nvtx -- scripts/infer.py --batch-size 4