import argparse
import os

import jax
from openpi_client.messages import InferRequest
from openpi_client.messages import InferType

from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "


def main(args):
    config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

    # Create a trained policy.
    policy = policy_config.create_trained_policy(config, checkpoint_dir, sample_kwargs={"num_steps": args.num_steps})

    # Run inference on a dummy example.
    example = libero_policy.make_libero_example()
    request = InferRequest(robot_id="test_robot", observation=example, infer_type=InferType.SYNC, params=None)
    requests = [request] * args.batch_size

    print("Warming up...")
    outputs = policy.infer_batch(requests)
    jax.block_until_ready(outputs)
    print("Warmed up")

    print("Inferring...")
    for _ in range(5):
        outputs = policy.infer_batch(requests)
        jax.block_until_ready(outputs)
    print("Inference complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    main(args)
