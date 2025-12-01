import argparse

import jax

from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


def main(args):
    config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

    # Create a trained policy.
    policy = policy_config.create_trained_policy(config, checkpoint_dir, sample_kwargs={"num_steps": args.num_steps})
    # Run inference on a dummy example.
    example = libero_policy.make_libero_example()
    examples = [example] * args.batch_size

    print("Warming up...")
    outputs = policy.infer_batch(examples)
    jax.block_until_ready(outputs)
    print("Warmed up")

    flow_matching_states = []
    for i in range(args.num_steps):
        flow_matching_state = policy.make_flow_matching_example()
        flow_matching_state.time = 1.0 - 0.1 * i
        flow_matching_states.append(flow_matching_state)

    while flow_matching_states:
        flow_matching_states = policy._policy.flow_matching_step(flow_matching_states, args.num_steps)
        new_flow_matching_states = []
        for flow_matching_state in flow_matching_states:
            if flow_matching_state.done:
                print(flow_matching_state.x)
            else:
                new_flow_matching_states.append(flow_matching_state)
        flow_matching_states = new_flow_matching_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    main(args)
