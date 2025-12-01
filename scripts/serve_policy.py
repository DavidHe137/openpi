import dataclasses
import logging
import socket
from typing import Any

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import EnvMode
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8080
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Batch size to use for inference.
    batch_size: int = 1

    # Number of steps to use for sampling.
    num_steps: int = 10


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.LIBERO_REALTIME: Checkpoint(
        config="pi0_libero",
        dir=".cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_dexmal",
    ),
}


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, sample_kwargs: dict[str, Any] | None = None
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            sample_kwargs=sample_kwargs,
            use_triton_optimized=(env == EnvMode.LIBERO_REALTIME),
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs={"num_steps": args.num_steps},
                use_triton_optimized=(args.env == EnvMode.LIBERO_REALTIME),
            )
        case Default():
            return create_default_policy(
                args.env, default_prompt=args.default_prompt, sample_kwargs={"num_steps": args.num_steps}
            )


def main(args: Args) -> None:
    # Create policy factory to avoid CUDA context fork issues
    def policy_factory():
        policy = create_policy(args)
        # Ensure policy metadata includes env for make_example()
        if "env" not in policy._metadata:
            policy._metadata["env"] = args.env.value
        # Record the policy's behavior.
        if args.record:
            policy = _policy.PolicyRecorder(policy, "policy_records")
        return policy

    # Build metadata without loading the model to avoid CUDA initialization
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                train_config = _config.get_config(checkpoint.config)
            else:
                raise ValueError(f"Unsupported environment mode: {args.env}")

    policy_metadata = train_config.policy_metadata or {}
    policy_metadata["num_steps"] = args.num_steps
    policy_metadata["action_horizon"] = train_config.model.action_horizon
    policy_metadata["env"] = args.env.value
    policy_metadata["batch_size"] = args.batch_size

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy_factory=policy_factory,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        batch_size=args.batch_size,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
