from openpi.training import config as _config
from openpi.policies import policy_config, libero_policy
from openpi.shared import download

config = _config.get_config("pi05_libero")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = libero_policy.make_libero_example()
action_chunk = policy.infer(example)["actions"]
