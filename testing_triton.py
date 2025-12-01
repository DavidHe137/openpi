# from transformers import AutoProcessor, AutoModelForVision2Seq

# processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
# model = AutoModelForVision2Seq.from_pretrained("google/paligemma-3b-pt-224")

# from openpi.training import config as _config
# from openpi.policies import policy_config
# from openpi.shared import download

# config = _config.get_config("pi0_libero")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import libero_policy
import time
import numpy as np

def test_n_inferences(policy, n, warmup=False):
    inference_times = []
    for i in range(n):
        example = libero_policy.make_libero_example()
        # print(f"Inferring {i+1}...")
        start_time = time.time()
        result = policy.infer(example)
        inference_times.append(time.time() - start_time)
        # print("Done. took", inference_times[-1], "seconds")
        # print("Actions shape:", result["actions"].shape)

    if not warmup:
        print("Average inference time:", np.mean(inference_times))
        print("Median inference time:", np.median(inference_times))
        print("Max inference time:", np.max(inference_times))
        print("Min inference time:", np.min(inference_times))

def test_triton():
    config = _config.get_config("pi0_libero")
    checkpoint_dir = ".cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_dexmal"

    print("Loading triton policy...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir, use_triton_optimized=True)
    print("Loaded policy.")

    print("Cold start inference...")
    test_n_inferences(policy, 3, warmup=True)

    print("Inference...")
    test_n_inferences(policy, 100)
    del policy

def test_pytorch():

    config = _config.get_config("pi0_libero")
    checkpoint_dir = ".cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch_openpi"

    print("Loading openpi policy...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir, use_triton_optimized=False)
    print("Loaded policy.")

    print("Cold start inference...")
    test_n_inferences(policy, 3, warmup=True)

    print("Inference...")
    test_n_inferences(policy, 100)

    del policy

if __name__ == "__main__":
    test_triton()
    test_pytorch()
