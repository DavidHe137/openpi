import os
import sys

import h5py
import numpy as np
from scipy import sparse
from transformers import AutoProcessor


def sparsify_matrix(matrix, threshold=1e-6):
    # Zero out small elements
    matrix[np.abs(matrix) < threshold] = 0
    # Convert to sparse format
    return sparse.csr_matrix(matrix)


def denormalize_action_mean_std(action_norm, mean, std):
    # action: [t, horizon]
    return action_norm * std + mean


def normalize_action_percentile(actions):
    # following  FAST paper to normalize the action with 1%-99% percentile
    # input: [t, action_dim]
    # output: [t, action_dim], min, max
    min_val = np.percentile(actions, 1, axis=0)
    max_val = np.percentile(actions, 99, axis=0)
    return (actions - min_val) / (max_val - min_val), min_val, max_val


def denormalize_action_percentile(action_norm, min_val, max_val):
    # input: [t, action_dim]
    # output: [t, action_dim]
    return action_norm * (max_val - min_val) + min_val


## Read state and action data
state_data = []
action_data = []
with h5py.File("/coc/flash7/zhenyang/data/robomimic-sim/low_dim_v141.hdf5", "r") as f:
    f_dict = f["data"]
    print(f_dict.keys())
    print(f_dict["demo_0"].keys())
    for demo in f_dict:
        state_data.append(f_dict[demo]["states"][:])
        action_data.append(f_dict[demo]["actions"][:])

state_data_array = np.concatenate(state_data, axis=0)
action_data_array = np.concatenate(action_data, axis=0)

print(state_data_array.shape)
print(action_data_array.shape)

###########################
### Normalize the data old ###
# state_mean = np.mean(state_data_array, axis=0)
# state_std = np.std(state_data_array, axis=0)
# state_data_array = (state_data_array - state_mean) / state_std

# action_mean = np.mean(action_data_array, axis=0)
# action_std = np.std(action_data_array, axis=0)
# action_data_norm = (action_data_array - action_mean) / action_std
###########################

###########################
### Normalize data with percentile ###
action_norm, action_min, action_max = normalize_action_percentile(action_data_array)
print(
    f"action_norm shape: {action_norm.shape}, action_min shape: {action_min.shape}, action_max shape: {action_max.shape}"
)
test_action_norm = action_norm[10:30, :]

# Set all cache directories before any HF imports
cache_dir = "/coc/flash7/zhenyang/.cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir

# Clear any existing HF modules from sys.modules
for module in list(sys.modules.keys()):
    if module.startswith(("transformers", "huggingface")):
        del sys.modules[module]

# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True, local_files_only=True)

# Tokenize & decode action chunks (we use dummy data here)
# action_data = np.random.rand(1, 20, 14)    # one batch of action chunks
# action_data = action_data_norm[np.newaxis, :, :]
action_data = test_action_norm[np.newaxis, :, :]
tokens = tokenizer(action_data)  # tokens = list[list[int]]
print(f"[test action norm] tokens number: {len(tokens[0])}")
decoded_actions = tokenizer.decode(tokens)
decoded_actions = denormalize_action_percentile(decoded_actions, action_min, action_max)
print(f"[test action norm] decoded actions are: {decoded_actions.shape}")

##### Test cut-off tokens #####
# tokens_short = tokens[0][:len(tokens[0])-3]
tokens_short = tokens[0][:-2]
tokens_short = [tokens_short]  # list[list[int]]
decoded_actions_short = tokenizer.decode(tokens_short)
decoded_actions_short = denormalize_action_percentile(decoded_actions_short, action_min, action_max)
print(
    f"[test action norm short] tokens_short number {len(tokens_short)} decoded short action shape {decoded_actions_short.shape}"
)

action_denorm = denormalize_action_percentile(action_data, action_min, action_max)
x = action_denorm[..., 0][0]
y = action_denorm[..., 1][0]
z = action_denorm[..., 2][0]

x_rec = decoded_actions[..., 0][0]
y_rec = decoded_actions[..., 1][0]
z_rec = decoded_actions[..., 2][0]

x_rec_short = decoded_actions_short[..., 0][0]
y_rec_short = decoded_actions_short[..., 1][0]
z_rec_short = decoded_actions_short[..., 2][0]

# print(f"x_rec_short: {x_rec_short.shape}, y_rec_short: {y_rec_short.shape}, z_rec_short: {z_rec_short.shape}")
# print(f"x_rec: {x_rec.shape}, y_rec: {y_rec.shape}, z_rec: {z_rec.shape}")
# print(f"x: {x.shape}, y: {y.shape}, z: {z.shape}")
# print(f"error between x_rec and x_rec_short: {np.mean(np.abs(x_rec - x_rec_short))}")

horizon = min(action_denorm.shape[0], decoded_actions_short.shape[0])
print(
    f"decoded short action error is {np.mean(np.abs(decoded_actions_short[0, :horizon, :] - action_denorm[0, :horizon, :]), axis=0)}"
)
print(
    f"original action error is {np.mean(np.abs(decoded_actions[0, :horizon, :] - action_denorm[0, :horizon, :]), axis=0)}"
)
