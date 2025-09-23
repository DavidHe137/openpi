import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    
    # Handle both single images and batch of images
    if len(image.shape) == 4:  # Batch of images: (batch, c, h, w)
        if image.shape[1] == 3:  # Channel dimension is at index 1
            image = einops.rearrange(image, "b c h w -> b h w c")
    elif len(image.shape) == 3:  # Single image: (c, h, w)
        if image.shape[0] == 3:  # Channel dimension is at index 0
            image = einops.rearrange(image, "c h w -> h w c")
    
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Handle batch dimensions for state and masks
        state = data["observation/state"]
        is_batch = len(state.shape) > 1
        
        if is_batch:
            # Batch processing: create appropriate masks for each sample in batch
            batch_size = state.shape[0]
            base_mask = np.ones(batch_size, dtype=bool)
            wrist_mask = np.ones(batch_size, dtype=bool)
            right_wrist_mask = np.ones(batch_size, dtype=bool) if self.model_type == _model.ModelType.PI0_FAST else np.zeros(batch_size, dtype=bool)
        else:
            # Single sample processing (original behavior)
            base_mask = np.True_
            wrist_mask = np.True_
            right_wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": base_mask,
                "left_wrist_0_rgb": wrist_mask,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data["actions"])

        # Handle both single sample and batch processing
        if len(actions.shape) == 3:  # Batch: (batch_size, action_horizon, action_dim)
            # Keep full horizon, slice action dim
            return {"actions": actions[:, :, :7]}
        elif len(actions.shape) == 2:  # Single: (action_horizon, action_dim)
            # Keep full horizon, slice action dim
            return {"actions": actions[:, :7]}
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")
