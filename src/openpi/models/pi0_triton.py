# src/openpi/models/pi0_triton.py (or pi0_fast_triton.py)

import pickle
import torch
import numpy as np
from typing_extensions import override
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

from openpi.shared.pi0_infer import Pi0Inference

class Pi0Triton(_model.BaseModel):
    """Optimized Pi0 model using Triton kernels for real-time inference.
    
    This is a PyTorch-based implementation that provides significantly faster
    inference than the standard Pi0Pytorch model by using custom Triton kernels.
    
    Usage:
        This model requires a converted checkpoint created using convert_from_jax.py
        from the realtime-vla project.
    """
    
    def __init__(self, config: pi0_config.Pi0Config, converted_checkpoint_path: str, num_views: int = 2):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        
        # Load the converted checkpoint
        with open(converted_checkpoint_path, 'rb') as f:
            converted_checkpoint = pickle.load(f)
        
        # Initialize the fast inference engine
        self.inference_engine = Pi0Inference(
            converted_checkpoint, 
            num_views=num_views, 
            chunk_size=config.action_horizon
        )
        self.config = config
        self.num_views = num_views
        
    def to(self, device):
        """Move model to device (for PyTorch compatibility).
        
        Note: Triton kernels require CUDA, so this is mostly a no-op
        but needed for compatibility with the Policy interface.
        """
        if device != 'cuda' and 'cuda' not in str(device):
            raise ValueError(
                "Pi0Triton requires CUDA device. Triton kernels are GPU-only."
            )
        self._device = device
        return self
    
    def eval(self):
        """Set model to evaluation mode (for PyTorch compatibility).
        
        This is a no-op since Pi0Triton is inference-only.
        """
        return self

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute loss for training.
        
        Note: This method is not implemented for the Triton-optimized model,
        which is inference-only.
        """
        raise NotImplementedError(
            "Pi0Triton is an inference-only model and does not support training. "
            "Use PI0Pytorch for training."
        )
    
    @override
    def guided_inference(
        self, 
        rng: at.KeyArrayLike, 
        prev_action: _model.Actions, 
        observation: _model.Observation, 
        **kwargs
    ) -> _model.Actions:
        """Perform guided inference with previous actions.
        
        Note: Guided inference is not currently implemented for the Triton-optimized model.
        Falls back to standard sample_actions.
        """
        # For now, just call sample_actions and ignore prev_action
        # You could implement RTC (Receding Time Control) here if needed
        return self.sample_actions(rng, observation, **kwargs)   

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )    

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    @override
    @torch.no_grad()
    def sample_actions(self, device, observation: _model.Observation, 
                    noise=None, num_steps=10) -> torch.Tensor:
        """Sample actions using the optimized Triton kernels.
        
        Note: This implementation processes batch items sequentially as the underlying
        Triton kernels are optimized for single-sample inference.
        """
        
        # Preprocess observation (same as Pi0.sample_actions line 225)
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        images = images[:self.num_views]
        
        # Process each batch item
        batch_actions = []
        for batch_idx in range(bsize):
            # Extract and convert images for this batch item
            images_converted = []
            for img in images:
                img_sample = img[batch_idx]
                img_sample = img_sample.permute(1, 2, 0)  # Convert to (H, W, C)
                images_converted.append(img_sample)
            
            images_stacked = torch.stack(images_converted, dim=0)  # (num_views, H, W, C)
            images_bf16 = images_stacked.to(dtype=torch.bfloat16, device='cuda')
            
            state_bf16 = state[batch_idx].to(dtype=torch.bfloat16, device='cuda')
            noise_input = noise[batch_idx].to(dtype=torch.bfloat16, device='cuda')
            
            # Run inference for this sample
            output_actions = self.inference_engine.forward(
                images_bf16,
                state_bf16,
                noise_input
            )
            
            batch_actions.append(output_actions)
        
        # Stack all batch results: (batch_size, action_horizon, action_dim)
        return torch.stack(batch_actions, dim=0).to(dtype=torch.float32)
    
    @classmethod
    def from_converted_checkpoint(cls, config: pi0_config.Pi0Config, 
                                  checkpoint_path: str, num_views: int = 2):
        """Factory method to create a Pi0Triton model from a converted checkpoint."""
        return cls(config, checkpoint_path, num_views)