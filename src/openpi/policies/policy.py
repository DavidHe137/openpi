from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        enable_profiling: bool = True,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            enable_profiling: Whether to enable granular timing profiling. When enabled,
                            uses slower profiled execution path with detailed timing.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._enable_profiling = enable_profiling

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            if enable_profiling and hasattr(model, "sample_actions_profiled"):
                # Use non-JIT profiled version for timing
                self._sample_actions = model.sample_actions_profiled
                # Also JIT the individual components for performance
                self._model._preprocess_observation_jit = nnx_utils.module_jit(model._preprocess_observation_jit)
                self._model._embed_prefix_jit = nnx_utils.module_jit(model._embed_prefix_jit)
                self._model._setup_prefix_kv_cache_jit = nnx_utils.module_jit(model._setup_prefix_kv_cache_jit)
                self._model._run_diffusion_loop_jit = nnx_utils.module_jit(model._run_diffusion_loop_jit)
            else:
                # Use normal JIT-compiled version
                self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        if self._enable_profiling and hasattr(self._model, "sample_actions_profiled"):
            # Use profiled execution path
            actions, granular_timing = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
        else:
            # Use normal execution path
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }
            granular_timing = {}

        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)

        # Collect timing information
        policy_timing = {"infer_ms": model_time * 1000}
        policy_timing.update(granular_timing)

        outputs["policy_timing"] = policy_timing
        return outputs

    def infer_batch(self, obs_batch: list[dict], *, noise: np.ndarray | None = None) -> list[dict]:
        """Run inference on a batch of observations.

        Args:
            obs_batch: List of observation dictionaries
            noise: Optional noise tensor for batch (shape: batch_size, action_horizon, action_dim)

        Returns:
            List of result dictionaries, one for each input observation
        """
        if not obs_batch:
            return []

        # Check if all observations have the same structure
        first_obs = obs_batch[0]
        batch_size = len(obs_batch)

        # Stack observations into batch format
        batched_obs = {}
        for key in first_obs:
            if key in first_obs:
                # Stack all values for this key
                values = [obs[key] for obs in obs_batch]
                if isinstance(values[0], np.ndarray):
                    batched_obs[key] = np.stack(values, axis=0)
                elif isinstance(values[0], dict):
                    # Handle nested dictionaries (like images)
                    batched_obs[key] = {}
                    for subkey in values[0]:
                        subvalues = [obs[key][subkey] for obs in obs_batch]
                        if isinstance(subvalues[0], np.ndarray):
                            batched_obs[key][subkey] = np.stack(subvalues, axis=0)
                        else:
                            batched_obs[key][subkey] = subvalues
                else:
                    batched_obs[key] = values
            else:
                batched_obs[key] = [obs.get(key, None) for obs in obs_batch]

        # Apply transforms to batched observation
        inputs = jax.tree.map(lambda x: x, batched_obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Convert to jax.Array (already batched)
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)

        outputs = self._output_transform(outputs)

        # Split batch results back into individual results
        results = []
        for i in range(batch_size):
            result = {}
            for key, value in outputs.items():
                if key == "policy_timing":
                    result[key] = value  # Timing is shared
                elif isinstance(value, np.ndarray) and len(value.shape) > 0:
                    result[key] = value[i]
                else:
                    result[key] = value
            results.append(result)

        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
