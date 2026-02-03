from collections.abc import Sequence
import enum
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferResponse
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LIBERO_PI0 = "libero_pi0"
    LIBERO_PYTORCH = "libero_pytorch"
    LIBERO_REALTIME = "libero_realtime"


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
        is_triton_optimized: bool = False,
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
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._is_triton_optimized = is_triton_optimized

        if self._is_pytorch_model:
            # assert isinstance(self._model, torch.nn.Module), "Model must be a PyTorch model"
            self._model = self._model.to(pytorch_device)
            self._model.eval()
        else:
            self._rng = rng or jax.random.key(0)
            self._model.sample_actions = nnx_utils.module_jit(
                self._model.sample_actions, static_argnames=["return_debug_data", "use_rtc"]
            )
        self._sample_actions = model.sample_actions

    @override
    def infer(
        self,
        obs: dict,
        *,
        prev_action: np.ndarray | None = None,
        use_rtc: bool = False,
        noise: np.ndarray | None = None,
        s_param: int = 5,
        d_param: int = 4,
        return_debug_data: bool = False,
    ) -> dict:  # type: ignore[misc]
        raw_obs = None
        if return_debug_data:
            raw_obs = jax.tree.map(lambda x: np.array(x) if hasattr(x, "__array__") else x, obs)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if self._is_triton_optimized:
            # Triton policy expects already-repacked LIBERO dict keys (base_0_rgb, etc) and
            # applies its own preprocessing/normalization internally.
            triton_obs: dict[str, Any] = {
                "state": np.asarray(inputs["observation/state"])[None, ...],
                "base_0_rgb": np.asarray(inputs["observation/image"])[None, ...],
                "left_wrist_0_rgb": np.asarray(inputs["observation/wrist_image"])[None, ...],
                "right_wrist_0_rgb": np.asarray(inputs["observation/wrist_image"])[None, ...],
                # Keep prompt as object array so downstream tokenization can `.item()` if needed.
                "prompt": np.asarray([inputs.get("prompt", "")], dtype=object),
            }

            sample_kwargs = dict(self._sample_kwargs)
            sample_kwargs["return_debug_data"] = return_debug_data
            if noise is not None:
                sample_kwargs["noise"] = np.asarray(noise)

            sample_rng_or_pytorch_device = self._pytorch_device
            actions, debug_data = self._sample_actions(sample_rng_or_pytorch_device, triton_obs, **sample_kwargs)

            actions_np = np.asarray(actions)
            if actions_np.ndim == 3 and actions_np.shape[0] == 1:
                actions_np = actions_np[0]

            # Build outputs in the same *normalized* space as the JAX model path:
            # - `actions`: normalized (H, 32) from Triton model
            # - `state`: normalized (32,) computed from raw state and checkpoint norm stats
            ns = getattr(self._model, "norm_stats", None)
            raw_state = np.asarray(inputs["observation/state"], dtype=np.float32)
            state_norm = np.pad(raw_state, (0, max(0, 32 - raw_state.shape[-1])), constant_values=0.0)
            if ns is not None and "state" in ns:
                mean = np.asarray(ns["state"]["mean"], dtype=np.float32)
                std = np.asarray(ns["state"]["std"], dtype=np.float32)
                mean = np.pad(mean, (0, max(0, 32 - mean.shape[-1])), constant_values=0.0)
                std = np.pad(std, (0, max(0, 32 - std.shape[-1])), constant_values=1.0)
                state_norm = (state_norm - mean) / (std + 1e-6)

            outputs: dict[str, Any] = {"state": state_norm, "actions": actions_np}

            # Apply the full output transform (including Unnormalize) to match the JAX policy.
            outputs = self._output_transform(outputs)
            if return_debug_data:
                # Build a minimal debug payload that lets us compare Triton vs saved JAX runs.
                # Note: for Triton, the model may not populate a rich debug dict; we always include
                # the raw model output (`output_actions`) plus the final post-processed actions.
                def to_numpy(x):
                    if hasattr(x, "numpy"):
                        return x.numpy()
                    if hasattr(x, "__array__"):
                        return np.asarray(x)
                    return x

                debug_data_numpy: dict[str, Any] = {}
                if debug_data is not None:
                    debug_data_numpy = jax.tree.map(to_numpy, debug_data)
                debug_data_numpy["output_actions"] = np.asarray(actions_np)
                if noise is not None:
                    debug_data_numpy["noise"] = np.asarray(noise)
                debug_data_numpy["final_actions"] = outputs["actions"]
                if raw_obs is not None:
                    debug_data_numpy["raw_obs"] = raw_obs
                outputs["debug_data"] = debug_data_numpy
        else:
            inputs = self._input_transform(inputs)
            if not self._is_pytorch_model:
                # Make a batch and convert to jax.Array.
                inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
                self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
            else:
                # Convert inputs to PyTorch tensors and move to correct device
                inputs = jax.tree.map(
                    lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                    inputs,
                )
                sample_rng_or_pytorch_device = self._pytorch_device

            # Prepare kwargs for sample_actions
            sample_kwargs = dict(self._sample_kwargs)
            sample_kwargs["return_debug_data"] = return_debug_data
            if noise is not None:
                noise = (
                    torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
                )

                if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                    noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
                sample_kwargs["noise"] = noise

            observation = _model.Observation.from_dict(inputs)
            debug_data: dict | None = None

            actions, debug_data = self._sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                prev_action=prev_action,
                use_rtc=use_rtc,
                s=s_param,
                d=d_param,
                **sample_kwargs,
            )
            outputs = {
                "state": observation.state,
                "actions": actions,
            }

            # Collect data for JAX models (after JIT execution)
            if not self._is_pytorch_model and hasattr(self._model, "output_actions_save"):
                self._model.output_actions_save.append(actions)

            if self._is_pytorch_model:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

            outputs = self._output_transform(outputs)

            # Add debug data if requested
            if debug_data is not None:
                # Convert JAX arrays to numpy for serialization
                def to_numpy(x):
                    if hasattr(x, "numpy"):
                        return x.numpy()
                    if hasattr(x, "__array__"):
                        return np.asarray(x)
                    return x

                debug_data_numpy = jax.tree.map(to_numpy, debug_data)
                # save the final post-processed actions (after unnormalization)
                debug_data_numpy["final_actions"] = outputs["actions"]
                # save the raw observation (before any transforms) for det replay
                if raw_obs is not None:
                    debug_data_numpy["raw_obs"] = raw_obs
                outputs["debug_data"] = debug_data_numpy

        return outputs

    def create_batch_obs(self, observations: list[dict]) -> _model.Observation:
        # Stack observations into batch format
        batched_obs = {}

        if self._is_triton_optimized:
            return {
                "state": np.stack([obs["observation/state"] for obs in observations], axis=0),
                "base_0_rgb": np.stack([obs["observation/image"] for obs in observations], axis=0),
                "left_wrist_0_rgb": np.stack([obs["observation/wrist_image"] for obs in observations], axis=0),
                "right_wrist_0_rgb": np.stack([obs["observation/wrist_image"] for obs in observations], axis=0),
                "prompt": np.stack([obs["prompt"] for obs in observations], axis=0),
            }

        # FIXME: don't hardcode these values
        keys = (
            "observation/state",
            "observation/image",
            "observation/wrist_image",
            "prompt",
        )
        for key in keys:
            # Stack all values for this key
            values = [obs[key] for obs in observations]
            if isinstance(values[0], np.ndarray):
                batched_obs[key] = np.stack(values, axis=0)
            elif isinstance(values[0], dict):
                # Handle nested dictionaries (like images)
                batched_obs[key] = {}
                for subkey in values[0]:
                    subvalues = [obs[key][subkey] for obs in observations]
                    if isinstance(subvalues[0], np.ndarray):
                        batched_obs[key][subkey] = np.stack(subvalues, axis=0)
                    else:
                        batched_obs[key][subkey] = subvalues
            else:
                batched_obs[key] = values

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, batched_obs)
        # Apply transforms to batched observation
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Convert to jax.Array (already batched)
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)

        return _model.Observation.from_dict(inputs)

    def infer_batch(
        self,
        requests: list[InferRequest],
        *,
        noise: np.ndarray | None = None,
        return_debug_data: bool = False,
    ) -> list[InferResponse]:
        """Run inference on a batch of request.

        Args:
            obs_batch: List of InferRequest objects of the same infer_type.
            noise: Optional noise tensor for batch (shape: batch_size, action_horizon, action_dim)
            return_debug_data: Whether to return debug data (obs before/after preprocessing, noise, output)

        Returns:
            List of InferResponse objects, one for each input request.
        """
        if not requests:
            return []

        # Check if any request wants debug data (use per-request flag if available)
        any_debug_data = return_debug_data or any(getattr(req, "return_debug_data", False) for req in requests)

        # Capture raw observations before any transforms (for deterministic replay)
        raw_obs_list = None
        if any_debug_data:
            raw_obs_list = [
                jax.tree.map(
                    lambda x: np.array(x) if hasattr(x, "__array__") else x,
                    req.observation,
                )
                for req in requests
            ]

        # TODO: separate logic for jax and pytorch?
        if not self._is_pytorch_model:
            # Convert to jax.Array (already batched)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            sample_rng_or_pytorch_device = self._pytorch_device

        observation = self.create_batch_obs([req.observation for req in requests])

        if self._is_triton_optimized:
            # Batched Triton inference path - TODO Rohan: can be squashed into Jax batch path once below TODO is resolved

            # Prepare kwargs for sample_actions
            sample_kwargs = dict(self._sample_kwargs)
            sample_kwargs["return_debug_data"] = any_debug_data

            # Handle batched noise if provided
            if noise is not None:
                sample_kwargs["noise"] = np.asarray(noise)

            # TODO Rohan: return state_norm since Triton kernels bypass input_transform for internal method. Figure out why input_transform doesn't work
            actions, state_norm, debug_data = self._sample_actions(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            )

            # Convert actions to numpy
            actions_np = np.asarray(actions)

            # FIXME: I don't think the code below returns proper InferResponses?
            # Process each batch element
            results: list[InferResponse] = []
            for i in range(len(requests)):
                req = requests[i]

                # Extract actions for this batch element
                action_i = actions_np[i]

                # Extract normalized state for this batch element
                state_norm_i = state_norm[i]

                result: dict[str, Any] = {"state": state_norm_i, "actions": action_i}

                # Apply the full output transform (including Unnormalize)
                result = self._output_transform(result)

                # Add debug data if requested
                if any_debug_data or getattr(req, "return_debug_data", False):
                    debug_np: dict[str, Any] = {}
                    if debug_data is not None:
                        # Extract debug data for this batch element
                        for debug_key, debug_value in debug_data.items():
                            if isinstance(debug_value, dict):
                                debug_np[debug_key] = {}
                                for subkey, subvalue in debug_value.items():
                                    if isinstance(subvalue, np.ndarray) and len(subvalue.shape) > 0:
                                        debug_np[debug_key][subkey] = (
                                            subvalue[i] if subvalue.shape[0] == len(requests) else subvalue
                                        )
                                    else:
                                        debug_np[debug_key][subkey] = subvalue
                            elif isinstance(debug_value, np.ndarray) and len(debug_value.shape) > 0:
                                debug_np[debug_key] = (
                                    debug_value[i] if debug_value.shape[0] == len(requests) else debug_value
                                )
                            else:
                                debug_np[debug_key] = debug_value

                    debug_np["output_actions"] = action_i

                    # Handle per-request noise
                    req_noise = getattr(req, "noise", None)
                    if req_noise is None and noise is not None:
                        req_noise = noise[i] if noise.ndim == 3 else noise
                    if req_noise is not None:
                        debug_np["noise"] = np.asarray(req_noise)

                    debug_np["final_actions"] = result["actions"]
                    if raw_obs_list is not None:
                        debug_np["raw_obs"] = raw_obs_list[i]
                    result["debug_data"] = debug_np

                results.append(result)

            return results
        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs["return_debug_data"] = any_debug_data
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            sample_kwargs["noise"] = noise

        actions, debug_data = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": observation.state,
            "actions": actions,
        }

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)

        outputs = self._output_transform(outputs)

        # Split batch results back into individual results
        results = []
        for i in range(len(requests)):
            result = {}
            for key, value in outputs.items():
                if isinstance(value, np.ndarray) and len(value.shape) > 0:
                    result[key] = value[i]
                else:
                    result[key] = value

            # Add debug data for this batch element if available
            if debug_data is not None:

                def to_numpy(x):
                    if hasattr(x, "numpy"):
                        return x.numpy()
                    if hasattr(x, "__array__"):
                        return np.asarray(x)
                    return x

                # First convert all JAX arrays to numpy recursively
                debug_data_numpy = jax.tree.map(to_numpy, debug_data)

                # Extract debug data for this batch element
                result_debug = {}
                for debug_key, debug_value in debug_data_numpy.items():
                    if isinstance(debug_value, dict):
                        result_debug[debug_key] = {}
                        for subkey, subvalue in debug_value.items():
                            if isinstance(subvalue, np.ndarray) and len(subvalue.shape) > 0:
                                result_debug[debug_key][subkey] = subvalue[i]
                            else:
                                result_debug[debug_key][subkey] = subvalue
                    elif isinstance(debug_value, np.ndarray) and len(debug_value.shape) > 0:
                        result_debug[debug_key] = debug_value[i]
                    else:
                        result_debug[debug_key] = debug_value

                # Add final_actions (post-output-transform) for this batch element
                result_debug["final_actions"] = result["actions"]

                # Add raw_obs (before any transforms) for this batch element
                if raw_obs_list is not None:
                    result_debug["raw_obs"] = raw_obs_list[i]

                result["debug_data"] = result_debug

            results.append(result)

        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def make_example(self) -> dict:
        assert "env" in self._metadata, "Environment not set in metadata"
        env = EnvMode(self._metadata["env"])
        if env == EnvMode.ALOHA:
            return make_aloha_example()
        if env == EnvMode.DROID:
            return make_droid_example()
        if env in [
            EnvMode.LIBERO,
            EnvMode.LIBERO_REALTIME,
            EnvMode.LIBERO_PYTORCH,
            EnvMode.LIBERO_PI0,
        ]:
            return make_libero_example()

        raise ValueError(f"Unknown environment: {env}")

    def make_example_actions(self) -> np.ndarray:
        return self._model.make_example_actions()


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
