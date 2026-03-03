import abc
import dataclasses
import enum
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client.messages import InferRequest
from openpi_client.messages import InferType
from openpi_client.messages import RTCParams
from typing_extensions import override

from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example

BasePolicy: TypeAlias = _base_policy.BasePolicy

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LIBERO_PI0 = "libero_pi0"
    LIBERO_PYTORCH = "libero_pytorch"
    LIBERO_REALTIME = "libero_realtime"


@dataclasses.dataclass
class InferResult:
    actions: np.ndarray  # (action_horizon, action_dim), unnormalized
    noise: np.ndarray  # (action_horizon, noise_dim)
    state: np.ndarray  # (state_dim,), normalized


class PolicyBackend(abc.ABC):
    @abc.abstractmethod
    def sample_noise(self, batch_size: int = 1) -> np.ndarray:
        """Returns noise of shape (batch_size, action_horizon, noise_dim)."""
        ...

    @abc.abstractmethod
    def infer(
        self,
        obs: dict,
        noise: np.ndarray,
        *,
        use_rtc: bool = False,
        prev_action: np.ndarray | None = None,
        s: int = 5,
        d: int = 4,
    ) -> InferResult:
        """Single-sample inference. noise shape: (action_horizon, noise_dim)."""
        ...

    @abc.abstractmethod
    def infer_batch(self, observations: list[dict], batch_noise: np.ndarray) -> list[InferResult]:
        """Batched inference. batch_noise shape: (batch_size, action_horizon, noise_dim)."""
        ...

    @abc.abstractmethod
    def make_example_actions(self) -> np.ndarray:
        """Returns example actions of shape (action_horizon, action_dim)."""
        ...


def _client_obs_to_policy_obs(obs: dict) -> dict:
    """Convert client observation keys to policy observation keys."""
    return {
        "observation/state": obs["state"],
        "observation/image": obs["image"],
        "observation/wrist_image": obs["wrist_image"],
        "prompt": obs["prompt"],
    }


def _stack_observations(observations: list[dict]) -> dict:
    """Stack a list of observation dicts into a single batched dict."""
    batched: dict = {}
    for key in observations[0]:
        values = [obs[key] for obs in observations]
        if isinstance(values[0], np.ndarray):
            batched[key] = np.stack(values, axis=0)
        elif isinstance(values[0], dict):
            batched[key] = {}
            for subkey in values[0]:
                subvalues = [v[subkey] for v in values]
                if isinstance(subvalues[0], np.ndarray):
                    batched[key][subkey] = np.stack(subvalues, axis=0)
                else:
                    batched[key][subkey] = subvalues
        else:
            batched[key] = values
    return batched


class Policy(BasePolicy):
    def __init__(self, backend: PolicyBackend, *, metadata: dict[str, Any] | None = None):
        self._backend = backend
        self._metadata = metadata or {}

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
    ) -> dict:  # type: ignore[misc]
        if noise is None:
            noise = self._backend.sample_noise(batch_size=1)[0]  # (ah, ad)
        result = self._backend.infer(obs, noise, use_rtc=use_rtc, prev_action=prev_action, s=s_param, d=d_param)
        return {"actions": result.actions, "noise": result.noise, "state": result.state}

    def infer_batch(self, requests: list[InferRequest]) -> list[InferResult]:
        """Run inference on a batch of requests.

        Args:
            requests: List of InferRequest objects.

        Returns:
            List of InferResult objects, one for each input request.
        """
        if not requests:
            return []
        batch_noise = self._backend.sample_noise(batch_size=len(requests))  # (B, ah, ad)
        for i, req in enumerate(requests):
            if req.noise is not None:
                batch_noise[i] = req.noise
        observations = [_client_obs_to_policy_obs(req.observation) for req in requests]
        return self._backend.infer_batch(observations, batch_noise)

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
        return self._backend.make_example_actions()

    def make_infer_request(self) -> InferRequest:
        observation = self.make_example()
        return InferRequest(
            robot_id="test_robot",
            start_step=0,
            request_timestamp=0,
            deadline=0,
            observation=observation,
            infer_type=InferType.SYNC,
            params=None,
            noise=None,
        )

    def warmup(self, max_batch_size: int) -> None:
        """Warm up policy by running inference to trigger JIT compilation."""
        observation = self.make_example()

        requests = [
            InferRequest(
                robot_id="test_robot",
                start_step=0,
                request_timestamp=0,
                deadline=0,
                observation=observation,
                infer_type=InferType.SYNC,
                params=None,
            ),
            InferRequest(
                robot_id="test_robot",
                start_step=0,
                request_timestamp=0,
                deadline=0,
                observation=observation,
                infer_type=InferType.INFERENCE_TIME_RTC,
                params=RTCParams(prev_action=self.make_example_actions(), s_param=5, d_param=3),
            ),
        ]

        for request in requests:
            for batch_size in range(1, max_batch_size + 1):
                logger.info(f"Warming up {request.infer_type} for batch_size={batch_size}")
                # Warm up with full batch_size (we always pad to this size)
                batch = [request] * batch_size
                actions = self.infer_batch(batch)

        logger.info(f"Output size: {actions[0].actions.shape}")


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
