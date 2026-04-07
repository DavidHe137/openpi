from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from openpi_client.messages import InferRequest
from openpi_client.messages import InferType
from openpi_client.messages import RTCParams

from openpi.policies.policy import Policy


def _make_request(
    *,
    infer_type: InferType,
    params: RTCParams | None = None,
) -> InferRequest:
    obs = {
        "state": np.zeros(7, dtype=np.float32),
        "image": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "prompt": "",
    }
    return InferRequest(
        robot_id="robot_0",
        observation=obs,
        observation_step=0,
        action_start_step=0,
        request_timestamp=0.0,
        deadline=0.0,
        execution_horizon=0,
        infer_type=infer_type,
        params=params,
        noise=None,
    )


def test_infer_batch_honors_rtc_params_and_preserves_model_space_actions():
    policy = Policy.__new__(Policy)
    policy._is_pytorch_model = False
    policy._is_triton_optimized = False
    policy._rng = jax.random.key(0)
    policy._sample_kwargs = {}
    policy._output_transform = lambda outputs: outputs

    class _FakeModel:
        def sample_noise(self, rng, batch_size: int = 1):
            return jnp.zeros((batch_size, 2, 3), dtype=jnp.float32)

    calls: list[dict] = []

    def fake_sample_actions(_rng, observation, **kwargs):
        calls.append(kwargs)
        batch_size = observation.state.shape[0]
        if kwargs.get("use_rtc", False):
            prev_action = kwargs["prev_action"]
            s = kwargs["s"][:, None, None]
            d = kwargs["d"][:, None, None]
            return prev_action + s + d
        return jnp.full((batch_size, 2, 3), -1.0, dtype=jnp.float32)

    policy._model = _FakeModel()
    policy._sample_actions = fake_sample_actions
    policy.create_batch_obs = lambda observations: SimpleNamespace(
        state=jnp.zeros((len(observations), 1), dtype=jnp.float32)
    )

    rtc_prev_1 = np.ones((2, 3), dtype=np.float32)
    rtc_prev_2 = np.full((2, 3), 2.0, dtype=np.float32)
    requests = [
        _make_request(infer_type=InferType.SYNC),
        _make_request(
            infer_type=InferType.INFERENCE_TIME_RTC,
            params=RTCParams(prev_action=rtc_prev_1, s_param=2, d_param=1),
        ),
        _make_request(
            infer_type=InferType.INFERENCE_TIME_RTC,
            params=RTCParams(prev_action=rtc_prev_2, s_param=4, d_param=3),
        ),
    ]

    results = policy.infer_batch(requests)

    assert len(calls) == 2
    assert calls[0].get("use_rtc", False) is False
    assert calls[1]["use_rtc"] is True
    np.testing.assert_array_equal(np.asarray(calls[1]["prev_action"]), np.stack([rtc_prev_1, rtc_prev_2], axis=0))
    np.testing.assert_array_equal(np.asarray(calls[1]["s"]), np.array([2, 4], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(calls[1]["d"]), np.array([1, 3], dtype=np.int32))

    expected_sync = np.full((2, 3), -1.0, dtype=np.float32)
    expected_rtc_1 = rtc_prev_1 + 3.0
    expected_rtc_2 = rtc_prev_2 + 7.0

    np.testing.assert_array_equal(results[0]["actions"], expected_sync)
    np.testing.assert_array_equal(results[0]["rtc_prev_actions"], expected_sync)
    np.testing.assert_array_equal(results[1]["actions"], expected_rtc_1)
    np.testing.assert_array_equal(results[1]["rtc_prev_actions"], expected_rtc_1)
    np.testing.assert_array_equal(results[2]["actions"], expected_rtc_2)
    np.testing.assert_array_equal(results[2]["rtc_prev_actions"], expected_rtc_2)
