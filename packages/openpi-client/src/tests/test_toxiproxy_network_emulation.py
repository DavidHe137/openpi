import json

import pytest

from openpi_client.network_emulation.toxiproxy import LogNormalRttSampler
from openpi_client.network_emulation.toxiproxy import NetworkEmulationConfig
from openpi_client.network_emulation.toxiproxy import NetworkEmulationConfigError
from openpi_client.network_emulation.toxiproxy import RobotNetworkHook
from openpi_client.network_emulation.toxiproxy import WorkerNetworkContext


def _minimal_config() -> dict:
    return {
        "toxiproxy": {
            "api_url": "http://127.0.0.1:8474",
            "listen_host": "127.0.0.1",
            "listen_port_base": 18080,
        },
        "sampling": {
            "default_seed": 7,
            "resample_every_requests": 1,
        },
        "robots": {
            "robot_0": {"rtt_median_ms": 25.0, "rtt_sigma": 0.2},
        },
    }


def test_log_normal_sampler_is_deterministic_for_fixed_seed() -> None:
    sampler_a = LogNormalRttSampler(median_ms=100.0, sigma=0.35, seed=42)
    sampler_b = LogNormalRttSampler(median_ms=100.0, sigma=0.35, seed=42)

    seq_a = [sampler_a.sample() for _ in range(5)]
    seq_b = [sampler_b.sample() for _ in range(5)]
    assert seq_a == pytest.approx(seq_b, rel=0.0, abs=1e-12)


def test_log_normal_sampler_sigma_zero_returns_median() -> None:
    sampler = LogNormalRttSampler(median_ms=83.5, sigma=0.0, seed=123)
    assert [sampler.sample() for _ in range(4)] == [83.5, 83.5, 83.5, 83.5]


def test_network_config_rejects_legacy_mean_std_fields() -> None:
    data = _minimal_config()
    data["robots"]["robot_0"] = {"rtt_mean_ms": 25.0, "rtt_std_ms": 3.0}
    with pytest.raises(NetworkEmulationConfigError, match="rtt_median_ms and rtt_sigma"):
        NetworkEmulationConfig.from_dict(data)


def test_robot_hook_resample_cadence_and_trace_schema(monkeypatch, tmp_path) -> None:
    class FakeToxiproxyHttpClient:
        instances = []

        def __init__(self, api_url, *, session=None, timeout_s=2.0):
            self.calls = []
            FakeToxiproxyHttpClient.instances.append(self)

        def upsert_latency_toxic(self, proxy_name, toxic_name, stream, latency_ms, **kwargs):
            self.calls.append((proxy_name, toxic_name, stream, latency_ms))

    import openpi_client.network_emulation.toxiproxy as toxiproxy_module

    monkeypatch.setattr(toxiproxy_module, "ToxiproxyHttpClient", FakeToxiproxyHttpClient)

    context = WorkerNetworkContext(
        robot_id="robot_0",
        proxy_name="p0",
        proxy_host="127.0.0.1",
        proxy_port=18080,
        api_url="http://127.0.0.1:8474",
        rtt_median_ms=100.0,
        rtt_sigma=0.0,
        seed=7,
        resample_every_requests=2,
        trace_path=str(tmp_path / "robot_0_latency_trace.jsonl"),
    )
    hook = RobotNetworkHook(context)
    hook.before_send()  # resample
    hook.before_send()  # reuse
    hook.before_send()  # resample
    hook.flush_trace()

    client = FakeToxiproxyHttpClient.instances[0]
    assert len(client.calls) == 4  # two toxics per resample, two resamples
    assert [entry.resampled for entry in hook._trace] == [True, False, True]  # noqa: SLF001

    lines = (tmp_path / "robot_0_latency_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first_entry = json.loads(lines[0])
    assert "sampled_rtt_ms" in first_entry
    assert "upstream_latency_ms" in first_entry
    assert "downstream_latency_ms" in first_entry
    assert "clipped" not in first_entry
    assert "raw_sampled_rtt_ms" not in first_entry
