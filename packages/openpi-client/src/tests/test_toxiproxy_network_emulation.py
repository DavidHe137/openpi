import json

import pytest

from openpi_client.network_emulation.toxiproxy import NetworkEmulationConfigError
from openpi_client.network_emulation.toxiproxy import RobotNetworkHook
from openpi_client.network_emulation.toxiproxy import load_network_emulation_config


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


def test_load_network_config_applies_defaults(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "robots": {
                    "robot_0": {"rtt_median_ms": 25.0, "rtt_sigma": 0.2},
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = load_network_emulation_config(path)
    assert cfg["toxiproxy"]["api_url"] == "http://127.0.0.1:8474"
    assert cfg["toxiproxy"]["listen_host"] == "127.0.0.1"
    assert cfg["toxiproxy"]["listen_port_base"] == 18080
    assert cfg["sampling"]["default_seed"] == 0
    assert cfg["sampling"]["resample_every_requests"] == 1


def test_network_config_rejects_legacy_mean_std_fields(tmp_path) -> None:
    path = tmp_path / "bad_config.json"
    data = _minimal_config()
    data["robots"]["robot_0"] = {"rtt_mean_ms": 25.0, "rtt_std_ms": 3.0}
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(NetworkEmulationConfigError, match="rtt_median_ms and rtt_sigma"):
        load_network_emulation_config(path)


def test_hook_sampling_is_deterministic_for_fixed_seed(monkeypatch, tmp_path) -> None:
    class FakeToxiproxyController:
        def __init__(self, api_url, **kwargs):
            self.calls = []

        def set_latency(self, proxy_name, upstream_ms, downstream_ms):
            self.calls.append((proxy_name, upstream_ms, downstream_ms))

    import openpi_client.network_emulation.toxiproxy as toxiproxy_module

    monkeypatch.setattr(toxiproxy_module, "ToxiproxyController", FakeToxiproxyController)

    context = {
        "robot_id": "robot_0",
        "proxy_name": "p0",
        "proxy_host": "127.0.0.1",
        "proxy_port": 18080,
        "api_url": "http://127.0.0.1:8474",
        "rtt_median_ms": 100.0,
        "rtt_sigma": 0.35,
        "seed": 42,
        "resample_every_requests": 1,
        "trace_path": str(tmp_path / "robot_0_latency_trace.jsonl"),
    }

    hook_a = RobotNetworkHook(dict(context))
    hook_b = RobotNetworkHook(dict(context, trace_path=str(tmp_path / "robot_0_latency_trace_b.jsonl")))

    for _ in range(5):
        hook_a.before_send()
        hook_b.before_send()

    seq_a = [entry["sampled_rtt_ms"] for entry in hook_a._trace]  # noqa: SLF001
    seq_b = [entry["sampled_rtt_ms"] for entry in hook_b._trace]  # noqa: SLF001
    assert seq_a == pytest.approx(seq_b, rel=0.0, abs=1e-12)


def test_robot_hook_resample_cadence_and_trace_schema(monkeypatch, tmp_path) -> None:
    class FakeToxiproxyController:
        instances = []

        def __init__(self, api_url, **kwargs):
            self.calls = []
            FakeToxiproxyController.instances.append(self)

        def set_latency(self, proxy_name, upstream_ms, downstream_ms):
            self.calls.append((proxy_name, upstream_ms, downstream_ms))

    import openpi_client.network_emulation.toxiproxy as toxiproxy_module

    monkeypatch.setattr(toxiproxy_module, "ToxiproxyController", FakeToxiproxyController)

    context = {
        "robot_id": "robot_0",
        "proxy_name": "p0",
        "proxy_host": "127.0.0.1",
        "proxy_port": 18080,
        "api_url": "http://127.0.0.1:8474",
        "rtt_median_ms": 100.0,
        "rtt_sigma": 0.0,
        "seed": 7,
        "resample_every_requests": 2,
        "trace_path": str(tmp_path / "robot_0_latency_trace.jsonl"),
    }
    hook = RobotNetworkHook(context)
    hook.before_send()  # resample
    hook.before_send()  # reuse
    hook.before_send()  # resample
    hook.flush_trace()

    controller = FakeToxiproxyController.instances[0]
    assert len(controller.calls) == 2  # one set_latency per resample
    assert [entry["resampled"] for entry in hook._trace] == [True, False, True]  # noqa: SLF001

    lines = (tmp_path / "robot_0_latency_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first_entry = json.loads(lines[0])
    assert "sampled_rtt_ms" in first_entry
    assert "upstream_latency_ms" in first_entry
    assert "downstream_latency_ms" in first_entry
    assert "clipped" not in first_entry
    assert "raw_sampled_rtt_ms" not in first_entry
