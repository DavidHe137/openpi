import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any
from typing import Dict

import pytest

from openpi_client.network_emulation import load_experiment_config


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "examples" / "libero" / "generate_experiment_config.py"

_SPEC = importlib.util.spec_from_file_location("generate_experiment_config", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _homogeneous_profile() -> Dict[str, Any]:
    return {
        "uplink_median_ms": 8.0,
        "uplink_sigma": 0.2,
        "downlink_median_ms": 9.0,
        "downlink_sigma": 0.25,
        "execution_horizon": 10,
    }


def _max_profile() -> Dict[str, Any]:
    return _MODULE._build_max_profile(  # noqa: SLF001
        _homogeneous_profile(),
        max_uplink_median_ms=80.0,
        max_uplink_sigma=0.8,
        max_downlink_median_ms=90.0,
        max_downlink_sigma=0.9,
        max_execution_horizon=16,
    )


def _build_config(*, k: float, num_robots: int = 5) -> Dict[str, Any]:
    return _MODULE.build_output_config(
        num_robots=num_robots,
        homogeneous_profile=_homogeneous_profile(),
        max_profile=_max_profile(),
        heterogeneity_k=k,
        action_chunk_broker_type="rtc",
        trials_per_robot=3,
        toxiproxy_api_url="http://127.0.0.1:8474",
        toxiproxy_listen_host="127.0.0.1",
        toxiproxy_listen_port_base=15000,
        toxiproxy_server_args=[],
        sampling_default_seed=7,
        sampling_resample_every_requests=1,
    )


def test_k_zero_uses_homogeneous_profile_for_all_robots() -> None:
    cfg = _build_config(k=0.0, num_robots=4)
    homo = _homogeneous_profile()

    for i in range(4):
        robot = cfg["robots"][f"robot_{i}"]
        assert robot["uplink_median_ms"] == pytest.approx(homo["uplink_median_ms"])
        assert robot["uplink_sigma"] == pytest.approx(homo["uplink_sigma"])
        assert robot["downlink_median_ms"] == pytest.approx(homo["downlink_median_ms"])
        assert robot["downlink_sigma"] == pytest.approx(homo["downlink_sigma"])
        assert robot["execution_horizon"] == homo["execution_horizon"]
        assert robot["seed"] == 7 + i


def test_k_one_last_robot_reaches_max_and_first_stays_homogeneous() -> None:
    cfg = _build_config(k=1.0, num_robots=5)
    homo = _homogeneous_profile()
    max_profile = _max_profile()

    first = cfg["robots"]["robot_0"]
    assert first["uplink_median_ms"] == pytest.approx(homo["uplink_median_ms"])
    assert first["uplink_sigma"] == pytest.approx(homo["uplink_sigma"])
    assert first["downlink_median_ms"] == pytest.approx(homo["downlink_median_ms"])
    assert first["downlink_sigma"] == pytest.approx(homo["downlink_sigma"])
    assert first["execution_horizon"] == homo["execution_horizon"]

    last = cfg["robots"]["robot_4"]
    assert last["uplink_median_ms"] == pytest.approx(max_profile["uplink_median_ms"])
    assert last["uplink_sigma"] == pytest.approx(max_profile["uplink_sigma"])
    assert last["downlink_median_ms"] == pytest.approx(max_profile["downlink_median_ms"])
    assert last["downlink_sigma"] == pytest.approx(max_profile["downlink_sigma"])
    assert last["execution_horizon"] == max_profile["execution_horizon"]

    # Monotonic with robot index for this configuration.
    for field in ("uplink_median_ms", "uplink_sigma", "downlink_median_ms", "downlink_sigma", "execution_horizon"):
        values = [cfg["robots"][f"robot_{i}"][field] for i in range(5)]
        assert values == sorted(values)


def test_mid_k_interpolates_between_homogeneous_and_generated_max() -> None:
    cfg = _build_config(k=0.5, num_robots=3)

    # For 3 robots, robot_2 has position=1.0 and robot_1 has position=0.5.
    robot_2 = cfg["robots"]["robot_2"]
    assert robot_2["uplink_median_ms"] == pytest.approx(44.0)  # 8 + 0.5 * (80 - 8)
    assert robot_2["downlink_median_ms"] == pytest.approx(49.5)  # 9 + 0.5 * (90 - 9)
    assert robot_2["execution_horizon"] == 13  # round(10 + 0.5 * (16 - 10))

    robot_1 = cfg["robots"]["robot_1"]
    assert robot_1["uplink_median_ms"] == pytest.approx(26.0)  # 8 + 0.5 * (0.5 * (80 - 8))
    assert robot_1["downlink_median_ms"] == pytest.approx(29.25)
    assert robot_1["execution_horizon"] == 12  # round(10 + 0.5 * (0.5 * 6))


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="heterogeneity-k"):
        _build_config(k=1.5)


def test_max_profile_rejects_values_below_homogeneous() -> None:
    with pytest.raises(ValueError, match="must be >= homogeneous value"):
        _MODULE._build_max_profile(  # noqa: SLF001
            _homogeneous_profile(),
            max_uplink_median_ms=7.0,
            max_uplink_sigma=0.8,
            max_downlink_median_ms=90.0,
            max_downlink_sigma=0.9,
            max_execution_horizon=16,
        )


def test_cli_generates_valid_config(tmp_path: pathlib.Path) -> None:
    profile_path = tmp_path / "homo_profile.jsonc"
    output_path = tmp_path / "generated.jsonc"
    _write_json(profile_path, _homogeneous_profile())

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--num-robots",
            "4",
            "--homogeneous-robot-profile",
            str(profile_path),
            "--max-uplink-median-ms",
            "80",
            "--max-uplink-sigma",
            "0.8",
            "--max-downlink-median-ms",
            "90",
            "--max-downlink-sigma",
            "0.9",
            "--max-execution-horizon",
            "16",
            "--heterogeneity-k",
            "0.6",
            "--output-config",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    generated = load_experiment_config(output_path)
    assert generated["experiment"]["num_robots"] == 4
    assert set(generated["robots"].keys()) == {"robot_0", "robot_1", "robot_2", "robot_3"}
