import importlib.util
import pathlib

_MODULE_PATH = pathlib.Path(__file__).with_name("serve_policy.py")
_SPEC = importlib.util.spec_from_file_location("serve_policy", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
serve_policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(serve_policy)


def test_build_scheduler_kwargs_for_receding_horizon_ilp_defaults():
    args = serve_policy.Args(scheduling_algorithm="receding_horizon_ilp")

    kwargs = serve_policy.build_scheduler_kwargs(args, action_horizon_steps=50)

    assert kwargs == {
        "tick_ms": 10,
        "horizon_steps": 160,
        "execution_fraction": 0.25,
        "solve_timeout_ms": 500,
        "action_horizon_steps": 50,
    }


def test_build_scheduler_kwargs_for_receding_horizon_ilp_with_override():
    args = serve_policy.Args(
        scheduling_algorithm="receding_horizon_ilp",
        ilp_action_horizon_steps=25,
        ilp_horizon_steps=200,
    )

    kwargs = serve_policy.build_scheduler_kwargs(args, action_horizon_steps=50)

    assert kwargs is not None
    assert kwargs["action_horizon_steps"] == 25
    assert kwargs["horizon_steps"] == 200


def test_build_scheduler_kwargs_for_lookahead():
    args = serve_policy.Args(
        scheduling_algorithm="lookahead",
        lookahead_horizon_ms=700,
        lookahead_timestep_ms=20,
        lookahead_control_hz=30,
    )

    kwargs = serve_policy.build_scheduler_kwargs(args, action_horizon_steps=16)

    assert kwargs == {
        "horizon_ms": 700,
        "timestep_ms": 20,
        "action_horizon_steps": 16,
        "control_hz": 30,
    }
