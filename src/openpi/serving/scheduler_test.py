from openpi.scheduling.receding_horizon_ilp import RecedingHorizonILPScheduler
from openpi.serving.scheduler import _SCHEDULER_REGISTRY


def test_scheduler_registry_contains_receding_horizon_ilp():
    assert _SCHEDULER_REGISTRY["receding_horizon_ilp"] is RecedingHorizonILPScheduler
