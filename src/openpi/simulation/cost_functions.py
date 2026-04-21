from .classes import RobotState


def constant_cost_function(robot: RobotState, step: int) -> float:
    return float(1 - (robot.actions_left(step) > 0))


def observation_staleness_cost_function(robot: RobotState, step: int) -> float:
    return step if not robot.chunks else step - (robot.chunks[-1].infer_step - robot.chunks[-1].d_send)
