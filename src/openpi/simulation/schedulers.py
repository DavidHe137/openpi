from collections.abc import Callable, Generator
import itertools
import os
import random
import subprocess

from tqdm import tqdm

from .classes import CONTROL_PERIOD
from .classes import RobotState
from .classes import SimulatorParameters
from .cost_functions import constant_cost_function
from .sim import Scheduler
from .sim import Simulator


class ILPBatchedScheduler(Scheduler):
    """Optimal offline scheduler using ILP with batching (uniform step grid only)."""

    def __init__(self, simulator_parameters: SimulatorParameters):
        super().__init__(simulator_parameters)
        from solve import solve_ilp_batched

        self.plan = solve_ilp_batched(simulator_parameters)

    def select_robots(self, simulator: Simulator) -> list[int]:
        from classes import time_to_step

        return self.plan.get(time_to_step(simulator.state.time), [])


class ILPExactScheduler(Scheduler):
    """Optimal offline scheduler using ms-resolution ILP with batching."""

    def __init__(self, simulator_parameters: SimulatorParameters, warm_start: bool = False):
        super().__init__(simulator_parameters)
        from solve import solve_ilp_batched_exact

        self.plan = solve_ilp_batched_exact(simulator_parameters, warm_start=warm_start)

    def select_robots(self, simulator: Simulator) -> list[int]:
        return self.plan.get(simulator.state.time, [])


class EDFScheduler(Scheduler):
    def select_robots(self, simulator: Simulator) -> list[int]:
        """Earliest Deadline First: always batch size 1, pick robot with earliest deadline."""
        earliest_index = min(
            range(self.simulator_parameters.num_robots),
            key=lambda i: simulator.state.robots[i].deadline(),
        )
        return [earliest_index]


class GreedyScheduler(Scheduler):
    def select_robots(self, simulator: Simulator) -> list[str]:
        """
        Greedy scheduler with deadline awareness: select robots with earliest deadlines first,
        using the largest batch size that fits within the earliest deadline.
        """
        params = self.simulator_parameters

        # Sort by deadline in ms (per-robot step times)
        def deadline_ms(r: str) -> int:
            return params.step_to_time_robot(simulator.state.robots[i].deadline(), r)

        robot_ids = sorted(params.control_step_times.keys(), key=deadline_ms)
        earliest_deadline = deadline_ms(robot_ids[0])
        largest_batch_size = next(
            (i for i in sorted(params.d_infer.keys(), reverse=True) if params.d_infer[i] <= earliest_deadline),
            0,
        )
        if largest_batch_size == 0:
            largest_batch_size = 1

        return [i for i in robot_ids[:largest_batch_size]]


class RoundRobinScheduler(Scheduler):
    def __init__(self, simulator_parameters: SimulatorParameters):
        super().__init__(simulator_parameters)
        self.robot_index = 0

    def select_robots(self, simulator: Simulator) -> list[int]:
        """Round-robin scheduler: cycle through robots one at a time."""
        robot_index = self.robot_index
        self.robot_index = (self.robot_index + 1) % self.simulator_parameters.num_robots
        return [robot_index]


class RandomScheduler(Scheduler):
    def select_robots(self, simulator: Simulator) -> list[int]:
        """Random scheduler: select a random robot."""
        robot_index = random.randint(0, self.simulator_parameters.num_robots - 1)
        return [robot_index]


class RandomBatchScheduler(Scheduler):
    def select_robots(self, simulator: Simulator) -> list[int]:
        """Random batch scheduler: select a random batch size and random robots."""
        # Determine valid batch sizes based on d_infer configuration
        max_batch_size = min(
            self.simulator_parameters.num_robots,
            len(self.simulator_parameters.d_infer) - 1,
        )

        # Randomly select a batch size (1 to max_batch_size)
        batch_size = random.randint(1, max_batch_size)

        # Randomly select robots without replacement
        robot_indices = random.sample(range(self.simulator_parameters.num_robots), batch_size)

        return robot_indices


class LookaheadScheduler(Scheduler):
    """Search over all possible batch configurations."""

    def __init__(
        self,
        simulator_parameters: SimulatorParameters,
        cost_function: Callable[[RobotState, int], float] = constant_cost_function,
    ):
        super().__init__(simulator_parameters)
        self.cost_function = cost_function
        self.schedule = iter(self.search())

    def _generate_candidates(
        self,
        simulator: Simulator,
    ) -> Generator[list[int], None, None]:
        """Generate all possible batch configurations."""
        params = self.simulator_parameters
        num_robots = params.num_robots
        max_batch_size = min(num_robots, len(params.d_infer) - 1)

        # Sort by deadline in ms for correct cross-robot ordering
        def deadline_ms(i: int) -> int:
            return params.step_to_time_robot(simulator.state.robots[i].deadline(), i)

        # Exclude robots that were just served
        last_served = set(simulator.batch_history[-1].robot_indices) if simulator.batch_history else set()
        sorted_indices = [i for i in sorted(range(num_robots), key=deadline_ms) if i not in last_served]

        for batch_size in reversed(range(1, min(max_batch_size, len(sorted_indices)) + 1)):
            yield sorted_indices[:batch_size]

    def _lower_bound(self, simulator: Simulator) -> int:
        """Compute a lower bound on total starvation including inevitable future starvation."""
        params = self.simulator_parameters
        current_starvation = simulator.calculate_starvation()
        for i, robot in enumerate(simulator.state.robots):
            deadline = params.step_to_time_robot(robot.deadline(), i)
            # Earliest a new chunk could possibly arrive for this robot
            min_delivery = simulator.state.server_available_at + params.d_infer[1] + params.d_recv[i]
            if min_delivery > deadline:
                starve_steps = (min_delivery - deadline) // CONTROL_PERIOD
                current_starvation += starve_steps
        return current_starvation

    def search(self) -> list[list[str]]:
        """Search over all possible trajectories."""
        simulator = Simulator(self.simulator_parameters)

        # Seed best_starvation with greedy solution for tighter initial bound
        greedy_sim = Simulator(self.simulator_parameters)
        greedy = GreedyScheduler(self.simulator_parameters)
        greedy_sim.run(greedy)
        best_starvation = greedy_sim.calculate_starvation()
        best_trajectory = []
        # Reconstruct greedy trajectory from batch history
        for batch in greedy_sim.batch_history:
            best_trajectory.append(batch.robot_indices)
        print(f"Greedy seed: starvation={best_starvation}")

        pbar = tqdm(desc="Leaf nodes evaluated", unit="leaves")

        def dfs(current_trajectory):
            nonlocal best_starvation, best_trajectory
            current_time = simulator.state.time
            if current_time > simulator.params.end_time:
                # Advance all robots to end_time for consistent starvation window
                saved_lengths = [len(r.action_index) for r in simulator.state.robots]
                for i, robot in enumerate(simulator.state.robots):
                    robot.advance_to(self.simulator_parameters.time_to_step_robot(simulator.params.end_time - 1, i))
                starvation = simulator.calculate_starvation()
                # Restore action_index
                for i, robot in enumerate(simulator.state.robots):
                    del robot.action_index[saved_lengths[i] :]
                pbar.set_postfix(best=best_starvation, cur=starvation)
                pbar.update(1)
                if starvation < best_starvation:
                    best_starvation = starvation
                    best_trajectory = list(current_trajectory)  # copy
                return

            candidates = self._generate_candidates(simulator)
            for candidate in candidates:
                current_trajectory.append(candidate)
                simulator.step(candidate)
                # Prune: lower bound includes inevitable future starvation
                if self._lower_bound(simulator) < best_starvation:
                    dfs(current_trajectory)
                simulator.undo()
                current_trajectory.pop()

        dfs([])
        pbar.close()
        print(best_starvation, best_trajectory)

        return best_trajectory

    def select_robots(self, simulator: Simulator) -> list[int]:
        return next(self.schedule, [])


class FixedScheduler(Scheduler):
    """Replays a user-provided schedule. Loops through the schedule."""

    def __init__(self, simulator_parameters: SimulatorParameters, schedule: list[list[int]]):
        super().__init__(simulator_parameters)
        self._schedule = itertools.cycle(schedule)

    def select_robots(self, simulator: Simulator) -> list[int]:
        return next(self._schedule, [])


class CppLookaheadScheduler(Scheduler):
    """LookaheadScheduler backed by compiled C++ for speed."""

    _cpp_binary = os.path.join(os.path.dirname(__file__), "search")

    def __init__(self, simulator_parameters: SimulatorParameters):
        super().__init__(simulator_parameters)
        self.schedule = iter(self._run_cpp(simulator_parameters))

    @classmethod
    def _run_cpp(cls, p: SimulatorParameters) -> list[list[int]]:
        args = [
            cls._cpp_binary,
            str(p.end_time),
            ",".join(map(str, p.d_infer)),
            str(p.num_robots),
            ",".join(map(str, p.d_send)),
            ",".join(map(str, p.d_recv)),
            ",".join(map(str, p.execution_horizon)),
        ]
        import sys

        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError("C++ search failed")
        lines = result.stdout.strip().split("\n")
        starvation = int(lines[0])
        num_batches = int(lines[1])
        schedule = []
        for i in range(num_batches):
            schedule.append([int(x) for x in lines[2 + i].split()])
        print(f"C++ search: starvation={starvation}, schedule length={num_batches}")
        return schedule

    def select_robots(self, simulator: Simulator) -> list[int]:
        return next(self.schedule, [])
