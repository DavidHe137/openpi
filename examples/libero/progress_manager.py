from __future__ import annotations

from contextlib import nullcontext
import multiprocessing
import queue as queue_module
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


@dataclass
class RobotState:
    """Track state for a single robot."""

    robot_idx: int
    task_id: int
    task_suite_name: str

    # Episode tracking
    current_episode: int = 0
    total_episodes: int = 0

    # Step tracking
    current_step: int = 0
    max_steps: int = 300

    # Success tracking
    successes: int = 0

    # Progress bar IDs (Rich TaskIDs)
    episode_bar_id: Optional[TaskID] = None
    step_bar_id: Optional[TaskID] = None

    # Step tracking
    steps_per_sec: float = 0.0

    # Status
    active: bool = True
    completed: bool = False


@dataclass
class JobStats:
    """Aggregate stats for current job."""

    total_jobs: int
    completed_jobs: int = 0
    total_episodes: int = 0
    completed_episodes: int = 0
    total_successes: int = 0
    start_time: float = 0.0


class ProgressManager:
    """
    Context manager that handles multi-robot progress display.

    Architecture:
    - Main process creates this manager as a context manager
    - Queue is shared with worker processes
    - Background monitoring thread reads queue and updates Rich Progress
    - Uses Rich Live display with custom table layout
    """

    def __init__(
        self,
        num_robots: int,
        total_jobs: int,
        max_steps: int = 300,
        update_interval: float = 0.1,
    ):
        """
        Initialize the progress manager.

        Args:
            num_robots: Number of robot workers
            total_jobs: Total number of jobs to complete
            max_steps: Maximum steps per episode
            update_interval: How often to check queue (seconds)
        """
        self.num_robots = num_robots
        self.total_jobs = total_jobs
        self.max_steps = max_steps
        self.update_interval = update_interval

        # Cross-process communication
        self.queue: multiprocessing.Queue = multiprocessing.Queue()

        # State tracking
        self.robot_states: dict[int, RobotState] = {}
        self.job_stats = JobStats(total_jobs=total_jobs)

        # Rich Progress components
        self.progress: Optional[Progress] = None
        self.live: Optional[Live] = None
        self.console = Console()

        # Overall job progress bar ID
        self.overall_bar_id: Optional[TaskID] = None

        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def __enter__(self) -> ProgressManager:
        """Initialize Rich Progress and start monitoring thread."""
        # Create Rich Progress with custom columns
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
        )

        # Create overall progress bar
        self.overall_bar_id = self.progress.add_task(
            "[bold green]Overall Progress",
            total=self.total_jobs,
        )

        # Start Rich Live display
        self.live = Live(
            self._generate_display(),
            console=self.console,
            refresh_per_second=4,
        )
        self.live.start()

        # Start monitoring thread
        self.job_stats.start_time = time.time()
        self._monitor_thread = threading.Thread(
            target=self._monitor_queue,
            daemon=True,
        )
        self._monitor_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean shutdown: stop thread, drain queue, close display."""
        # Signal stop
        self._stop_event.set()

        # Wait for monitor thread to finish
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Drain remaining messages
        self._drain_queue()

        # Stop live display
        if self.live:
            self.live.stop()

        # Print final summary
        self._print_final_summary()

        return False

    def _generate_display(self):
        """Generate the Rich display."""
        # Use Progress object directly with stats table below
        layout = Table.grid(padding=(0, 1))
        layout.add_column(justify="left", ratio=1)

        # Add progress bars
        if self.progress:
            layout.add_row(self.progress)

        # Add stats summary
        with self._lock:
            elapsed = time.time() - self.job_stats.start_time
            success_rate = (
                self.job_stats.total_successes / self.job_stats.completed_episodes * 100
                if self.job_stats.completed_episodes > 0
                else 0.0
            )

            stats_text = (
                f"\n[bold]Jobs:[/bold] {self.job_stats.completed_jobs}/{self.job_stats.total_jobs}  "
                f"[bold]Episodes:[/bold] {self.job_stats.completed_episodes}/{self.job_stats.total_episodes}  "
                f"[bold]Success Rate:[/bold] {success_rate:.1f}%  "
                f"[bold]Time:[/bold] {elapsed:.1f}s\n"
            )

            # Add per-robot stats
            active_robots = sorted(
                [
                    rs
                    for rs in self.robot_states.values()
                    if rs.active and not rs.completed
                ],
                key=lambda rs: rs.robot_idx,
            )

            for robot_state in active_robots:
                robot_success_rate = (
                    robot_state.successes / robot_state.current_episode * 100
                    if robot_state.current_episode > 0
                    else 0.0
                )
                stats_text += (
                    f"[cyan]R{robot_state.robot_idx}[/cyan] Task {robot_state.task_id}: "
                    f"Ep {robot_state.current_episode}/{robot_state.total_episodes} "
                    f"Step {robot_state.current_step}/{robot_state.max_steps} | "
                    f"Success: {robot_state.successes}/{robot_state.current_episode} ({robot_success_rate:.0f}%) | "
                    f"Speed: {robot_state.steps_per_sec:.1f} steps/s\n"
                )

            layout.add_row(stats_text)

        return layout

    def _monitor_queue(self):
        """Background thread that monitors the queue and updates state."""
        while not self._stop_event.is_set():
            try:
                # Non-blocking queue check
                try:
                    message = self.queue.get(timeout=self.update_interval)
                    self._handle_message(message)
                except queue_module.Empty:
                    pass

                # Update display
                if self.live and self.progress:
                    self.live.update(self._generate_display())

            except Exception as e:
                self.console.print(f"[red]Error in monitor thread: {e}[/red]")

    def _drain_queue(self):
        """Drain any remaining messages in the queue."""
        while not self.queue.empty():
            try:
                message = self.queue.get_nowait()
                self._handle_message(message)
            except Exception:
                break

    def _handle_message(self, message: dict):
        """Process a single message from the queue."""
        msg_type = message["type"]

        with self._lock:
            if msg_type == "worker_init":
                self._handle_worker_init(message)
            elif msg_type == "episode_start":
                self._handle_episode_start(message)
            elif msg_type == "episode_end":
                self._handle_episode_end(message)
            elif msg_type == "step_batch":
                self._handle_step_batch(message)
            elif msg_type == "worker_complete":
                self._handle_worker_complete(message)

    def _handle_worker_init(self, message: dict):
        """Handle worker initialization message."""
        robot_idx = message["robot_idx"]
        job_info = message["job_info"]

        # Create robot state
        robot_state = RobotState(
            robot_idx=robot_idx,
            task_id=job_info["task_id"],
            task_suite_name=job_info["task_suite_name"],
            total_episodes=job_info["num_episodes"],
            max_steps=self.max_steps,
            active=True,
        )

        # Create progress bars for this robot
        if self.progress:
            robot_state.episode_bar_id = self.progress.add_task(
                f"[cyan]Robot {robot_idx}[/cyan] - Episodes",
                total=job_info["num_episodes"],
                visible=True,
            )

            robot_state.step_bar_id = self.progress.add_task(
                f"[cyan]Robot {robot_idx}[/cyan] - Steps",
                total=self.max_steps,
                visible=True,
            )

        self.robot_states[robot_idx] = robot_state
        self.job_stats.total_episodes += job_info["num_episodes"]

    def _handle_episode_start(self, message: dict):
        """Handle episode start message."""
        robot_idx = message["robot_idx"]

        if robot_idx in self.robot_states:
            robot_state = self.robot_states[robot_idx]
            robot_state.current_step = 0

            # Reset step progress bar
            if self.progress and robot_state.step_bar_id is not None:
                self.progress.update(
                    robot_state.step_bar_id,
                    completed=0,
                )

    def _handle_episode_end(self, message: dict):
        """Handle episode end message."""
        robot_idx = message["robot_idx"]

        if robot_idx in self.robot_states:
            robot_state = self.robot_states[robot_idx]
            robot_state.current_episode += 1

            if message["success"]:
                robot_state.successes += 1
                self.job_stats.total_successes += 1

            self.job_stats.completed_episodes += 1

            # Update episode progress bar
            if self.progress and robot_state.episode_bar_id is not None:
                self.progress.update(
                    robot_state.episode_bar_id,
                    completed=robot_state.current_episode,
                )

    def _handle_step_batch(self, message: dict):
        """Handle step batch update message."""
        robot_idx = message["robot_idx"]

        if robot_idx in self.robot_states:
            robot_state = self.robot_states[robot_idx]
            robot_state.current_step = message["step_count"]
            robot_state.steps_per_sec = message["steps/s"]

            # Update step progress bar
            if self.progress and robot_state.step_bar_id is not None:
                self.progress.update(
                    robot_state.step_bar_id,
                    completed=robot_state.current_step,
                )

    def _handle_worker_complete(self, message: dict):
        """Handle worker completion message."""
        robot_idx = message["robot_idx"]

        if robot_idx in self.robot_states:
            robot_state = self.robot_states[robot_idx]
            robot_state.active = False
            robot_state.completed = True

            # Hide the progress bars for this completed robot
            if self.progress:
                if robot_state.episode_bar_id is not None:
                    self.progress.update(robot_state.episode_bar_id, visible=False)
                if robot_state.step_bar_id is not None:
                    self.progress.update(robot_state.step_bar_id, visible=False)

            # Update overall progress
            self.job_stats.completed_jobs += 1
            if self.progress and self.overall_bar_id is not None:
                self.progress.update(
                    self.overall_bar_id,
                    completed=self.job_stats.completed_jobs,
                )

    def _print_final_summary(self):
        """Print final summary after completion."""
        with self._lock:
            total_time = time.time() - self.job_stats.start_time
            success_rate = (
                self.job_stats.total_successes / self.job_stats.completed_episodes * 100
                if self.job_stats.completed_episodes > 0
                else 0.0
            )

            self.console.print(
                "\n[bold green]===== Evaluation Complete =====[/bold green]"
            )
            self.console.print(
                f"Total Jobs: {self.job_stats.completed_jobs}/{self.job_stats.total_jobs}"
            )
            self.console.print(
                f"Total Episodes: {self.job_stats.completed_episodes}/{self.job_stats.total_episodes}"
            )
            self.console.print(f"Total Successes: {self.job_stats.total_successes}")
            self.console.print(f"Overall Success Rate: {success_rate:.2f}%")
            self.console.print(f"Total Time: {total_time:.2f}s")


class ConciseProgressManager(ProgressManager):
    """
    Simplified progress manager that shows only overall stats and a compact robot grid.

    Displays:
    - Overall progress bar
    - Summary stats (Jobs, Episodes, Success Rate, Time)
    - Compact grid showing each robot's current episode and step/s
    """

    def __enter__(self) -> ConciseProgressManager:
        """Initialize Rich Progress and start monitoring thread."""
        # Create Rich Progress with only overall bar
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
        )

        # Create overall progress bar
        self.overall_bar_id = self.progress.add_task(
            "[bold green]Overall Progress",
            total=self.total_jobs,
        )

        # Start Rich Live display
        self.live = Live(
            self._generate_display(),
            console=self.console,
            refresh_per_second=4,
        )
        self.live.start()

        # Start monitoring thread
        self.job_stats.start_time = time.time()
        self._monitor_thread = threading.Thread(
            target=self._monitor_queue,
            daemon=True,
        )
        self._monitor_thread.start()

        return self

    def _generate_display(self):
        """Generate the concise Rich display."""
        layout = Table.grid(padding=(0, 0))
        layout.add_column(justify="left", ratio=1)

        # Add overall progress bar
        if self.progress:
            layout.add_row(self.progress)

        # Add summary stats
        with self._lock:
            elapsed = time.time() - self.job_stats.start_time
            success_rate = (
                self.job_stats.total_successes / self.job_stats.completed_episodes * 100
                if self.job_stats.completed_episodes > 0
                else 0.0
            )

            stats_text = (
                f"[bold]Jobs:[/bold] {self.job_stats.completed_jobs}/{self.job_stats.total_jobs}  "
                f"[bold]Episodes:[/bold] {self.job_stats.completed_episodes}/{self.job_stats.total_episodes}  "
                f"[bold]Success Rate:[/bold] {success_rate:.1f}%  "
                f"[bold]Time:[/bold] {elapsed:.1f}s"
            )
            layout.add_row(stats_text)

            # Add compact robot grid in two columns
            active_robots = sorted(
                [
                    rs
                    for rs in self.robot_states.values()
                    if rs.active and not rs.completed
                ],
                key=lambda rs: rs.robot_idx,
            )

            if active_robots:
                # Split robots into two columns
                mid = (len(active_robots) + 1) // 2
                left_robots = active_robots[:mid]
                right_robots = active_robots[mid:]

                # Create two-column layout
                columns = Table.grid(padding=(0, 2))
                columns.add_column()
                columns.add_column()

                # Create left table
                left_table = Table(
                    show_header=True, box=None, padding=(0, 1), show_edge=False
                )
                left_table.add_column("Robot", style="cyan", width=6)
                left_table.add_column("T", width=3)
                left_table.add_column("Ep", justify="right", width=8)
                left_table.add_column("Step/s", justify="right", width=7)
                left_table.add_column("Succ", justify="right", width=10)

                for rs in left_robots:
                    success_pct = (
                        rs.successes / rs.current_episode * 100
                        if rs.current_episode > 0
                        else 0.0
                    )
                    left_table.add_row(
                        f"R{rs.robot_idx}",
                        str(rs.task_id),
                        f"{rs.current_episode}/{rs.total_episodes}",
                        f"{rs.steps_per_sec:.1f}",
                        f"{rs.successes}/{rs.current_episode} ({success_pct:.0f}%)",
                    )

                # Create right table
                right_table = Table(
                    show_header=True, box=None, padding=(0, 1), show_edge=False
                )
                right_table.add_column("Robot", style="cyan", width=6)
                right_table.add_column("T", width=3)
                right_table.add_column("Ep", justify="right", width=8)
                right_table.add_column("Step/s", justify="right", width=7)
                right_table.add_column("Succ", justify="right", width=10)

                for rs in right_robots:
                    success_pct = (
                        rs.successes / rs.current_episode * 100
                        if rs.current_episode > 0
                        else 0.0
                    )
                    right_table.add_row(
                        f"R{rs.robot_idx}",
                        str(rs.task_id),
                        f"{rs.current_episode}/{rs.total_episodes}",
                        f"{rs.steps_per_sec:.1f}",
                        f"{rs.successes}/{rs.current_episode} ({success_pct:.0f}%)",
                    )

                # Add both tables side by side
                columns.add_row(left_table, right_table)
                layout.add_row(columns)

        return layout

    def _handle_worker_init(self, message: dict):
        """Handle worker initialization message (no individual progress bars)."""
        robot_idx = message["robot_idx"]
        job_info = message["job_info"]

        # Create robot state (without progress bars)
        robot_state = RobotState(
            robot_idx=robot_idx,
            task_id=job_info["task_id"],
            task_suite_name=job_info["task_suite_name"],
            total_episodes=job_info["num_episodes"],
            max_steps=self.max_steps,
            active=True,
        )

        self.robot_states[robot_idx] = robot_state
        self.job_stats.total_episodes += job_info["num_episodes"]

    def _handle_episode_start(self, message: dict):
        """Handle episode start message (no progress bar updates)."""
        robot_idx = message["robot_idx"]
        if robot_idx in self.robot_states:
            self.robot_states[robot_idx].current_step = 0

    def _handle_worker_complete(self, message: dict):
        """Handle worker completion message."""
        robot_idx = message["robot_idx"]

        if robot_idx in self.robot_states:
            robot_state = self.robot_states[robot_idx]
            robot_state.active = False
            robot_state.completed = True

            # Update overall progress
            self.job_stats.completed_jobs += 1
            if self.progress and self.overall_bar_id is not None:
                self.progress.update(
                    self.overall_bar_id,
                    completed=self.job_stats.completed_jobs,
                )


class LoggingProgressManager(ProgressManager):
    """
    Simple logging progress manager that prints messages without progress bars.

    Logs all events except step_batch messages to reduce verbosity.
    """

    def __enter__(self) -> LoggingProgressManager:
        """Initialize without Rich Progress - just use console for logging."""
        self.console = Console()

        # Start monitoring thread
        self.job_stats.start_time = time.time()
        self._monitor_thread = threading.Thread(
            target=self._monitor_queue,
            daemon=True,
        )
        self._monitor_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean shutdown: stop thread, drain queue."""
        # Signal stop
        self._stop_event.set()

        # Wait for monitor thread to finish
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Drain remaining messages
        self._drain_queue()

        # Print final summary
        self._print_final_summary()

        return False

    def _monitor_queue(self):
        """Background thread that monitors the queue and logs messages."""
        while not self._stop_event.is_set():
            try:
                # Non-blocking queue check
                try:
                    message = self.queue.get(timeout=self.update_interval)
                    self._handle_message(message)
                except queue_module.Empty:
                    pass
            except Exception as e:
                self.console.print(f"[red]Error in monitor thread: {e}[/red]")

    def _handle_message(self, message: dict):
        """Process a single message from the queue and log it."""
        msg_type = message["type"]

        with self._lock:
            if msg_type == "worker_init":
                self._handle_worker_init(message)
                self.console.print(
                    f"[cyan][Robot {message['robot_idx']}][/cyan] Starting task {message['job_info']['task_id']} "
                    f"({message['job_info']['task_suite_name']}) - {message['job_info']['num_episodes']} episodes"
                )
            elif msg_type == "episode_start":
                self._handle_episode_start(message)
                robot_idx = message["robot_idx"]
                if robot_idx in self.robot_states:
                    rs = self.robot_states[robot_idx]
                    self.console.print(
                        f"[cyan][Robot {robot_idx}][/cyan] Episode {message['episode_idx'] + 1}/{rs.total_episodes} started"
                    )
            elif msg_type == "episode_end":
                robot_idx = message["robot_idx"]
                if robot_idx in self.robot_states:
                    rs = self.robot_states[robot_idx]
                    status = (
                        "[green]SUCCESS[/green]"
                        if message["success"]
                        else "[red]FAILURE[/red]"
                    )
                    self.console.print(
                        f"[cyan][Robot {robot_idx}][/cyan] Episode {message['episode_idx'] + 1}/{rs.total_episodes} ended: {status}"
                    )
                self._handle_episode_end(message)
            elif msg_type == "step_batch":
                # Skip step_batch messages to reduce verbosity
                self._handle_step_batch(message)
            elif msg_type == "worker_complete":
                robot_idx = message["robot_idx"]
                self.console.print(
                    f"[cyan][Robot {robot_idx}][/cyan] Completed: "
                    f"{message['total_successes']}/{message['total_episodes']} successes "
                    f"({message['total_successes'] / message['total_episodes'] * 100:.1f}%)"
                )
                self._handle_worker_complete(message)

    def _handle_worker_init(self, message: dict):
        """Handle worker initialization message."""
        robot_idx = message["robot_idx"]
        job_info = message["job_info"]

        # Create robot state (without progress bars)
        robot_state = RobotState(
            robot_idx=robot_idx,
            task_id=job_info["task_id"],
            task_suite_name=job_info["task_suite_name"],
            total_episodes=job_info["num_episodes"],
            max_steps=self.max_steps,
            active=True,
        )

        self.robot_states[robot_idx] = robot_state
        self.job_stats.total_episodes += job_info["num_episodes"]

    def _handle_episode_start(self, message: dict):
        """Handle episode start message."""
        robot_idx = message["robot_idx"]
        if robot_idx in self.robot_states:
            self.robot_states[robot_idx].current_step = 0

    def _handle_worker_complete(self, message: dict):
        """Handle worker completion message."""
        robot_idx = message["robot_idx"]

        if robot_idx in self.robot_states:
            robot_state = self.robot_states[robot_idx]
            robot_state.active = False
            robot_state.completed = True

            # Update overall progress
            self.job_stats.completed_jobs += 1


class DebugQueue:
    """Mock queue that prints messages immediately for debug mode."""

    def put_nowait(self, message: dict):
        msg_type = message["type"]
        if msg_type == "worker_init":
            print(
                f"[Robot {message['robot_idx']}] Starting task {message['job_info']['task_id']}"
            )
        elif msg_type == "episode_start":
            print(
                f"[Robot {message['robot_idx']}] Episode {message['episode_idx']} started"
            )
        elif msg_type == "episode_end":
            status = "SUCCESS" if message["success"] else "FAILURE"
            print(
                f"[Robot {message['robot_idx']}] Episode {message['episode_idx']} ended: {status}"
            )
        elif msg_type == "step_batch":
            print(f"[Robot {message['robot_idx']}] Step {message['step_count']}")
        elif msg_type == "worker_complete":
            print(
                f"[Robot {message['robot_idx']}] Completed: {message['total_successes']}/{message['total_episodes']} successes"
            )


def get_progress_manager(
    progress_type: Literal["verbose", "concise", "logging", None],
    num_robots: int = 1,
    total_jobs: int = 1,
    max_steps: int = 300,
    update_interval: float = 0.1,
) -> ProgressManager:
    """
    Factory function to create the appropriate progress manager.

    Args:
        progress_type: Type of progress manager ("verbose", "concise", "logging", or None)
        num_robots: Number of robot workers
        total_jobs: Total number of jobs to complete
        max_steps: Maximum steps per episode
        update_interval: How often to check queue (seconds)

    Returns:
        The appropriate progress manager or nullcontext if progress_type is None
    """
    if progress_type == "verbose":
        return ProgressManager(num_robots, total_jobs, max_steps, update_interval)
    elif progress_type == "concise":
        return ConciseProgressManager(
            num_robots, total_jobs, max_steps, update_interval
        )
    elif progress_type == "logging":
        return LoggingProgressManager(
            num_robots, total_jobs, max_steps, update_interval
        )
    else:
        return nullcontext()
