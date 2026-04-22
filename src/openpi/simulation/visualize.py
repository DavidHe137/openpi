from jaxtyping import Int
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from .classes import CONTROL_PERIOD


def plot_actions_left(
    actions_left: Int[np.ndarray, "num_robots num_steps"],
    title: str = "Actions Left Per Robot Over Time",
    filename: str = "actions_left.png",
    control_hz: int = 20,
):
    """Plot heatmap of actions left for each robot over time."""
    num_robots, num_steps = actions_left.shape
    fig, ax = plt.subplots(figsize=(max(12, num_steps // 20), max(4, num_robots * 0.6)))

    im = ax.imshow(
        actions_left[::-1],
        aspect="auto",
        cmap="RdYlGn",
        interpolation="nearest",
        origin="lower",
        vmin=0,
    )

    # X-axis: wall-clock time in seconds (one tick per second)
    tick_interval = control_hz  # one tick per second
    x_ticks = np.arange(0, num_steps, tick_interval)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(t // tick_interval)}s" for t in x_ticks], fontsize=6)
    ax.set_xlabel("Wall-clock time in seconds", fontweight="bold")

    # Y-axis
    ax.set_yticks(range(num_robots))
    ax.set_yticklabels([f"robot_{i}" for i in reversed(range(num_robots))], fontsize=8)
    ax.set_ylabel("Robot", fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Actions left in queue", fontweight="bold")

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_staleness(
    staleness: np.ndarray, title: str = "Staleness per Robot over Time", filename: str = "staleness.png"
):
    """Plot heatmap of staleness (steps since last chunk arrival) for each robot over time."""
    staleness_transposed = staleness.T  # (num_robots, num_steps)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(staleness_transposed, aspect="auto", cmap="Reds", interpolation="nearest", origin="lower")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Robot ID")
    ax.set_title(title)

    num_robots = staleness_transposed.shape[0]
    ax.set_yticks(range(num_robots))
    ax.set_yticklabels([f"Robot {i}" for i in range(num_robots)])

    plt.colorbar(im, ax=ax, label="Steps Since Last Chunk Arrival")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def _get_valid_chunks(robot, step: int):
    """Return chunks that have arrived and still have actions at the given control step."""
    return [c for c in robot.chunks if c.arrival_step <= step and c.actions_left_at(step) > 0]


def plot_chunk_waste(
    simulator, end_time: int, title: str = "Action Chunk Overlap and Waste", filename: str = "chunk_waste.png"
):
    """
    Plot all action chunks for each robot, showing which portions are active vs wasted.

    Args:
        simulator: Simulator object with state containing robots and their chunks
        end_time: End time in milliseconds
        title: Plot title
        filename: Output filename
    """
    robots = simulator.state.robots
    num_robots = len(robots)
    num_control_steps = end_time // CONTROL_PERIOD

    fig, axes = plt.subplots(num_robots, 1, figsize=(16, max(8, num_robots * 2)), sharex=True, squeeze=False)

    for robot_index, robot in enumerate(robots):
        ax = axes[robot_index, 0]

        if not robot.chunks:
            ax.text(num_control_steps / 2, 0.5, "No action chunks", ha="center", va="center", fontsize=12, color="gray")
            ax.set_ylim(0, 1)
        else:
            # For each chunk, we'll draw it as a horizontal bar
            # We need to stack overlapping chunks vertically
            chunk_y_positions = []

            for chunk_idx, chunk in enumerate(robot.chunks):
                arrival = chunk.arrival_step
                end_step = chunk.start_action + chunk.horizon

                # Find a Y position that doesn't overlap with existing chunks
                y_pos = 0
                while True:
                    overlap = False
                    for other_arrival, other_end, other_y in chunk_y_positions:
                        if other_y == y_pos and not (end_step <= other_arrival or arrival >= other_end):
                            overlap = True
                            break
                    if not overlap:
                        break
                    y_pos += 1

                chunk_y_positions.append((arrival, end_step, y_pos))

                # Draw the chunk, coloring each timestep based on active/wasted status
                for step_idx in range(max(0, arrival), min(end_step, num_control_steps)):
                    if chunk.actions_left_at(step_idx) == 0:
                        continue

                    # Check if this chunk is active at this timestep
                    valid_chunks = _get_valid_chunks(robot, step_idx)
                    if valid_chunks:
                        max_actions = max(c.actions_left_at(step_idx) for c in valid_chunks)
                        chunks_with_max = [c for c in valid_chunks if c.actions_left_at(step_idx) == max_actions]
                        active_chunk = max(chunks_with_max, key=lambda c: c.arrival_step)
                        is_active = active_chunk == chunk
                    else:
                        is_active = False

                    # Draw a small rectangle for this timestep
                    color = "#2ecc71" if is_active else "#e74c3c"  # Green if active, red if wasted
                    alpha = 1.0 if is_active else 0.6

                    rect = Rectangle(
                        (step_idx, y_pos),
                        1,  # width = 1 control step
                        0.8,  # height
                        facecolor=color,
                        edgecolor="none",
                        alpha=alpha,
                    )
                    ax.add_patch(rect)

                # Add chunk label
                chunk_center = (arrival + min(end_step, num_control_steps)) / 2
                ax.text(
                    chunk_center,
                    y_pos + 0.4,
                    f"C{chunk_idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                )

            max_y = max(y for _, _, y in chunk_y_positions) if chunk_y_positions else 0
            ax.set_ylim(-0.5, max_y + 1.5)

        ax.set_ylabel(f"Robot {robot_index}", fontweight="bold")
        ax.set_xlim(0, num_control_steps)
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_yticks([])

    axes[-1, 0].set_xlabel("Control Step", fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Active (executed)"),
        Patch(facecolor="#e74c3c", alpha=0.6, label="Wasted (overwritten)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gpu_timeline(
    batch_history, num_robots: int, end_time: int, title: str = "GPU Timeline", filename: str = "gpu_timeline.png"
):
    """
    Plot timeline showing when GPU is processing which robots.

    Args:
        batch_history: List of BatchRecord objects
        num_robots: Total number of robots
        end_time: End time in milliseconds
    """
    fig, ax = plt.subplots(figsize=(14, max(6, num_robots * 0.5)))

    # Color palette for robots
    colors = plt.cm.hsv(np.linspace(0, 1, num_robots, endpoint=False))

    # Plot each batch as a rectangle
    for batch_record in batch_history:
        for robot_index in batch_record.robot_indices:
            # Draw rectangle for this robot's processing time
            rect = Rectangle(
                (batch_record.start_time, robot_index - 0.4),
                batch_record.duration,
                0.8,
                facecolor=colors[robot_index],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, end_time)
    ax.set_ylim(-0.5, num_robots - 0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Robot ID")
    ax.set_title(title)
    ax.set_yticks(range(num_robots))
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scheduler_comparison(scheduler_results: dict[str, dict], filename: str = "scheduler_comparison.png"):
    """
    Create comprehensive comparison plots across schedulers.

    Args:
        scheduler_results: Dict mapping scheduler_name -> dict with keys:
            - 'actions_left': numpy array (num_robots x num_steps)
            - 'metrics': dict of computed metrics
    """
    scheduler_names = list(scheduler_results.keys())
    num_schedulers = len(scheduler_names)

    # Create display names by removing "Scheduler" suffix
    display_names = [name.replace("Scheduler", "") for name in scheduler_names]

    # Extract metrics for convenience
    results = {name: data["metrics"] for name, data in scheduler_results.items()}

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    # 1. Percentage of GPU Time Used comparison (bar plot)
    ax1 = fig.add_subplot(gs[0, 0])
    percentage_of_gpu_time_used = [results[name]["percentage_of_gpu_time_used"] * 100 for name in scheduler_names]
    bars = ax1.bar(
        display_names, percentage_of_gpu_time_used, color=plt.cm.Set2(np.linspace(0, 1, num_schedulers, endpoint=False))
    )
    ax1.set_ylabel("Percentage of GPU Time Used (%)")
    ax1.set_title("Percentage of GPU Time Used")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(len(display_names)))
    ax1.set_xticklabels(display_names, rotation=45, ha="right")

    # 2. Total starvations (bar plot)
    ax2 = fig.add_subplot(gs[0, 1])
    total_starvations = [results[name]["total_starvations"] for name in scheduler_names]
    bars = ax2.bar(
        display_names, total_starvations, color=plt.cm.Set3(np.linspace(0, 1, num_schedulers, endpoint=False))
    )
    ax2.set_ylabel("Total Starvations")
    ax2.set_title("Total Robot Starvations")
    ax2.set_xticks(range(len(display_names)))
    ax2.set_xticklabels(display_names, rotation=45, ha="right")
    for bar, val in zip(bars, total_starvations):
        if val > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val}", ha="center", va="bottom", fontsize=9
            )

    # # 3. Wasted actions (bar plot)
    # ax3 = fig.add_subplot(gs[0, 2])
    # wasted_actions = [results[name]['wasted_actions'] for name in scheduler_names]
    # bars = ax3.bar(display_names, wasted_actions, color=sns.color_palette("Pastel1", num_schedulers))
    # ax3.set_ylabel('Wasted Actions')
    # ax3.set_title('Wasted Actions from Chunk Overlap')
    # ax3.set_xticklabels(display_names, rotation=45, ha='right')
    # for bar, val in zip(bars, wasted_actions):
    #     if val > 0:
    #         ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(wasted_actions) * 0.02,
    #                 f'{val}', ha='center', va='bottom', fontsize=9)

    # 3. Actions left distribution (violin plot)
    ax5 = fig.add_subplot(gs[0, 2])
    actions_data = []
    for name in scheduler_names:
        # Flatten the actions_left array (num_steps x num_robots) to get all values
        actions_left = scheduler_results[name]["actions_left"]
        actions_data.append(actions_left.flatten())

    # Create violin plot
    parts = ax5.violinplot(actions_data, positions=range(len(scheduler_names)), showmeans=True, showmedians=True)
    ax5.set_xticks(range(len(scheduler_names)))
    ax5.set_xticklabels(display_names, rotation=45, ha="right")
    ax5.set_ylabel("Actions Left")
    ax5.set_title("Distribution of Actions Left")
    ax5.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Scheduler Performance Comparison", fontsize=16, y=0.995)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def create_all_visualizations(scheduler_results: dict[str, dict], output_dir: str = "."):
    """
    Create all visualizations for scheduler comparison.

    Args:
        scheduler_results: Dict mapping scheduler_name -> dict with keys:
            - 'actions_left': numpy array
            - 'batch_history': list of BatchRecord objects
            - 'simulator': Simulator object
            - 'metrics': dict of computed metrics
            - 'num_robots': int
            - 'end_time': int (milliseconds)
    """
    import os

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # For each scheduler, create individual plots
    for scheduler_name, data in scheduler_results.items():
        safe_name = scheduler_name.lower().replace(" ", "_")

        # Actions left heatmap
        plot_actions_left(
            data["actions_left"],
            title=f"{scheduler_name}: Actions Left per Robot over Time",
            filename=f"{output_dir}/{safe_name}_actions_left.png",
        )

        # GPU timeline
        plot_gpu_timeline(
            data["batch_history"],
            data["num_robots"],
            data["end_time"],
            title=f"{scheduler_name}: GPU Processing Timeline",
            filename=f"{output_dir}/{safe_name}_gpu_timeline.png",
        )

        # Chunk waste visualization
        plot_chunk_waste(
            data["simulator"],
            data["end_time"],
            title=f"{scheduler_name}: Action Chunk Overlap and Waste",
            filename=f"{output_dir}/{safe_name}_chunk_waste.png",
        )

    # Create comparison plot
    plot_scheduler_comparison(scheduler_results, filename=f"{output_dir}/scheduler_comparison.png")

    print(f"\nVisualizations saved to {output_dir}/")
    print(
        "   - Individual scheduler plots: [scheduler]_actions_left.png, [scheduler]_staleness.png, [scheduler]_gpu_timeline.png, [scheduler]_chunk_waste.png"
    )
    print("   - Comparison: scheduler_comparison.png")


def create_pdf_report(scheduler_results: dict[str, dict], filename: str = "scheduler_report.pdf"):
    """
    Create a comprehensive PDF report with all visualizations.

    Args:
        scheduler_results: Dict mapping scheduler_name -> dict with keys:
            - 'actions_left': numpy array
            - 'batch_history': list of BatchRecord objects
            - 'metrics': dict of computed metrics
            - 'num_robots': int
            - 'end_time': int (milliseconds)
        filename: Output PDF filename
    """
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(filename) as pdf:
        # Pages 1-5: One comparison chart per page
        print("   Creating comparison charts...")

        scheduler_names = list(scheduler_results.keys())
        num_schedulers = len(scheduler_names)
        metrics_dict = {name: data["metrics"] for name, data in scheduler_results.items()}
        display_names = [name.replace("Scheduler", "") for name in scheduler_names]

        # Page 1: Percentage of GPU Time Used
        fig, ax = plt.subplots(figsize=(10, 6))
        percentage_of_gpu_time_used = [
            metrics_dict[name]["percentage_of_gpu_time_used"] * 100 for name in scheduler_names
        ]
        bars = ax.bar(
            display_names,
            percentage_of_gpu_time_used,
            color=plt.cm.Set2(np.linspace(0, 1, num_schedulers, endpoint=False)),
        )
        ax.set_ylabel("Percentage of GPU Time Used (%)")
        ax.set_title("Percentage of GPU Time Used")
        ax.set_ylim(0, 110)
        ax.set_xticks(range(len(display_names)))
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        for bar, val in zip(bars, percentage_of_gpu_time_used):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 2: Total starvations
        fig, ax = plt.subplots(figsize=(10, 6))
        total_starvations = [metrics_dict[name]["total_starvations"] for name in scheduler_names]
        bars = ax.bar(
            display_names, total_starvations, color=plt.cm.Set3(np.linspace(0, 1, num_schedulers, endpoint=False))
        )
        ax.set_ylabel("Total Starvations")
        ax.set_title("Total Robot Starvations")
        ax.set_xticks(range(len(display_names)))
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        for bar, val in zip(bars, total_starvations):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # # Page 3: Wasted actions
        # fig, ax = plt.subplots(figsize=(10, 6))
        # wasted_actions = [metrics_dict[name]['wasted_actions'] for name in scheduler_names]
        # bars = ax.bar(display_names, wasted_actions, color=sns.color_palette("Pastel1", num_schedulers))
        # ax.set_ylabel('Wasted Actions')
        # ax.set_title('Wasted Actions from Chunk Overlap')
        # ax.set_xticklabels(display_names, rotation=45, ha='right')
        # max_wasted = max(wasted_actions) if wasted_actions else 1
        # for bar, val in zip(bars, wasted_actions):
        #     if val > 0:
        #         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_wasted * 0.02,
        #                 f'{val}', ha='center', va='bottom', fontsize=9)
        # plt.tight_layout()
        # pdf.savefig(fig, bbox_inches='tight')
        # plt.close()

        # Page 3: Actions left distribution (violin plot)
        fig, ax = plt.subplots(figsize=(10, 6))
        actions_data = []
        for name in scheduler_names:
            actions_left = scheduler_results[name]["actions_left"]
            actions_data.append(actions_left.flatten())
        parts = ax.violinplot(actions_data, positions=range(len(scheduler_names)), showmeans=True, showmedians=True)
        ax.set_xticks(range(len(scheduler_names)))
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        ax.set_ylabel("Actions Left")
        ax.set_title("Distribution of Actions Left")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Pages 2+: For each scheduler, add actions_left and gpu_timeline
        for scheduler_name, data in scheduler_results.items():
            print(f"   Adding {scheduler_name} plots...")

            # Actions left heatmap
            al = data["actions_left"]
            num_robots_al, num_steps_al = al.shape
            fig, ax = plt.subplots(figsize=(max(12, num_steps_al // 20), max(4, num_robots_al * 0.6)))
            im = ax.imshow(
                al,
                aspect="auto",
                cmap="RdYlGn",
                interpolation="nearest",
                origin="lower",
                vmin=0,
            )
            tick_interval = 20  # control_hz: one tick per second
            x_ticks = np.arange(0, num_steps_al, tick_interval)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{int(t // tick_interval)}s" for t in x_ticks], fontsize=6)
            ax.set_xlabel("Wall-clock time in seconds", fontweight="bold")
            ax.set_yticks(range(num_robots_al))
            ax.set_yticklabels([f"robot_{i}" for i in range(num_robots_al)], fontsize=8)
            ax.set_ylabel("Robot", fontweight="bold")
            ax.set_title(f"{scheduler_name}: Actions Left Per Robot Over Time", fontsize=14, fontweight="bold")
            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label("Actions left in queue", fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # GPU timeline
            fig, ax = plt.subplots(figsize=(14, max(6, data["num_robots"] * 0.5)))
            colors = plt.cm.hsv(np.linspace(0, 1, data["num_robots"], endpoint=False))
            for batch_record in data["batch_history"]:
                for robot_index in batch_record.robot_indices:
                    rect = Rectangle(
                        (batch_record.start_time, robot_index - 0.4),
                        batch_record.duration,
                        0.8,
                        facecolor=colors[robot_index],
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.7,
                    )
                    ax.add_patch(rect)
            ax.set_xlim(0, data["end_time"])
            ax.set_ylim(-0.5, data["num_robots"] - 0.5)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Robot ID")
            ax.set_title(f"{scheduler_name}: GPU Processing Timeline")
            ax.set_yticks(range(data["num_robots"]))
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

            # Chunk waste visualization
            simulator = data["simulator"]
            robots = simulator.state.robots
            num_robots_sim = len(robots)
            num_control_steps = data["end_time"] // CONTROL_PERIOD

            fig, axes = plt.subplots(
                num_robots_sim, 1, figsize=(16, max(8, num_robots_sim * 2)), sharex=True, squeeze=False
            )

            for robot_index, robot in enumerate(robots):
                ax = axes[robot_index, 0]

                if not robot.chunks:
                    ax.text(
                        num_control_steps / 2,
                        0.5,
                        "No action chunks",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="gray",
                    )
                    ax.set_ylim(0, 1)
                else:
                    chunk_y_positions = []

                    for chunk_idx, chunk in enumerate(robot.chunks):
                        arrival = chunk.arrival_step
                        end_step = chunk.start_action + chunk.horizon

                        # Find Y position
                        y_pos = 0
                        while True:
                            overlap = False
                            for other_arrival, other_end, other_y in chunk_y_positions:
                                if other_y == y_pos and not (end_step <= other_arrival or arrival >= other_end):
                                    overlap = True
                                    break
                            if not overlap:
                                break
                            y_pos += 1

                        chunk_y_positions.append((arrival, end_step, y_pos))

                        # Draw chunk
                        for step_idx in range(max(0, arrival), min(end_step, num_control_steps)):
                            if chunk.actions_left_at(step_idx) == 0:
                                continue

                            valid_chunks = _get_valid_chunks(robot, step_idx)
                            if valid_chunks:
                                max_actions = max(c.actions_left_at(step_idx) for c in valid_chunks)
                                chunks_with_max = [
                                    c for c in valid_chunks if c.actions_left_at(step_idx) == max_actions
                                ]
                                active_chunk = max(chunks_with_max, key=lambda c: c.arrival_step)
                                is_active = active_chunk == chunk
                            else:
                                is_active = False

                            color = "#2ecc71" if is_active else "#e74c3c"
                            alpha = 1.0 if is_active else 0.6

                            rect = Rectangle((step_idx, y_pos), 1, 0.8, facecolor=color, edgecolor="none", alpha=alpha)
                            ax.add_patch(rect)

                        chunk_center = (arrival + min(end_step, num_control_steps)) / 2
                        ax.text(
                            chunk_center,
                            y_pos + 0.4,
                            f"C{chunk_idx}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color="white",
                        )

                    max_y = max(y for _, _, y in chunk_y_positions) if chunk_y_positions else 0
                    ax.set_ylim(-0.5, max_y + 1.5)

                ax.set_ylabel(f"Robot {robot_index}", fontweight="bold")
                ax.set_xlim(0, num_control_steps)
                ax.grid(True, alpha=0.3, axis="x")
                ax.set_yticks([])

            axes[-1, 0].set_xlabel("Control Step", fontweight="bold")

            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#2ecc71", label="Active (executed)"),
                Patch(facecolor="#e74c3c", alpha=0.6, label="Wasted (overwritten)"),
            ]
            fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

            plt.suptitle(f"{scheduler_name}: Action Chunk Overlap and Waste", fontsize=14, fontweight="bold", y=0.995)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    print(f"PDF report saved to {filename}")
