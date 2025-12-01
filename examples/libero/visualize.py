"""
Action chunk visualization for robot policy rollouts.

This module provides utilities to visualize action chunk execution over time,
showing how chunks are predicted and executed in overlapping fashion.

Usage:
    frame_metadata = []
    for t in range(max_steps):
        frame_metadata.append(
            ActionFrameMetadata(
                timestep=t,
                chunk_id=current_chunk_id,
                action_index=action_idx_within_chunk,
            )
        )

    visualized = add_action_chunk_visualization(frames, frame_metadata, replan_steps=5)

For main_rtc_parallel.py integration, you need to expose ActionChunkBroker's internal
state (current chunk_id and action_index) to create the metadata.
"""

import dataclasses
from typing import List

import cv2
import numpy as np


@dataclasses.dataclass
class ActionFrameMetadata:
    """Metadata for a single frame describing action chunk execution.

    Attributes:
        timestep: Absolute timestep in the episode
        chunk_id: ID of the action chunk being executed
        action_index: Index within the chunk (0 to replan_steps-1)
    """
    timestep: int
    chunk_id: int
    action_index: int

    @property
    def chunk_start_time(self) -> int:
        """Compute when this chunk was first predicted."""
        return self.timestep - self.action_index


def add_action_chunk_visualization(
    frames: List[np.ndarray],
    metadata: List[ActionFrameMetadata],
    replan_steps: int = 5,
    time_window: int = 10,
    action_width: int = 18,
    action_spacing: int = 2,
    viz_height: int = 120,
) -> List[np.ndarray]:
    """
    Add action chunk timeline visualization below video frames.

    Creates a visualization showing:
    - Static playhead (yellow line) at center
    - Action chunks scrolling leftward as time progresses
    - Current and previous chunks visible
    - Each chunk in a different color
    - Currently executing action highlighted in green

    Args:
        frames: List of video frames (H, W, 3) as uint8
        metadata: Per-frame metadata describing chunk execution
        replan_steps: Number of actions per chunk
        time_window: Number of timesteps visible before/after playhead
        action_width: Width of each action square in pixels
        action_spacing: Gap between action squares in pixels
        viz_height: Height of visualization area in pixels

    Returns:
        List of frames with visualization added at bottom
    """
    if len(frames) != len(metadata):
        raise ValueError(
            f"Number of frames ({len(frames)}) must match metadata ({len(metadata)})"
        )

    if not frames:
        return []

    # Visualization parameters
    h, w = frames[0].shape[:2]
    chunk_row_height = 20
    action_height = 15
    playhead_x = w // 2
    pixels_per_timestep = action_width + action_spacing

    # Color palette for chunks (cycles through 8 distinct colors)
    color_palette = [
        (255, 150, 100),  # Orange
        (100, 150, 255),  # Light blue
        (150, 255, 100),  # Light green
        (255, 100, 150),  # Pink
        (150, 100, 255),  # Purple
        (255, 255, 100),  # Yellow
        (100, 255, 255),  # Cyan
        (255, 150, 255),  # Magenta
    ]

    visualized_frames = []

    for frame, meta in zip(frames, metadata):
        # Create expanded frame
        expanded_frame = np.zeros((h + viz_height, w, 3), dtype=np.uint8)
        expanded_frame[:h, :, :] = frame
        expanded_frame[h:, :, :] = (40, 40, 40)  # Dark gray background

        viz_y_start = h
        current_time = meta.timestep
        current_chunk_id = meta.chunk_id

        # Draw playhead line
        cv2.line(
            expanded_frame,
            (playhead_x, viz_y_start),
            (playhead_x, h + viz_height),
            (255, 255, 0),
            2,
        )

        # Determine which chunks to draw (current and previous)
        chunks_to_draw = []

        # Current chunk
        current_chunk_start = meta.chunk_start_time
        chunks_to_draw.append((current_chunk_id, current_chunk_start))

        # Previous chunk (if exists and still visible)
        previous_chunk_id = current_chunk_id - 1
        if previous_chunk_id >= 0:
            previous_chunk_start = current_chunk_start - replan_steps
            previous_chunk_end = previous_chunk_start + replan_steps - 1
            time_min = current_time - time_window

            if previous_chunk_end >= time_min:
                chunks_to_draw.append((previous_chunk_id, previous_chunk_start))

        # Sort so previous is on top, current at bottom
        chunks_to_draw.sort(key=lambda x: x[0])

        # Draw each chunk
        for row_idx, (chunk_id, chunk_start_time) in enumerate(chunks_to_draw):
            # Calculate vertical position (bottom up)
            chunk_y = h + viz_height - 10 - (row_idx + 1) * (chunk_row_height + 5)
            if chunk_y < viz_y_start:
                break

            # Get chunk color
            base_color = color_palette[chunk_id % len(color_palette)]

            # Draw each action in the chunk
            for action_idx in range(replan_steps):
                action_time = chunk_start_time + action_idx

                # Position based on timestep
                timestep_offset = action_time - current_time
                action_x = playhead_x + timestep_offset * pixels_per_timestep

                # Skip if outside visible area
                if action_x + action_width < 0 or action_x > w:
                    continue

                # Check if this is the currently executing action
                is_executing = (action_time == current_time)

                # Choose color
                if is_executing:
                    color = (0, 255, 0)  # Bright green
                else:
                    color = base_color

                # Draw action rectangle
                x1 = int(action_x)
                y1 = int(chunk_y - action_height // 2)
                x2 = int(action_x + action_width)
                y2 = int(chunk_y + action_height // 2)
                cv2.rectangle(expanded_frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(expanded_frame, (x1, y1), (x2, y2), (60, 60, 60), 1)

        # Add time indicator
        time_text = f"t={current_time}"
        cv2.putText(
            expanded_frame,
            time_text,
            (10, viz_y_start + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        visualized_frames.append(expanded_frame)

    return visualized_frames
