from __future__ import annotations

import pathlib
import tyro
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
from dataclasses import dataclass
from examples.libero import logging_config
import numpy as np
from typing import Tuple, Dict, Literal
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip


# Color tint presets (RGB multipliers)
COLOR_TINTS: Dict[str, Tuple[float, float, float]] = {
    "red": (1.3, 0.7, 0.7),
    "green": (0.7, 1.3, 0.7),
    "blue": (0.7, 0.7, 1.3),
    "cyan": (0.7, 1.2, 1.2),
    "magenta": (1.2, 0.7, 1.2),
    "yellow": (1.2, 1.2, 0.7),
    "orange": (1.3, 0.9, 0.6),
    "purple": (1.1, 0.7, 1.3),
    "none": (1.0, 1.0, 1.0),
}

ColorTint = Literal[
    "red", "green", "blue", "cyan", "magenta", "yellow", "orange", "purple", "none"
]


@dataclass
class Args:
    video1: str
    """Path to the first video file"""

    video2: str
    """Path to the second video file"""

    output: str = "comparison.mp4"
    """Path for the output comparison video"""

    opacity: float = 0.5
    """Opacity for each video (0.0 to 1.0), default is 0.5 for half transparency"""

    tint1: ColorTint = "red"
    """Color tint for video1"""

    tint2: ColorTint = "cyan"
    """Color tint for video2"""


def get_color_rgb(tint: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert tint multipliers to display RGB color (for legend)."""
    # Apply tint to white (255, 255, 255) to get the resulting color
    r = int(np.clip(255 * tint[0], 0, 255))
    g = int(np.clip(255 * tint[1], 0, 255))
    b = int(np.clip(255 * tint[2], 0, 255))
    return (r, g, b)


LEGEND_HEIGHT = 60


def create_legend(
    video1_path: pathlib.Path,
    video2_path: pathlib.Path,
    tint1: ColorTint,
    tint2: ColorTint,
    width: int,
    duration: float,
) -> ImageClip:
    """Create a legend bar showing which video is which color."""
    # Create black background for legend
    img = Image.new("RGB", (width, LEGEND_HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Get full paths as strings
    path1_str = str(video1_path)
    path2_str = str(video2_path)

    # Calculate appropriate font size based on path length
    max_path_len = max(len(path1_str), len(path2_str))
    # Estimate character width and adjust font size
    available_width = width - 50  # Leave space for box and padding
    if max_path_len * 8 > available_width:  # Rough estimate: 8 pixels per char
        font_size = max(10, int(available_width / max_path_len))
    else:
        font_size = 14

    # Load font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except Exception:
            font = ImageFont.load_default()

    # Get colors for legend
    color1_rgb = get_color_rgb(COLOR_TINTS[tint1])
    color2_rgb = get_color_rgb(COLOR_TINTS[tint2])

    # Legend settings
    box_size = 18
    padding = 5
    line_height = LEGEND_HEIGHT // 2
    x = padding

    # Draw first video entry (top row)
    y1 = (line_height - box_size) // 2
    draw.rectangle([x, y1, x + box_size, y1 + box_size], fill=color1_rgb)
    draw.text((x + box_size + 8, y1), path1_str, font=font, fill=color1_rgb)

    # Draw second video entry (bottom row)
    y2 = line_height + (line_height - box_size) // 2
    draw.rectangle([x, y2, x + box_size, y2 + box_size], fill=color2_rgb)
    draw.text((x + box_size + 8, y2), path2_str, font=font, fill=color2_rgb)

    return ImageClip(np.array(img)).set_duration(duration)


def apply_tint(clip: VideoFileClip, tint: Tuple[float, float, float]) -> VideoFileClip:
    """
    Apply an RGB tint to a video clip.

    Args:
        clip: The video clip to tint
        tint: RGB multipliers (e.g., (1.2, 0.8, 0.8) for red tint)

    Returns:
        Tinted video clip
    """
    r_mult, g_mult, b_mult = tint

    def tint_frame(frame):
        # Apply tint by multiplying RGB channels
        frame = frame.astype(float)
        frame[:, :, 0] *= r_mult  # Red
        frame[:, :, 1] *= g_mult  # Green
        frame[:, :, 2] *= b_mult  # Blue
        return np.clip(frame, 0, 255).astype("uint8")

    return clip.fl_image(tint_frame)


def compare_videos(
    video1_path: pathlib.Path,
    video2_path: pathlib.Path,
    output_path: pathlib.Path,
    opacity: float = 0.5,
    tint1: ColorTint = "red",
    tint2: ColorTint = "cyan",
) -> None:
    """
    Overlay two videos with adjustable transparency to compare them.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Path for output video
        opacity: Opacity level for each video (0.0-1.0)
        tint1: Color name for video1 tint
        tint2: Color name for video2 tint
    """
    # Convert color names to RGB multipliers (validation already done by tyro)
    tint1_rgb = COLOR_TINTS[tint1]
    tint2_rgb = COLOR_TINTS[tint2]
    print("Loading videos...")
    clip1 = VideoFileClip(str(video1_path))
    clip2 = VideoFileClip(str(video2_path))

    # Use the longer duration and freeze the shorter video's last frame
    duration = max(clip1.duration, clip2.duration)

    if clip1.duration < duration:
        # Freeze last frame of clip1 (use a frame 0.1s before end to avoid corrupted frames)
        freeze_time = max(0, clip1.duration - 0.1)
        frozen = clip1.to_ImageClip(freeze_time).set_duration(duration - clip1.duration)
        clip1 = concatenate_videoclips([clip1, frozen])

    if clip2.duration < duration:
        # Freeze last frame of clip2 (use a frame 0.1s before end to avoid corrupted frames)
        freeze_time = max(0, clip2.duration - 0.1)
        frozen = clip2.to_ImageClip(freeze_time).set_duration(duration - clip2.duration)
        clip2 = concatenate_videoclips([clip2, frozen])

    # Get the larger dimensions to ensure both videos fit
    w = max(clip1.w, clip2.w)
    h = max(clip1.h, clip2.h)

    # Resize if needed to match dimensions
    if clip1.size != (w, h):
        clip1 = clip1.resize((w, h))
    if clip2.size != (w, h):
        clip2 = clip2.resize((w, h))

    # Apply tints
    if tint1 != "none":
        print(f"Applying {tint1} tint to video 1...")
        clip1 = apply_tint(clip1, tint1_rgb)
    if tint2 != "none":
        print(f"Applying {tint2} tint to video 2...")
        clip2 = apply_tint(clip2, tint2_rgb)

    # Set opacity for both clips
    clip1 = clip1.set_opacity(opacity)
    clip2 = clip2.set_opacity(opacity)

    print(f"Creating composite with {opacity} opacity...")
    # Position video clips below the legend
    clip1 = clip1.set_position((0, LEGEND_HEIGHT))
    clip2 = clip2.set_position((0, LEGEND_HEIGHT))

    # Create legend at the top
    legend = create_legend(video1_path, video2_path, tint1, tint2, w, duration)
    legend = legend.set_position((0, 0))

    # Create final composition with extended height
    final_height = h + LEGEND_HEIGHT
    final = CompositeVideoClip([clip1, clip2, legend], size=(w, final_height))

    print(f"Writing output to {output_path}...")
    final.write_videofile(str(output_path), codec="libx264")

    print(f"Comparison video saved to {output_path}")


def main(args: Args) -> None:
    video1_path = pathlib.Path(args.video1)
    video2_path = pathlib.Path(args.video2)
    output_path = pathlib.Path(args.output)

    if not video1_path.exists():
        raise FileNotFoundError(f"Video 1 not found: {video1_path}")
    if not video2_path.exists():
        raise FileNotFoundError(f"Video 2 not found: {video2_path}")

    compare_videos(
        video1_path, video2_path, output_path, args.opacity, args.tint1, args.tint2
    )


if __name__ == "__main__":
    logging_config.setup_logging()
    main(tyro.cli(Args))
