from __future__ import annotations

import json
import pathlib
import tyro

from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, ImageClip
from dataclasses import dataclass
from examples.libero import logging_config
from openpi_client.schemas import Timestamp
from examples.libero.subscribers.saver import Result

from typing import Tuple
import math
from tqdm import tqdm
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from typing import Dict, List

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclass
class Args:
    output_dir: str = "data/libero/multi_robot_videos"


def get_grid_dimensions(num_videos: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(num_videos)))
    rows = int(math.ceil(num_videos / cols))

    return rows, cols


def get_grid_index(video_id: int, grid_dimensions: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = grid_dimensions
    return video_id // cols, video_id % cols


@dataclass
class Video:
    path: pathlib.Path
    row: int
    col: int
    start_time: float
    result: Result
    clip: VideoFileClip = None


def get_video_size(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    return stream["width"], stream["height"]


def create_text_overlay(clip: VideoFileClip, text: str) -> ImageClip:
    """Create a text overlay banner for a video clip."""
    w, h = clip.size

    # Create semi-transparent black banner
    img = Image.new("RGBA", (w, 30), color=(0, 0, 0, 180))
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # Draw text
    draw.text((5, 5), text, font=font, fill=(255, 255, 255, 255))

    # Convert to ImageClip
    return (
        ImageClip(np.array(img))
        .set_duration(clip.duration)
        .set_position(("left", "top"))
    )


def create_success_border(
    clip: VideoFileClip, success: bool, duration: float = 0.5
) -> List[ColorClip]:
    """Create colored border clips indicating success/failure."""
    w, h = clip.size
    color = (0, 255, 0) if success else (255, 0, 0)
    thickness = 5
    start = max(0, clip.duration - duration)

    return [
        ColorClip(size=(w, thickness), color=color)
        .set_position(("center", "top"))
        .set_start(start)
        .set_duration(duration),
        ColorClip(size=(w, thickness), color=color)
        .set_position(("center", "bottom"))
        .set_start(start)
        .set_duration(duration),
        ColorClip(size=(thickness, h), color=color)
        .set_position(("left", "center"))
        .set_start(start)
        .set_duration(duration),
        ColorClip(size=(thickness, h), color=color)
        .set_position(("right", "center"))
        .set_start(start)
        .set_duration(duration),
    ]


def annotate_video(clip: VideoFileClip, result: Result) -> VideoFileClip:
    """Annotate video with setup information and success/failure indicator."""
    text = (
        f"Robot {result.robot_idx} | {result.task_suite_name} | Task {result.task_id}"
    )
    text_overlay = create_text_overlay(clip, text)
    border_clips = create_success_border(clip, result.success)

    return CompositeVideoClip([clip, text_overlay] + border_clips)


def grid_videos(videos, output, cols, rows, duration):
    """
    videos: list of Video objects with pre-annotated clips
    """
    w, h = videos[0].clip.size

    clips = []
    for video in videos:
        clip = video.clip.set_position((video.col * w, video.row * h)).set_start(
            video.start_time
        )
        clips.append(clip)

    bg = ColorClip(size=(cols * w, rows * h), color=(0, 0, 0), duration=duration)

    final = CompositeVideoClip([bg] + clips, size=(cols * w, rows * h))
    final.write_videofile(str(output), codec="libx264")


def load_timestamps(output_path: pathlib.Path) -> Dict[pathlib.Path, List[Timestamp]]:
    paths = list(output_path.glob("**/timestamps.csv"))
    return {path.parent: Timestamp.from_csv(path) for path in paths}


def load_results(output_path: pathlib.Path) -> Dict[pathlib.Path, Result]:
    paths = list(output_path.glob("**/metadata.json"))
    return {path.parent: Result.from_json(path) for path in paths}


def combine_videos(output_path: pathlib.Path) -> None:
    video_paths = list(output_path.glob("**/out.mp4"))
    timestamps = load_timestamps(output_path)
    results = load_results(output_path)
    min_timestamp = min(
        [min([t.timestamp for t in timestamps[path]]) for path in timestamps]
    )

    num_robots = len(set([result.robot_idx for result in results.values()]))
    rows, cols = get_grid_dimensions(num_robots)

    # Create video metadata
    videos = [
        Video(
            path,
            *get_grid_index(results[path.parent].robot_idx, (rows, cols)),
            start_time=timestamps[path.parent][0].timestamp - min_timestamp,
            result=results[path.parent],
        )
        for i, path in enumerate(video_paths)
    ]

    # Load and annotate all videos
    for video in tqdm(videos, desc="Loading and annotating videos"):
        clip = VideoFileClip(str(video.path))
        video.clip = annotate_video(clip, video.result)

    grid_videos(videos, output_path / "combined.mp4", cols, rows, duration=10)


def main(args: Args) -> None:
    combine_videos(pathlib.Path(args.output_dir))


if __name__ == "__main__":
    logging_config.setup_logging()
    main(tyro.cli(Args))
