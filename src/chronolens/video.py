from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


def probe_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise RuntimeError("Invalid FPS detected")

    duration = frame_count / fps

    return VideoInfo(
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration=duration,
    )


def read_frame(video_path: str | Path, timestamp: float) -> np.ndarray:
    info = probe_video(video_path)

    timestamp = max(0.0, min(float(timestamp), info.duration))

    cap = cv2.VideoCapture(info.path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame at {timestamp:.2f}s")

    return frame


def sample_timestamps(
    video_path: str | Path,
    count: int = 8,
) -> list[float]:
    info = probe_video(video_path)

    if count <= 0:
        raise ValueError("count must be greater than zero")

    if count == 1:
        return [info.duration / 2.0]

    margin = min(0.25, info.duration * 0.01)

    timestamps = np.linspace(
        margin,
        max(margin, info.duration - margin),
        count,
    )

    return [float(t) for t in timestamps]


def save_sample_frames(
    video_path: str | Path,
    output_dir: str | Path,
    count: int = 8,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for i, timestamp in enumerate(sample_timestamps(video_path, count), start=1):
        frame = read_frame(video_path, timestamp)

        output_path = output_dir / f"sample_{i:02d}_{timestamp:.2f}s.jpg"

        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to save {output_path}")

        saved.append(str(output_path))

    return saved
