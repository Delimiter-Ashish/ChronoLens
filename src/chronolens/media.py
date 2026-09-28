from __future__ import annotations

import hashlib
import html
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def file_fingerprint(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha1()
    h.update(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:16]


def probe_video(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps > 0 else 0.0
    cap.release()
    if duration <= 0:
        raise RuntimeError("Video duration could not be determined.")
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


def read_frame(path: str | Path, timestamp: float):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame at {timestamp:.2f}s")
    return frame


def save_thumbnail(path: str | Path, timestamp: float, out_path: str | Path) -> str:
    frame = read_frame(path, timestamp)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return str(out_path)


def cut_clip(source: str | Path, start: float, end: float, output: str | Path) -> str:
    source = str(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.25, float(end) - float(start))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}", "-i", source,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", "-y", str(output),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        fallback = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}", "-i", source,
            "-t", f"{duration:.3f}", "-map", "0:v:0", "-an",
            "-c:v", "mpeg4", "-q:v", "4", "-y", str(output),
        ]
        subprocess.run(fallback, check=True)
    return str(output)


def timeline_html(duration: float, hits: list[dict]) -> str:
    markers = []
    for hit in hits:
        pct = min(100.0, max(0.0, hit["timestamp"] / max(duration, 1e-6) * 100.0))
        score = float(hit["score"])
        title = html.escape(
            f"#{hit['rank']} {format_timestamp(hit['timestamp'])} score={score:.3f}"
        )
        markers.append(
            f'<div class="cl-marker" style="left:{pct:.3f}%;" title="{title}">'
            f'<span>{hit["rank"]}</span></div>'
        )
    return f"""
    <div class="cl-timeline-wrap">
      <div class="cl-timeline-labels">
        <span>00:00</span><span>{format_timestamp(duration)}</span>
      </div>
      <div class="cl-timeline">
        <div class="cl-track"></div>
        {''.join(markers)}
      </div>
    </div>
    """
