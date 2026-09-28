from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

from chronolens.evidence import EvidenceDetector
from chronolens.video import probe_video, read_frame


class EvidenceTracker:
    def __init__(self):
        self.detector = EvidenceDetector()

    @staticmethod
    def _create_tracker():
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()

        if hasattr(cv2, "legacy") and hasattr(
            cv2.legacy,
            "TrackerCSRT_create",
        ):
            return cv2.legacy.TrackerCSRT_create()

        raise RuntimeError(
            "CSRT tracker is unavailable in this OpenCV build."
        )

    def track(
        self,
        video_path: str,
        start_time: float,
        prompt: str,
        duration: float = 4.0,
        output_path: str = "outputs/tracking/tracked.mp4",
    ) -> dict:

        info = probe_video(video_path)

        first_frame = read_frame(
            video_path,
            start_time,
        )

        image, detections = self.detector.detect_video_frame(
            video_path,
            start_time,
            prompt,
        )

        if not detections:
            raise RuntimeError(
                f"No object detected for prompt: {prompt}"
            )

        best = detections[0]

        x1, y1, x2, y2 = best["box"]

        bbox = (
            int(x1),
            int(y1),
            max(2, int(x2 - x1)),
            max(2, int(y2 - y1)),
        )

        tracker = self._create_tracker()

        initialized = tracker.init(
            first_frame,
            bbox,
        )

        if initialized is False:
            raise RuntimeError(
                "Tracker initialization failed."
            )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            start_time * 1000.0,
        )

        output = Path(output_path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_avi = output.with_suffix(".avi")

        writer = cv2.VideoWriter(
            str(temp_avi),
            cv2.VideoWriter_fourcc(*"MJPG"),
            info.fps,
            (info.width, info.height),
        )

        total_frames = int(
            min(
                duration,
                max(0.0, info.duration - start_time),
            )
            * info.fps
        )

        tracked_frames = 0
        failed_frames = 0

        for frame_index in range(total_frames):

            ok, frame = cap.read()

            if not ok:
                break

            success, tracked_box = tracker.update(
                frame
            )

            if success:
                x, y, w, h = [
                    int(v)
                    for v in tracked_box
                ]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 170),
                    4,
                )

                cv2.putText(
                    frame,
                    f"{best['label']}  {best['score']:.2f}",
                    (x, max(35, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 170),
                    2,
                    cv2.LINE_AA,
                )

                center_x = x + w // 2
                center_y = y + h // 2

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    7,
                    (0, 255, 170),
                    -1,
                )

                tracked_frames += 1

            else:
                failed_frames += 1

                cv2.putText(
                    frame,
                    "TRACK LOST",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            writer.write(frame)

        cap.release()
        writer.release()

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temp_avi),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]

        subprocess.run(
            command,
            check=True,
        )

        temp_avi.unlink(
            missing_ok=True
        )

        return {
            "output": str(output),
            "label": best["label"],
            "detection_score": best["score"],
            "initial_box": best["box"],
            "tracked_frames": tracked_frames,
            "failed_frames": failed_frames,
            "total_frames": tracked_frames + failed_frames,
        }
