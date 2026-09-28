from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path("data/chronolens_demo.mp4")
W, H = 1280, 720
FPS = 30
SEGMENT_SECONDS = 6
SEGMENTS = [
    ("RED CAR", (20, 20, 220), "car"),
    ("BLUE BICYCLE", (220, 80, 20), "bicycle"),
    ("BLACK BACKPACK", (25, 25, 25), "backpack"),
    ("GREEN BALL", (20, 180, 70), "ball"),
]


def draw_scene(frame, label, color, kind, local_t):
    cv2.putText(
        frame, "ChronoLens synthetic systems test", (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (245, 245, 245), 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, label, (40, 135),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA
    )

    x = int(120 + local_t / SEGMENT_SECONDS * 900)
    y = int(360 + 80 * math.sin(local_t * 1.6))

    if kind == "car":
        cv2.rectangle(frame, (x, y), (x + 240, y + 95), color, -1)
        cv2.circle(frame, (x + 50, y + 105), 30, (210, 210, 210), -1)
        cv2.circle(frame, (x + 190, y + 105), 30, (210, 210, 210), -1)
    elif kind == "bicycle":
        cv2.circle(frame, (x, y), 70, color, 10)
        cv2.circle(frame, (x + 190, y), 70, color, 10)
        cv2.line(frame, (x, y), (x + 90, y - 95), color, 10)
        cv2.line(frame, (x + 90, y - 95), (x + 190, y), color, 10)
        cv2.line(frame, (x, y), (x + 115, y), color, 10)
    elif kind == "backpack":
        cv2.rectangle(frame, (x, y - 130), (x + 160, y + 120), color, -1)
        cv2.rectangle(frame, (x + 35, y - 165), (x + 125, y - 115), color, 12)
        cv2.rectangle(frame, (x + 30, y + 20), (x + 130, y + 90), (90, 90, 90), -1)
    else:
        cv2.circle(frame, (x, y), 85, color, -1)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    for label, color, kind in SEGMENTS:
        for i in range(SEGMENT_SECONDS * FPS):
            t = i / FPS
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:] = (8, 14, 25)
            draw_scene(frame, label, color, kind, t)
            writer.write(frame)
    writer.release()
    print(OUT.resolve())


if __name__ == "__main__":
    main()
