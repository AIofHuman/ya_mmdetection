from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2


def run_video_inference_opencv(
    input_video: str | Path,
    output_video: str | Path,
    frame_predictor: Callable,
):
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            predicted = frame_predictor(frame)
            writer.write(predicted)
    finally:
        cap.release()
        writer.release()

    return output_video
