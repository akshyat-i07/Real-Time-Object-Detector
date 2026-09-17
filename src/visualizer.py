"""
visualizer.py
-------------
Small helper module: draws bounding boxes, labels and an FPS counter on
a frame. Kept separate from detector.py so the detection engine stays
free of any drawing/UI concerns (single-responsibility).
"""

from typing import List

import cv2
import numpy as np

from src.detector import Detection

_BOX_COLOR = (0, 200, 0)
_TEXT_COLOR = (255, 255, 255)


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Return a copy of `frame` with bounding boxes + labels drawn on it."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)

        label = f"{det.label} {det.confidence * 100:.0f}%"
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), _BOX_COLOR, -1
        )
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA,
        )
    return annotated


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay an FPS counter in the top-left corner (Performance visibility)."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return frame
