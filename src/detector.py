"""
detector.py
-----------
FUNCTIONAL MODULE 1: Detection Engine

Responsible for loading the object-detection model and running inference
on a single frame/image. This module knows nothing about where frames
come from (webcam, video, folder) or what happens to the results
afterwards -- that separation of concerns is what keeps the codebase
modular and maintainable.

Model: YOLOv8-nano (ultralytics). Chosen because it:
  * runs acceptably fast on CPU only (no GPU required),
  * is downloaded automatically the first time it's used,
  * ships with 80 everyday COCO classes (person, car, dog, phone, ...).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MODEL_NAME
from src.logger_setup import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """A single detected object, in a plain, storage-friendly shape."""
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixel coordinates


class DetectorError(Exception):
    """Raised when the detection engine cannot load or run the model."""


class ObjectDetector:
    """Wraps a YOLOv8 model and exposes a simple `.detect(frame)` API."""

    def __init__(self, model_name: str = MODEL_NAME,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 iou_threshold: float = IOU_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load the YOLO model, raising a clear, catchable error on failure.

        This is the app's main "Reliability" safeguard: a bad/missing
        model file should not crash the whole program with a stack trace
        the user can't interpret.
        """
        try:
            from ultralytics import YOLO  # imported lazily so the rest of
            logger.info("Loading model '%s' ...", model_name)          # the app can be unit-tested without ultralytics installed
            model = YOLO(model_name)
            logger.info("Model loaded successfully.")
            return model
        except Exception as exc:  # noqa: BLE001 - we deliberately re-wrap
            logger.error("Failed to load model '%s': %s", model_name, exc)
            raise DetectorError(
                f"Could not load detection model '{model_name}'. "
                f"Check your internet connection (first run downloads the "
                f"model) or verify the model file. Original error: {exc}"
            ) from exc

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame and return Detection objects."""
        if frame is None or frame.size == 0:
            raise DetectorError("Received an empty frame - nothing to detect.")

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=names.get(cls_id, str(cls_id)),
                        confidence=confidence,
                        box=(x1, y1, x2, y2),
                    )
                )

        logger.debug("Detected %d object(s) in frame.", len(detections))
        return detections
