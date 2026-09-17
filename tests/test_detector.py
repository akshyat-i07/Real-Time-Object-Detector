"""
Unit tests for src/detector.py.

The real YOLO model is heavy and needs a network connection on first
run, so these tests mock it out and focus on ObjectDetector's own logic:
input validation and result parsing.
"""

import numpy as np
import pytest

from src.detector import Detection, DetectorError, ObjectDetector


class _FakeBox:
    """Mimics one ultralytics `Boxes` entry."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy)]


class _FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _FakeModel:
    """Stands in for `ultralytics.YOLO` so tests run without a real model."""

    def predict(self, source, conf, iou, verbose):
        names = {0: "person", 1: "car"}
        boxes = [_FakeBox(0, 0.91, (10, 10, 50, 50))]
        return [_FakeResult(names, boxes)]


@pytest.fixture
def detector():
    det = ObjectDetector.__new__(ObjectDetector)  # bypass real model loading
    det.confidence_threshold = 0.4
    det.iou_threshold = 0.45
    det.model = _FakeModel()
    return det


def test_detect_returns_detection_objects(detector):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    results = detector.detect(frame)

    assert len(results) == 1
    assert isinstance(results[0], Detection)
    assert results[0].label == "person"
    assert results[0].confidence == pytest.approx(0.91)
    assert results[0].box == (10, 10, 50, 50)


def test_detect_rejects_empty_frame(detector):
    with pytest.raises(DetectorError):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
