"""Unit tests for src/report_generator.py."""

import csv
import os

from src.detector import Detection
from src.report_generator import ReportGenerator


def test_log_frame_and_finalize_creates_files(tmp_path, monkeypatch):
    monkeypatch.setattr("src.report_generator.REPORTS_DIR", str(tmp_path))
    reporter = ReportGenerator(run_name="unittest")

    detections = [
        Detection(label="person", confidence=0.9, box=(0, 0, 10, 10)),
        Detection(label="car", confidence=0.8, box=(5, 5, 20, 20)),
    ]
    reporter.log_frame(0, detections)
    reporter.log_frame(1, [])

    summary_path = reporter.finalize()

    assert os.path.exists(reporter.csv_path)
    assert os.path.exists(summary_path)

    with open(reporter.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # header + 2 detection rows
    assert len(rows) == 3
    assert rows[0] == ["frame_id", "label", "confidence", "x1", "y1", "x2", "y2"]

    with open(summary_path, encoding="utf-8") as f:
        content = f.read()
    assert "Frames processed      : 2" in content
    assert "Total detections      : 2" in content
