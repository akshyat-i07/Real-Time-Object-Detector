"""
report_generator.py
--------------------
FUNCTIONAL MODULE 3: Reporting & Analytics

Collects detection results across a whole run (webcam session, video
file, or image folder) and turns them into:
  * a per-frame CSV log (outputs/reports/detections_<timestamp>.csv)
  * a plain-text summary (object counts, most common object, avg per frame)

This is what lets a user answer "what did the app actually see?"
without re-watching the whole video.
"""

import csv
import os
from collections import Counter
from datetime import datetime
from typing import List

from config import REPORTS_DIR
from src.detector import Detection
from src.logger_setup import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, run_name: str = "session"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(REPORTS_DIR, f"{run_name}_{timestamp}.csv")
        self.summary_path = os.path.join(REPORTS_DIR, f"{run_name}_{timestamp}_summary.txt")

        self._label_counter: Counter = Counter()
        self._frame_count = 0
        self._detection_count = 0

        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["frame_id", "label", "confidence", "x1", "y1", "x2", "y2"])

    def log_frame(self, frame_id: int, detections: List[Detection]) -> None:
        """Record one frame's worth of detections."""
        self._frame_count += 1
        for det in detections:
            self._detection_count += 1
            self._label_counter[det.label] += 1
            x1, y1, x2, y2 = det.box
            self._csv_writer.writerow(
                [frame_id, det.label, f"{det.confidence:.3f}", x1, y1, x2, y2]
            )

    def finalize(self) -> str:
        """Close the CSV, write a human-readable summary, and return its path."""
        self._csv_file.close()

        avg_per_frame = (
            self._detection_count / self._frame_count if self._frame_count else 0
        )
        most_common = self._label_counter.most_common(5)

        lines = [
            "Object Detection Session Summary",
            "=================================",
            f"Frames processed      : {self._frame_count}",
            f"Total detections      : {self._detection_count}",
            f"Average detections/fr : {avg_per_frame:.2f}",
            "",
            "Top detected object classes:",
        ]
        if most_common:
            for label, count in most_common:
                lines.append(f"  - {label}: {count}")
        else:
            lines.append("  (no objects detected)")

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info("Report written to %s", self.csv_path)
        logger.info("Summary written to %s", self.summary_path)
        return self.summary_path
