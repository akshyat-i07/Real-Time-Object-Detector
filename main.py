"""
main.py
-------
Entry point for the Real-Time Object Detection App.

Usage examples
--------------
Webcam (live):
    python main.py --source webcam --camera 0

Video file:
    python main.py --source video --path data/sample_videos/street.mp4

Folder of images:
    python main.py --source folder --path data/sample_images

Add --no-display to run headless (e.g. on a server) and --save to write
annotated frames/images to outputs/annotated/.
"""

import argparse
import os
import time

import cv2

from config import ANNOTATED_DIR
from src.detector import DetectorError, ObjectDetector
from src.input_handler import InputSourceError, build_input_source
from src.logger_setup import get_logger
from src.report_generator import ReportGenerator
from src.visualizer import draw_detections, draw_fps

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-Time Object Detection App")
    parser.add_argument(
        "--source", choices=["webcam", "video", "folder"], default="folder",
        help="Type of input source (default: folder)",
    )
    parser.add_argument(
        "--path", default="data/sample_images",
        help="Path to video file or image folder (ignored for webcam)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not open a preview window (useful on headless machines)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated frames/images to outputs/annotated/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ---- Module 1: Detection engine -----------------------------------
    try:
        detector = ObjectDetector()
    except DetectorError as exc:
        logger.error(str(exc))
        return 1

    # ---- Module 2: Input source ----------------------------------------
    source_value = args.camera if args.source == "webcam" else args.path
    try:
        input_source = build_input_source(args.source, source_value)
    except InputSourceError as exc:
        logger.error(str(exc))
        return 1

    # ---- Module 3: Reporting -------------------------------------------
    reporter = ReportGenerator(run_name=args.source)

    frame_times = []
    processed = 0

    try:
        for frame_id, frame in input_source.frames():
            start = time.time()

            try:
                detections = detector.detect(frame)
            except DetectorError as exc:
                logger.warning("Skipping frame %d: %s", frame_id, exc)
                continue

            reporter.log_frame(frame_id, detections)
            annotated = draw_detections(frame, detections)

            elapsed = time.time() - start
            frame_times.append(elapsed)
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            annotated = draw_fps(annotated, fps)

            if args.save:
                out_path = os.path.join(ANNOTATED_DIR, f"frame_{frame_id:05d}.jpg")
                cv2.imwrite(out_path, annotated)

            if not args.no_display:
                cv2.imshow("Real-Time Object Detection", annotated)
                # 'q' quits, works for both live video and image folder mode
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit key pressed - stopping.")
                    break

            processed += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        input_source.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        summary_path = reporter.finalize()

    avg_fps = (1.0 / (sum(frame_times) / len(frame_times))) if frame_times else 0.0
    logger.info("Processed %d frame(s). Average FPS: %.2f", processed, avg_fps)
    logger.info("See summary at: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
