"""
input_handler.py
-----------------
FUNCTIONAL MODULE 2: Input Handling

Provides a single, uniform way to iterate over frames regardless of
whether the source is a live webcam, a video file, or a folder of
still images. Consumers (main.py) just do:

    for frame_id, frame in input_source.frames():
        ...

Input validation lives here (Security non-functional requirement):
paths are checked against an allow-list of extensions and existence
before anything is opened.
"""

import os
from typing import Iterator, Tuple

import cv2
import numpy as np

from config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
from src.logger_setup import get_logger

logger = get_logger(__name__)


class InputSourceError(Exception):
    """Raised for any problem opening or reading an input source."""


class InputSource:
    """Base class - subclasses implement `frames()`."""

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class WebcamSource(InputSource):
    """Live camera feed."""

    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise InputSourceError(
                f"Could not open webcam at index {camera_index}. "
                f"Is a camera connected and not in use by another app?"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        logger.info("Webcam %d opened.", camera_index)

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        frame_id = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                logger.warning("Webcam read failed; stopping stream.")
                break
            yield frame_id, frame
            frame_id += 1

    def release(self) -> None:
        self.cap.release()
        logger.info("Webcam released.")


class VideoFileSource(InputSource):
    """A pre-recorded video file on disk."""

    def __init__(self, path: str):
        _validate_path(path, ALLOWED_VIDEO_EXTENSIONS)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise InputSourceError(f"Could not open video file: {path}")
        logger.info("Video file opened: %s", path)

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        frame_id = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame_id, frame
            frame_id += 1

    def release(self) -> None:
        self.cap.release()


class ImageFolderSource(InputSource):
    """A directory of still images, each treated as a single 'frame'."""

    def __init__(self, folder: str):
        if not os.path.isdir(folder):
            raise InputSourceError(f"Not a valid directory: {folder}")
        self.folder = folder
        self.image_paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise InputSourceError(
                f"No supported images ({ALLOWED_IMAGE_EXTENSIONS}) found in {folder}"
            )
        logger.info("Found %d image(s) in %s", len(self.image_paths), folder)

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        for frame_id, path in enumerate(self.image_paths):
            frame = cv2.imread(path)
            if frame is None:
                logger.warning("Skipping unreadable image: %s", path)
                continue
            yield frame_id, frame

    def release(self) -> None:
        pass


def _validate_path(path: str, allowed_extensions: set) -> None:
    """Basic input validation: existence + extension allow-list."""
    if not os.path.isfile(path):
        raise InputSourceError(f"File does not exist: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in allowed_extensions:
        raise InputSourceError(
            f"Unsupported file extension '{ext}'. Allowed: {allowed_extensions}"
        )


def build_input_source(source_type: str, source_value) -> InputSource:
    """Factory function used by main.py to construct the right source."""
    if source_type == "webcam":
        return WebcamSource(int(source_value))
    if source_type == "video":
        return VideoFileSource(source_value)
    if source_type == "folder":
        return ImageFolderSource(source_value)
    raise InputSourceError(f"Unknown source type: {source_type}")
