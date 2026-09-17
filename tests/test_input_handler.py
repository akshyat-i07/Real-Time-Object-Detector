"""Unit tests for src/input_handler.py."""

import os

import cv2
import numpy as np
import pytest

from src.input_handler import (
    ImageFolderSource,
    InputSourceError,
    build_input_source,
)


@pytest.fixture
def image_folder(tmp_path):
    """Create a temp folder with 2 valid images and 1 unsupported file."""
    folder = tmp_path / "images"
    folder.mkdir()

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(folder / "a.jpg"), img)
    cv2.imwrite(str(folder / "b.png"), img)
    (folder / "notes.txt").write_text("not an image")

    return str(folder)


def test_image_folder_source_finds_only_valid_images(image_folder):
    source = ImageFolderSource(image_folder)
    assert len(source.image_paths) == 2


def test_image_folder_source_yields_frames(image_folder):
    source = ImageFolderSource(image_folder)
    frames = list(source.frames())
    assert len(frames) == 2
    for frame_id, frame in frames:
        assert isinstance(frame_id, int)
        assert frame.shape == (20, 20, 3)


def test_image_folder_source_rejects_missing_dir():
    with pytest.raises(InputSourceError):
        ImageFolderSource("/path/does/not/exist")


def test_build_input_source_rejects_unknown_type():
    with pytest.raises(InputSourceError):
        build_input_source("carrier_pigeon", "irrelevant")
