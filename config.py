"""
config.py
---------
Central configuration for the Real-Time Object Detection App.

Keeping configuration in one place (instead of scattering magic numbers
across modules) directly supports the Maintainability and Scalability
non-functional requirements described in statement.md.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for _dir in (ANNOTATED_DIR, REPORTS_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Model settings
# ----------------------------------------------------------------------
# Nano model chosen deliberately: it is small (~6MB), CPU-friendly, and
# needs no manual weight download -- ultralytics fetches it automatically
# the first time it is used. This keeps "Resource efficiency" reasonable
# on a machine with no GPU.
MODEL_NAME = "yolov8n.pt"

# Minimum confidence for a detection to be kept. Filters out noisy /
# low-confidence boxes (helps both accuracy and performance).
CONFIDENCE_THRESHOLD = 0.4

# Intersection-over-Union threshold used by the model's own
# Non-Max-Suppression step.
IOU_THRESHOLD = 0.45

# ----------------------------------------------------------------------
# Video / camera settings
# ----------------------------------------------------------------------
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Skip N-1 out of every N frames when running on live video to keep the
# app responsive on modest hardware (Performance / Resource efficiency).
PROCESS_EVERY_N_FRAMES = 1

# ----------------------------------------------------------------------
# Supported file types (input validation - Security)
# ----------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = "INFO"
