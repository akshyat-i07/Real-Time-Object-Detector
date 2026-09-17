# Project Statement

## Problem Statement
Manually reviewing camera footage or large batches of images to identify
objects (people, vehicles, animals, everyday items) is slow, error-prone,
and does not scale. There is a need for a lightweight tool that can
automatically detect and label common objects in real time from a
webcam, a recorded video, or a folder of images, and summarize what it
found — without requiring specialized hardware or a cloud subscription.

## Scope
This project delivers a **Python, OpenCV + deep-learning based real-time
object detection application** that:

- Detects and localizes 80 everyday object classes (COCO dataset:
  person, car, bicycle, dog, chair, laptop, phone, etc.)
- Works on three kinds of input: a live webcam feed, a pre-recorded
  video file, or a folder of still images
- Draws bounding boxes and confidence scores on the output
- Logs every detection and produces a per-session summary report

Out of scope: training a custom model from scratch, multi-camera fusion,
and cloud deployment — these are listed under Future Enhancements in the
project report.

## Target Users
- Students/researchers who want a ready-made baseline to experiment with
  object detection.
- Small setups (a shop, a hobby robotics project, a home security
  camera) that want simple "what did the camera see" analytics without
  a paid service.
- Anyone who wants to batch-annotate a folder of images with detected
  objects.

## High-Level Features
1. **Detection Engine** — loads a pretrained YOLOv8-nano model and runs
   inference on any given frame, returning structured `Detection`
   objects (label, confidence, bounding box).
2. **Flexible Input Handling** — a common interface for webcam, video
   file, and image-folder sources, with input validation for file type
   and existence.
3. **Reporting & Analytics** — logs every detection to CSV and produces
   a plain-text session summary (object counts, most frequent classes,
   average detections per frame).
4. **Live Visualization** — bounding boxes, labels, and an FPS counter
   drawn directly on the video/image preview, with an option to save
   annotated frames to disk.
5. **CLI Interface** — a single `main.py` entry point with flags for
   source type, display mode, and saving results, so the whole app can
   run headless on a server or interactively on a laptop.
