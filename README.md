# Real-Time Object Detection App

A lightweight Python application that detects and labels everyday
objects (people, vehicles, animals, everyday items — 80 COCO classes)
in real time from a **webcam**, a **video file**, or a **folder of
images**, and produces a CSV + summary report of everything it saw.

Built as a course project in the **Computer Vision** domain.

## Overview
The app wraps a pretrained YOLOv8-nano model behind a small, modular
Python codebase with three functional modules:

1. **Detection Engine** (`src/detector.py`) — loads the model and runs
   inference on a frame.
2. **Input Handling** (`src/input_handler.py`) — one interface for
   webcam / video file / image folder sources, with input validation.
3. **Reporting & Analytics** (`src/report_generator.py`) — logs every
   detection to CSV and writes a plain-text session summary.

A visualization helper (`src/visualizer.py`) draws bounding boxes,
labels, and an FPS counter; `main.py` is the CLI that ties everything
together. See `docs/design_diagrams.md` for architecture, workflow, and
UML diagrams, and `statement.md` for the problem statement and scope.

## Features
- Real-time detection from a live webcam
- Batch detection over a video file or a folder of still images
- Bounding boxes + confidence scores + live FPS overlay
- Per-session CSV detection log and a text summary report
- Optional saving of annotated frames to disk
- Centralized logging (console + `logs/app.log`)
- Clear, catchable errors instead of raw crashes (bad camera, missing
  file, unreadable image, etc.)
- Unit tests for all three functional modules (mocked model — no GPU or
  network access needed to run the test suite)

## Technologies / Tools Used
- Python 3.9+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (nano
  model, CPU-friendly)
- OpenCV (`opencv-python`) — capture, drawing, display
- NumPy
- pytest — unit testing

## Project Structure
```
realtime-object-detection/
├── main.py                    # CLI entry point
├── config.py                  # central configuration
├── requirements.txt
├── statement.md                # problem statement / scope / users
├── src/
│   ├── detector.py             # Module 1: detection engine
│   ├── input_handler.py        # Module 2: input handling
│   ├── report_generator.py     # Module 3: reporting & analytics
│   ├── visualizer.py           # drawing helper
│   └── logger_setup.py         # shared logging config
├── tests/
│   ├── test_detector.py
│   ├── test_input_handler.py
│   └── test_report_generator.py
├── data/sample_images/         # put test images here
├── outputs/
│   ├── annotated/               # saved annotated frames (--save)
│   └── reports/                 # CSV + summary reports
└── docs/
    └── design_diagrams.md       # architecture, workflow & UML diagrams
```

## Steps to Install & Run

### 1. Clone and set up a virtual environment
```bash
git clone <your-repo-url>
cd realtime-object-detection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run on a folder of images (no camera needed)
```bash
python main.py --source folder --path data/sample_images --no-display --save
```

### 4. Run on your webcam (press 'q' to quit)
```bash
python main.py --source webcam --camera 0
```

### 5. Run on a video file
```bash
python main.py --source video --path path/to/video.mp4
```

> The first run downloads the YOLOv8-nano weights automatically
> (requires an internet connection once); after that it works offline.

### Output
- Annotated frames (if `--save` is used): `outputs/annotated/`
- Detection log + summary: `outputs/reports/`
- App logs: `logs/app.log`

## Instructions for Testing
Run the full unit test suite (uses a mocked model, so it needs no GPU
or network access):
```bash
pip install pytest
pytest tests/ -v
```
