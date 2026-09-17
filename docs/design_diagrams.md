# Design & Documentation

This file contains the required design artefacts for the project:
Problem Statement / Objectives (see `statement.md`), Functional &
Non-Functional Requirements, System Architecture, Workflow, and UML
diagrams. Diagrams are written in Mermaid syntax — they render
automatically on GitHub and in most Markdown viewers (VS Code, Obsidian,
etc.).

## Objectives
- Detect common real-world objects accurately enough for everyday use
  (person, vehicle, animal, everyday items) using a pretrained model.
- Support three input modes (webcam / video file / image folder)
  through one consistent interface.
- Produce an auditable record (CSV + summary) of what was detected in
  a session, not just a live preview.
- Keep the codebase modular enough that the detection model, the input
  source, or the reporting format can each be swapped independently.

## Functional Requirements
| # | Requirement |
|---|-------------|
| FR1 | The system shall detect and classify objects in a frame using a pretrained deep-learning model, returning label, confidence, and bounding box for each detection. |
| FR2 | The system shall accept input from a webcam, a video file, or a folder of images through a single unified interface. |
| FR3 | The system shall log every detection (frame id, label, confidence, box coordinates) to a CSV file and produce a human-readable summary at the end of a run. |
| FR4 | The system shall render bounding boxes, class labels, confidence scores and a live FPS counter on the output frame. |
| FR5 | The system shall allow the user to save annotated frames/images to disk via a command-line flag. |

## Non-Functional Requirements
| # | Category | Requirement |
|---|----------|-------------|
| NFR1 | **Performance** | The system should sustain real-time-comparable throughput on CPU (nano model, frame resizing) and reports FPS live so the user can judge performance on their machine. |
| NFR2 | **Reliability** | Camera/video/model failures raise clear, caught exceptions (`DetectorError`, `InputSourceError`) instead of crashing with a raw traceback; a single bad frame is skipped, not fatal. |
| NFR3 | **Security** | All file-based inputs are validated against an extension allow-list and existence check before being opened (`_validate_path`), preventing accidental processing of arbitrary/unsafe files. |
| NFR4 | **Usability** | A single CLI (`main.py`) with sensible defaults and `--help` covers all three input modes; live preview window shows FPS and labels directly. |
| NFR5 | **Maintainability** | Each concern (detection, input handling, drawing, reporting, logging, configuration) lives in its own module with a single responsibility. |
| NFR6 | **Logging/Monitoring** | A shared logger (`logger_setup.py`) writes timestamped, leveled logs to both console and `logs/app.log` for every module. |
| NFR7 | **Scalability** | New input types (e.g. an RTSP network stream) or a different model (e.g. a larger YOLO variant) can be added by extending `InputSource` or swapping `MODEL_NAME`, without touching other modules. |
| NFR8 | **Resource efficiency** | Frame size is capped (`FRAME_WIDTH`/`FRAME_HEIGHT`), a nano model is used by default, and a `PROCESS_EVERY_N_FRAMES` config option allows frame-skipping on slower hardware. |

## System Architecture Diagram
```mermaid
flowchart LR
    subgraph Input Layer
        A1[Webcam]
        A2[Video File]
        A3[Image Folder]
    end

    subgraph Core Application
        B[InputSource
        input_handler.py]
        C[ObjectDetector
        detector.py]
        D[Visualizer
        visualizer.py]
        E[ReportGenerator
        report_generator.py]
        F[main.py
        orchestrator / CLI]
    end

    subgraph Output Layer
        G[Live Preview Window]
        H[Annotated Images
        outputs/annotated]
        I[CSV + Summary Report
        outputs/reports]
    end

    A1 --> B
    A2 --> B
    A3 --> B
    B --> F
    F --> C
    C --> F
    F --> D
    F --> E
    D --> G
    D --> H
    E --> I
```

## Process Flow / Workflow Diagram
```mermaid
flowchart TD
    Start([Start]) --> Parse[Parse CLI arguments]
    Parse --> LoadModel[Load YOLOv8 model]
    LoadModel -->|failure| ErrLog1[Log error and exit]
    LoadModel -->|success| OpenSource[Open input source
    webcam / video / folder]
    OpenSource -->|failure| ErrLog2[Log error and exit]
    OpenSource -->|success| Loop{More frames?}
    Loop -->|yes| GetFrame[Read next frame]
    GetFrame --> Detect[Run detection]
    Detect --> Log[Log detections to report]
    Log --> Draw[Draw boxes + FPS]
    Draw --> SaveCheck{--save flag?}
    SaveCheck -->|yes| SaveFrame[Write annotated frame to disk]
    SaveCheck -->|no| ShowCheck
    SaveFrame --> ShowCheck{--no-display?}
    ShowCheck -->|no| Display[Show preview window]
    ShowCheck -->|yes| Loop
    Display --> QuitCheck{'q' pressed?}
    QuitCheck -->|yes| Finalize
    QuitCheck -->|no| Loop
    Loop -->|no more frames| Finalize[Finalize report:
    write CSV + summary]
    Finalize --> End([End])
```

## Use Case Diagram
```mermaid
flowchart LR
    User((User))
    UC1([Start detection on webcam])
    UC2([Run detection on a video file])
    UC3([Run detection on an image folder])
    UC4([View live annotated preview])
    UC5([Save annotated output])
    UC6([View session summary report])

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
```

## Class Diagram
```mermaid
classDiagram
    class Detection {
        +str label
        +float confidence
        +tuple box
    }

    class ObjectDetector {
        -model
        -confidence_threshold
        -iou_threshold
        +detect(frame) List~Detection~
    }

    class InputSource {
        <<abstract>>
        +frames() Iterator
        +release()
    }
    class WebcamSource
    class VideoFileSource
    class ImageFolderSource

    InputSource <|-- WebcamSource
    InputSource <|-- VideoFileSource
    InputSource <|-- ImageFolderSource

    class ReportGenerator {
        -csv_path
        -summary_path
        +log_frame(frame_id, detections)
        +finalize() str
    }

    class MainApp {
        +main()
    }

    MainApp --> ObjectDetector : uses
    MainApp --> InputSource : uses
    MainApp --> ReportGenerator : uses
    ObjectDetector --> Detection : creates
    ReportGenerator --> Detection : consumes
```

## Sequence Diagram (one frame, webcam mode)
```mermaid
sequenceDiagram
    participant U as User
    participant M as main.py
    participant I as InputSource
    participant D as ObjectDetector
    participant R as ReportGenerator
    participant V as Visualizer

    U->>M: python main.py --source webcam
    M->>D: ObjectDetector()
    M->>I: build_input_source("webcam", 0)
    loop for each frame
        M->>I: frames() -> (frame_id, frame)
        M->>D: detect(frame)
        D-->>M: List[Detection]
        M->>R: log_frame(frame_id, detections)
        M->>V: draw_detections(frame, detections)
        V-->>M: annotated frame
        M->>U: show preview window
    end
    U->>M: press 'q'
    M->>R: finalize()
    R-->>M: summary_path
    M->>I: release()
```

## Note on Database/Storage Design
This project does not use a database. Persistent output is file-based
(CSV detection logs and text summaries under `outputs/reports/`), which
is sufficient for the project's scope. If a future enhancement added
multi-session history/search, an ER diagram and schema would be added
here at that point (see `Future Enhancements` in the report).
