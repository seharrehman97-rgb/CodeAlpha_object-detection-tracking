# Vision Tracker: Real-Time Object Detection and Tracking

Detects and tracks people, vehicles and everyday objects in video (file or webcam), gives every object a persistent ID, and produces a report of when each object appeared and left. Built to run on a **CPU only** (no GPU needed).

## Features

- Real-time video input from a webcam or a video file (OpenCV)
- Pre-trained **YOLO11** detection (Ultralytics, COCO classes), with per-frame bounding boxes and labels
- **Deep SORT** tracking with persistent IDs, shown live on the video
- Two appearance modes for the tracker: a fast colour histogram (default) and the MobileNet CNN embedder
- Live telemetry on screen: FPS, detection time, tracking time
- Analytics: motion trails, a virtual **line-crossing counter** (up / down, per class), and a **restricted zone** with a live dwell-time timer
- **CSV report** with the time each object appeared and left, duration, zone entry and line crossing
- **Streamlit web app**: upload a video or use the webcam, choose the analysis duration, pick classes and a speed preset, view a summary and download the CSV
- Unit tests for the analytics and report logic (`pytest`)

## How it works

```
Video source -> YOLO detector -> Deep SORT tracker -> analytics -> annotated frames + CSV report
(file/webcam)   (boxes, labels)   (persistent IDs)    (trails, line counter, zone, event log)
```

The processing lives in one reusable `Pipeline` class, shared by the command-line tool and the web app.

## Project structure

```
object-detection-tracking/
├── app.py                 # Streamlit web app
├── main.py                # command-line entry point
├── src/
│   ├── video_source.py    # webcam / file input, resizing
│   ├── detector.py        # YOLO wrapper
│   ├── tracker.py         # Deep SORT wrapper (histogram or CNN appearance)
│   ├── analytics.py       # trails, line counter, zone monitor, event log
│   ├── visualizer.py      # drawing boxes, IDs, HUD
│   └── pipeline.py        # ties everything together
├── scripts/benchmark.py   # reproducible speed benchmark
├── tests/                 # unit tests
├── docs/benchmark.md      # benchmark results
├── assets/                # stylesheet and artwork for the web app
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/seharrehman97-rgb/CodeAlpha_object-detection-tracking.git
cd CodeAlpha_object-detection-tracking
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

The YOLO weights (`yolo11n.pt`) download automatically into `models/` on the first run. Sample videos are not included: put your own clip in `data/`.

## Usage

**Command line**

```bash
python main.py --source data/street.mp4 --seconds 10 --zone 0.3 0.55 0.75 1.0 --csv outputs/events.csv
python main.py --source 0                       # webcam
python main.py --source data/street.mp4 --classes 0 24 26 67   # person, backpack, handbag, cell phone
```

Press `q` to quit. Useful options: `--imgsz 480` (faster), `--tracker mobilenet` (CNN appearance), `--min-duration 1` (hide flickering tracks in the report).

**Web app**

```bash
streamlit run app.py
```

**Tests**

```bash
python -m pytest tests -q
```

## Benchmark (CPU only, first 150 frames, 960 px wide, headless)

| Configuration | Avg FPS | Detect (ms) | Track (ms) | Unique IDs |
|---|---|---|---|---|
| Baseline (YOLO11n, 640, histogram) | 6.47 | 101 | 13 | 19 |
| Smaller input (YOLO11n, 480, histogram) | 9.07 | 82 | 15 | 14 |
| Bigger model (YOLO11s, 640, histogram) | 4.49 | 190 | 19 | 34 |
| CNN embedder (YOLO11n, 640, mobilenet) | 1.33 | 265 | 479 | 20 |

Reproduce with `python scripts/benchmark.py`.

**Findings**

- The Deep SORT CNN embedder costs about 37x more tracking time than the colour histogram (479 ms vs 13 ms) with no clear stability gain on this clip, so the histogram is the default.
- A smaller YOLO input is the cheapest speed-up (about 40% faster).
- The larger YOLO model roughly doubles detection time.
- "Unique IDs" is only a rough proxy for tracking stability: a lower count can also mean fewer objects were detected.

## Limitations

- Detection is limited to the 80 COCO classes; small objects such as phones are hard to detect in wide scenes with the nano model.
- The colour-histogram tracker can give the same person a new ID after long occlusions, so appear/leave times and line-crossing counts are approximate in crowded scenes.
- I tried skipping detection on alternate frames and letting the tracker predict; it broke track confirmation, so it was dropped.
- Webcam capture in the web app works when the app runs on the same computer as the camera.
- Ultralytics YOLO is licensed under AGPL-3.0; this project is for learning and demonstration.

## Future work

- ONNX / OpenVINO export for faster CPU inference
- ByteTrack comparison
- Fine-tuning on custom classes, or an open-vocabulary detector
- Demo GIF and ground-truth evaluation of ID switches