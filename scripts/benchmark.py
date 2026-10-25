import argparse
import statistics
import sys
import time
from pathlib import Path

# let this script import from the project's src/ package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import Detector
from src.tracker import Tracker
from src.video_source import VideoSource

CLASSES = [0, 1, 2, 3, 5, 7]  # person and vehicles

CONFIGS = [
    {"name": "Baseline (YOLO11n, 640, histogram)",
     "model": "models/yolo11n.pt", "imgsz": 640, "tracker": "histogram"},
    {"name": "Smaller input (YOLO11n, 480, histogram)",
     "model": "models/yolo11n.pt", "imgsz": 480, "tracker": "histogram"},
    {"name": "Bigger model (YOLO11s, 640, histogram)",
     "model": "models/yolo11s.pt", "imgsz": 640, "tracker": "histogram"},
    {"name": "CNN embedder (YOLO11n, 640, mobilenet)",
     "model": "models/yolo11n.pt", "imgsz": 640, "tracker": "mobilenet"},
]


def run(cfg, source, width, max_frames):
    detector = Detector(cfg["model"], 0.4, CLASSES, cfg["imgsz"])
    tracker = Tracker(mode=cfg["tracker"])

    det_times, trk_times, ids = [], [], set()
    frames = 0
    start = time.time()

    with VideoSource(source, width) as video:
        while frames < max_frames:
            ok, frame = video.read()
            if not ok:
                break
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            t1 = time.perf_counter()
            tracks = tracker.update(detections, frame)
            t2 = time.perf_counter()

            det_times.append(t1 - t0)
            trk_times.append(t2 - t1)
            ids.update(t["track_id"] for t in tracks)
            frames += 1

    elapsed = time.time() - start
    # skip the first frame in timings: it includes one-time warm-up cost
    return {
        "name": cfg["name"],
        "fps": frames / elapsed,
        "detect_ms": statistics.mean(det_times[1:]) * 1000,
        "track_ms": statistics.mean(trk_times[1:]) * 1000,
        "ids": len(ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark detector/tracker setups")
    parser.add_argument("--source", default="data/street.mp4")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=150)
    args = parser.parse_args()

    rows = []
    for cfg in CONFIGS:
        print(f"Running: {cfg['name']} ...")
        rows.append(run(cfg, args.source, args.width, args.max_frames))

    lines = [
        f"Benchmark on {args.source}, first {args.max_frames} frames, "
        f"width {args.width}, CPU only",
        "",
        "| Configuration | Avg FPS | Detect (ms) | Track (ms) | Unique IDs |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['fps']:.2f} | {r['detect_ms']:.0f} "
                     f"| {r['track_ms']:.0f} | {r['ids']} |")
    table = "\n".join(lines)

    print("\n" + table)
    out = Path("docs")
    out.mkdir(exist_ok=True)
    (out / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    print("\nSaved to docs/benchmark.md")


if __name__ == "__main__":
    main()