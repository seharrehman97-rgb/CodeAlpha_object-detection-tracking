import argparse
import time

import cv2

from src.analytics import LineCounter, TrailManager
from src.detector import Detector
from src.tracker import Tracker
from src.video_source import VideoSource
from src.visualizer import draw_counter, draw_tracks, draw_trails


def parse_args():
    parser = argparse.ArgumentParser(description="Object detection and tracking")
    parser.add_argument("--source", default="0",
                        help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--save", default=None,
                        help="Optional path to save the output video")
    parser.add_argument("--model", default="models/yolo11n.pt",
                        help="Path to the YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Minimum confidence for detections")
    parser.add_argument("--width", type=int, default=960,
                        help="Resize frames to this width for speed")
    parser.add_argument("--classes", type=int, nargs="+",
                        default=[0, 1, 2, 3, 5, 7],
                        help="COCO class IDs to keep (default: person, bicycle, "
                             "car, motorcycle, bus, truck)")
    parser.add_argument("--tracker", choices=["histogram", "mobilenet"],
                        default="histogram",
                        help="Appearance feature used by Deep SORT")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO input size (smaller = faster)")
    parser.add_argument("--line-y", type=float, default=0.6,
                        help="Counting line height as a fraction of the frame "
                             "(0.0 = top, 1.0 = bottom)")
    parser.add_argument("--trail", type=int, default=30,
                        help="Trail length in frames (0 = no trails)")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    detector = Detector(args.model, args.conf, args.classes, args.imgsz)
    tracker = Tracker(mode=args.tracker)
    trails = TrailManager(length=max(args.trail, 1))

    frame_idx = 0
    seen_ids = set()
    start = time.time()

    with VideoSource(source, args.width) as video:
        print(f"Source opened: {video.width}x{video.height} @ {video.fps:.1f} FPS")
        counter = LineCounter(video.height * args.line_y)

        writer = None
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, video.fps,
                                     (video.width, video.height))

        while True:
            ok, frame = video.read()
            if not ok:
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            t1 = time.perf_counter()
            tracks = tracker.update(detections, frame)
            t2 = time.perf_counter()

            trails.update(tracks)
            counter.update(tracks)

            if args.trail > 0:
                frame = draw_trails(frame, trails, tracks)
            frame = draw_tracks(frame, tracks)
            frame = draw_counter(frame, counter)

            frame_idx += 1
            seen_ids.update(t["track_id"] for t in tracks)
            avg_fps = frame_idx / (time.time() - start)

            cv2.putText(frame, f"FPS: {avg_fps:.1f}  Tracked: {len(tracks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Detect: {(t1 - t0) * 1000:.0f} ms  "
                        f"Track: {(t2 - t1) * 1000:.0f} ms  "
                        f"[{args.tracker}, imgsz {args.imgsz}]",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if writer:
                writer.write(frame)

            cv2.imshow("Object Detection and Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.release()

    elapsed = time.time() - start
    print("\n--- Summary ---")
    print(f"Frames processed : {frame_idx}")
    print(f"Average FPS      : {frame_idx / elapsed:.2f}")
    print(f"Unique track IDs : {len(seen_ids)}")
    print(f"Crossed down     : {dict(counter.down)}")
    print(f"Crossed up       : {dict(counter.up)}")
    print(f"Settings         : tracker={args.tracker}, imgsz={args.imgsz}, "
          f"width={args.width}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()