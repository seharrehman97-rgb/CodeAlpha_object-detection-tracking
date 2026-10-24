import argparse
import time

import cv2

from src.detector import Detector
from src.tracker import Tracker
from src.video_source import VideoSource
from src.visualizer import draw_tracks


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
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    detector = Detector(args.model, args.conf)
    tracker = Tracker()

    with VideoSource(source) as video:
        print(f"Source opened: {video.width}x{video.height} @ {video.fps:.1f} FPS")

        writer = None
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, video.fps,
                                     (video.width, video.height))

        prev_time = time.time()
        while True:
            ok, frame = video.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            frame = draw_tracks(frame, tracks)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            cv2.putText(frame, f"FPS: {fps:.1f}  Tracked: {len(tracks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if writer:
                writer.write(frame)

            cv2.imshow("Object Detection and Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()