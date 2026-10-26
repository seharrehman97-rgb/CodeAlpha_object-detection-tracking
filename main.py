import argparse

import cv2

from src.pipeline import Config, Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Object detection and tracking")
    parser.add_argument("--source", default="0",
                        help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--save", default=None,
                        help="Optional path to save the output video")
    parser.add_argument("--csv", default=None,
                        help="Optional path to save the per-object CSV report")
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
                        help="Counting line height as a fraction of the frame")
    parser.add_argument("--trail", type=int, default=30,
                        help="Trail length in frames (0 = no trails)")
    parser.add_argument("--zone", type=float, nargs=4, default=None,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Restricted zone as fractions of the frame, "
                             "e.g. 0.3 0.55 0.75 1.0")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    cfg = Config(model=args.model, conf=args.conf, width=args.width,
                 classes=args.classes, tracker=args.tracker, imgsz=args.imgsz,
                 line_y=args.line_y, trail=args.trail,
                 zone=tuple(args.zone) if args.zone else None)
    pipeline = Pipeline(cfg)

    writer = None
    frames = pipeline.run(source)
    for frame, info in frames:
        if args.save and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, pipeline.video_fps, (w, h))
        if writer:
            writer.write(frame)

        cv2.imshow("Object Detection and Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frames.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n--- Summary ---")
    for key, value in pipeline.summary().items():
        print(f"{key:22}: {value}")

    if args.csv:
        df = pipeline.log_dataframe()
        df.to_csv(args.csv, index=False)
        print(f"\nSaved {len(df)} rows to {args.csv}")


if __name__ == "__main__":
    main()