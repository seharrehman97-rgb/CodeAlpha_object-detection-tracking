import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.analytics import (EventLog, LineCounter, TrailManager, ZoneMonitor,
                           rectangle_polygon)
from src.detector import Detector
from src.tracker import Tracker
from src.video_source import VideoSource
from src.visualizer import (draw_counter, draw_hud, draw_trails, draw_tracks,
                            draw_zone)


@dataclass
class Config:
    """All the settings of one run, in one place."""
    model: str = "models/yolo11n.pt"
    conf: float = 0.4
    width: int = 960
    classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    tracker: str = "histogram"
    imgsz: int = 640
    line_y: float = 0.6
    trail: int = 30
    zone: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2


class Pipeline:
    """Video -> detection -> tracking -> analytics -> annotated frames."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = Detector(cfg.model, cfg.conf, cfg.classes, cfg.imgsz)
        self.frames = 0
        self.elapsed = 0.0
        self.video_fps = 30.0
        self.total_frames = 0
        self.is_live = False
        self.seconds_analyzed = 0.0
        self.seen_ids = set()
        self.counter = None
        self.zone = None
        self.log = None

    def run(self, source, max_seconds=None):
        """Generator: yields (annotated_frame, info) for every frame.

        source: a video file path, or an int for a webcam.
        max_seconds: stop after this many seconds (None = whole video).
        Time is video time for files and the real clock for a webcam.
        """
        cfg = self.cfg
        tracker = Tracker(mode=cfg.tracker)
        trails = TrailManager(length=max(cfg.trail, 1))
        self.is_live = isinstance(source, int)
        self.frames = 0
        self.elapsed = 0.0
        self.seconds_analyzed = 0.0
        self.seen_ids = set()

        with VideoSource(source, cfg.width) as video:
            self.video_fps = video.fps
            self.total_frames = video.frame_count
            self.counter = LineCounter(video.height * cfg.line_y)
            self.zone = None
            if cfg.zone:
                polygon = rectangle_polygon(video.width, video.height, cfg.zone)
                self.zone = ZoneMonitor(polygon, video.fps)
            self.log = EventLog()
            tag = f"{cfg.tracker}, imgsz {cfg.imgsz}"
            start = time.time()

            while True:
                ok, frame = video.read()
                if not ok:
                    break

                now = time.time() - start
                timestamp = now if self.is_live else self.frames / video.fps
                if max_seconds is not None and timestamp >= max_seconds:
                    break

                t0 = time.perf_counter()
                detections = self.detector.detect(frame)
                t1 = time.perf_counter()
                tracks = tracker.update(detections, frame)
                t2 = time.perf_counter()

                trails.update(tracks)
                self.counter.update(tracks)
                if self.zone:
                    if self.is_live:
                        # webcam: use the measured processing rate as fps
                        self.zone.fps = max(self.frames, 1) / max(now, 1e-6)
                    self.zone.update(tracks)
                self.log.update(timestamp, tracks, self.zone)
                self.seconds_analyzed = timestamp

                if self.zone:
                    frame = draw_zone(frame, self.zone)
                if cfg.trail > 0:
                    frame = draw_trails(frame, trails, tracks)
                frame = draw_tracks(frame, tracks, self.zone)
                frame = draw_counter(frame, self.counter)

                self.frames += 1
                self.elapsed = time.time() - start
                self.seen_ids.update(t["track_id"] for t in tracks)
                avg_fps = self.frames / self.elapsed if self.elapsed else 0.0
                frame = draw_hud(frame, avg_fps, len(tracks),
                                 (t1 - t0) * 1000, (t2 - t1) * 1000, tag)

                info = {"frame": self.frames, "avg_fps": avg_fps,
                        "tracked": len(tracks), "timestamp": timestamp}
                yield frame, info

    def summary(self):
        """Final numbers for the run."""
        return {
            "frames": self.frames,
            "seconds_analyzed": round(self.seconds_analyzed, 1),
            "avg_fps": round(self.frames / self.elapsed, 2) if self.elapsed else 0.0,
            "unique_ids": len(self.seen_ids),
            "crossed_down": dict(self.counter.down) if self.counter else {},
            "crossed_up": dict(self.counter.up) if self.counter else {},
            "ids_that_entered_zone": len(self.zone.frames_inside) if self.zone else 0,
        }

    def log_dataframe(self, min_duration=0.0):
        """One row per tracked object, ready to save as CSV."""
        return self.log.to_dataframe(self.zone, self.counter, min_duration)