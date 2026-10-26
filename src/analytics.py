from collections import defaultdict, deque

import cv2
import numpy as np


def format_time(seconds):
    """Turns seconds into mm:ss text, e.g. 65.4 -> '01:05'."""
    seconds = max(int(seconds), 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class TrailManager:
    """Remembers the recent path of each tracked object."""

    def __init__(self, length=30):
        self.trails = defaultdict(lambda: deque(maxlen=length))

    def update(self, tracks):
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            # bottom-center of the box: where the object touches the ground
            self.trails[t["track_id"]].append((int((x1 + x2) / 2), int(y2)))


class LineCounter:
    """Counts objects whose tracks cross a horizontal line (once per ID)."""

    def __init__(self, line_y):
        self.line_y = line_y
        self.last_y = {}
        self.counted = set()
        self.direction = {}  # track_id -> "down" or "up"
        self.down = defaultdict(int)  # class label -> count
        self.up = defaultdict(int)

    def update(self, tracks):
        for t in tracks:
            tid = t["track_id"]
            y = t["bbox"][3]  # bottom edge of the box
            prev = self.last_y.get(tid)
            if prev is not None and tid not in self.counted:
                if prev < self.line_y <= y:
                    self.down[t["label"]] += 1
                    self.counted.add(tid)
                    self.direction[tid] = "down"
                elif prev >= self.line_y > y:
                    self.up[t["label"]] += 1
                    self.counted.add(tid)
                    self.direction[tid] = "up"
            self.last_y[tid] = y


def rectangle_polygon(width, height, fractions):
    """Turns (x1, y1, x2, y2) fractions of the frame into polygon corners."""
    fx1, fy1, fx2, fy2 = fractions
    x1, y1 = int(width * fx1), int(height * fy1)
    x2, y2 = int(width * fx2), int(height * fy2)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


class ZoneMonitor:
    """Tracks which IDs are inside a polygon zone and how long they stay.

    Dwell time = frames inside / fps. For video files fps is the video's
    fps (video time). For a live webcam the pipeline updates fps to the
    measured processing rate, so the time stays close to real time.
    """

    def __init__(self, polygon, fps):
        self.polygon = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        self.fps = fps
        self.frames_inside = defaultdict(int)  # track_id -> frames spent inside
        self.inside = set()  # IDs inside right now

    def update(self, tracks):
        self.inside = set()
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            foot_point = (float((x1 + x2) / 2), float(y2))
            if cv2.pointPolygonTest(self.polygon, foot_point, False) >= 0:
                self.inside.add(t["track_id"])
                self.frames_inside[t["track_id"]] += 1

    def is_inside(self, track_id):
        return track_id in self.inside

    def dwell_seconds(self, track_id):
        return self.frames_inside.get(track_id, 0) / self.fps


COLUMNS = ["track_id", "class", "appeared_at", "left_at", "first_seen_s",
           "last_seen_s", "duration_s", "entered_zone", "zone_dwell_s",
           "line_crossing"]


class EventLog:
    """Collects one summary row per tracked object for the CSV report.

    Timestamps are given in seconds by the pipeline (video time for files,
    real clock for the webcam).
    """

    def __init__(self):
        self.records = {}

    def update(self, timestamp, tracks, zone=None):
        for t in tracks:
            tid = t["track_id"]
            rec = self.records.get(tid)
            if rec is None:
                rec = {"class": t["label"], "first": timestamp,
                       "last": timestamp, "entered_zone": False}
                self.records[tid] = rec
            rec["last"] = timestamp
            if zone is not None and zone.is_inside(tid):
                rec["entered_zone"] = True

    def to_dataframe(self, zone=None, counter=None, min_duration=0.0):
        import pandas as pd

        rows = []
        for tid, rec in self.records.items():
            duration = rec["last"] - rec["first"]
            if duration < min_duration:
                continue  # flicker: seen too briefly to be a real visit
            rows.append({
                "track_id": tid,
                "class": rec["class"],
                "appeared_at": format_time(rec["first"]),
                "left_at": format_time(rec["last"]),
                "first_seen_s": round(rec["first"], 2),
                "last_seen_s": round(rec["last"], 2),
                "duration_s": round(duration, 2),
                "entered_zone": rec["entered_zone"],
                "zone_dwell_s": round(zone.dwell_seconds(tid), 2) if zone else 0.0,
                "line_crossing": counter.direction.get(tid, "none") if counter else "none",
            })
        return pd.DataFrame(rows, columns=COLUMNS)