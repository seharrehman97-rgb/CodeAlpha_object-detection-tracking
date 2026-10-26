from collections import defaultdict, deque

import cv2
import numpy as np


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

    Time is measured in VIDEO time (frames / video fps), not wall-clock
    time, so the result is correct even when processing is slower than
    real time.
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


class EventLog:
    """Collects one summary row per tracked object for the CSV report."""

    def __init__(self, fps):
        self.fps = fps
        self.records = {}

    def update(self, frame_idx, tracks, zone=None):
        for t in tracks:
            tid = t["track_id"]
            rec = self.records.get(tid)
            if rec is None:
                rec = {"class": t["label"], "first": frame_idx,
                       "last": frame_idx, "entered_zone": False}
                self.records[tid] = rec
            rec["last"] = frame_idx
            if zone is not None and zone.is_inside(tid):
                rec["entered_zone"] = True

    def to_dataframe(self, zone=None, counter=None):
        import pandas as pd

        rows = []
        for tid, rec in self.records.items():
            rows.append({
                "track_id": tid,
                "class": rec["class"],
                "first_seen_s": round(rec["first"] / self.fps, 2),
                "last_seen_s": round(rec["last"] / self.fps, 2),
                "duration_s": round((rec["last"] - rec["first"]) / self.fps, 2),
                "entered_zone": rec["entered_zone"],
                "zone_dwell_s": round(zone.dwell_seconds(tid), 2) if zone else 0.0,
                "line_crossing": counter.direction.get(tid, "none") if counter else "none",
            })
        return pd.DataFrame(rows)