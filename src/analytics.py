from collections import defaultdict, deque


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
                elif prev >= self.line_y > y:
                    self.up[t["label"]] += 1
                    self.counted.add(tid)
            self.last_y[tid] = y