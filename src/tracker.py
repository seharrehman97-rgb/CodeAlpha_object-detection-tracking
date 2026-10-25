import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


def _histogram_embedding(frame, bbox):
    """Cheap appearance feature: HSV color histogram of the object's crop."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.ones(128, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist


class Tracker:
    """Wraps Deep SORT: gives each detected object a persistent ID.

    mode="mobilenet": CNN appearance embedder (accurate, slow on CPU)
    mode="histogram": color-histogram appearance feature (fast)
    """

    def __init__(self, mode="histogram", max_age=30, n_init=3):
        self.mode = mode
        if mode == "mobilenet":
            self.tracker = DeepSort(max_age=max_age, n_init=n_init)
        elif mode == "histogram":
            self.tracker = DeepSort(max_age=max_age, n_init=n_init,
                                    embedder=None, max_cosine_distance=0.3)
        else:
            raise ValueError(f"Unknown tracker mode: {mode}")

    @staticmethod
    def _collect(tracks):
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            left, top, right, bottom = track.to_ltrb()
            results.append({
                "track_id": int(track.track_id),
                "bbox": (left, top, right, bottom),
                "label": track.get_det_class() or "object",
            })
        return results

    def update(self, detections, frame):
        """Full update: match new detections to existing tracks."""
        raw = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            raw.append(([x1, y1, x2 - x1, y2 - y1],
                        det["confidence"], det["label"]))

        if self.mode == "histogram":
            embeds = [_histogram_embedding(frame, d["bbox"]) for d in detections]
            tracks = self.tracker.update_tracks(raw, embeds=embeds, frame=frame)
        else:
            tracks = self.tracker.update_tracks(raw, frame=frame)
        return self._collect(tracks)

    def predict(self):
        """Used on frames without detection: move tracks using motion only."""
        inner = self.tracker.tracker
        inner.predict()
        return self._collect(inner.tracks)