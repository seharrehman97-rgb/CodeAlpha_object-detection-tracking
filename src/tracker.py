from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    """Wraps Deep SORT: gives each detected object a persistent ID."""

    def __init__(self, max_age=30, n_init=3):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections, frame):
        raw = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            raw.append(([x1, y1, x2 - x1, y2 - y1],
                        det["confidence"], det["label"]))

        tracks = self.tracker.update_tracks(raw, frame=frame)

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