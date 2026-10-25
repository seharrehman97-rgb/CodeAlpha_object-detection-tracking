import cv2
import numpy as np

PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 215, 0),
    (186, 85, 211), (0, 206, 209), (255, 140, 0), (220, 20, 60),
]


def draw_detections(frame, detections):
    """Draws a plain box and label for each raw detection."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["label"]} {det["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def draw_tracks(frame, tracks):
    """Draws a box, label, and tracking ID, with one color per ID."""
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        color = PALETTE[t["track_id"] % len(PALETTE)]
        label = f'{t["label"]} ID:{t["track_id"]}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def draw_trails(frame, trail_manager, tracks):
    """Draws the recent path of every currently tracked object."""
    for t in tracks:
        points = trail_manager.trails.get(t["track_id"])
        if points and len(points) > 1:
            color = PALETTE[t["track_id"] % len(PALETTE)]
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, color, 2)
    return frame


def draw_counter(frame, counter):
    """Draws the counting line and the running totals."""
    h, w = frame.shape[:2]
    y = int(counter.line_y)
    cv2.line(frame, (0, y), (w, y), (0, 255, 255), 2)
    down = sum(counter.down.values())
    up = sum(counter.up.values())
    cv2.putText(frame, f"Crossed down: {down}   up: {up}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame