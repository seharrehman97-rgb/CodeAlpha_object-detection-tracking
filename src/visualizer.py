import cv2
import numpy as np

PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 215, 0),
    (186, 85, 211), (0, 206, 209), (255, 140, 0), (220, 20, 60),
]
RED = (0, 0, 255)


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


def draw_tracks(frame, tracks, zone=None):
    """Draws box, label and ID. Objects inside the zone turn red with a timer."""
    for t in tracks:
        tid = t["track_id"]
        x1, y1, x2, y2 = map(int, t["bbox"])
        inside = zone is not None and zone.is_inside(tid)
        color = RED if inside else PALETTE[tid % len(PALETTE)]

        label = f'{t["label"]} ID:{tid}'
        if inside:
            secs = int(zone.dwell_seconds(tid))
            label += f" {secs // 60:02d}:{secs % 60:02d}"

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


def draw_zone(frame, zone):
    """Draws the restricted zone as a translucent red area."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone.polygon], RED)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [zone.polygon], True, RED, 2)
    x, y = zone.polygon[0][0]
    cv2.putText(frame, f"RESTRICTED ZONE ({len(zone.inside)} inside)",
                (int(x) + 5, int(y) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    return frame


def draw_hud(frame, avg_fps, n_tracks, detect_ms, track_ms, tag):
    """Draws the live telemetry (FPS and per-stage timing)."""
    cv2.putText(frame, f"FPS: {avg_fps:.1f}  Tracked: {n_tracks}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame,
                f"Detect: {detect_ms:.0f} ms  Track: {track_ms:.0f} ms  [{tag}]",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame