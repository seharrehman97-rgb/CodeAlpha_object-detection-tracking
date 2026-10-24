import cv2


def draw_detections(frame, detections):
    """Draws a box and label for each detection."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["label"]} {det["confidence"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame