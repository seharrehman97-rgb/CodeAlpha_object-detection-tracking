from ultralytics import YOLO


class Detector:
    """Runs a pre-trained YOLO model and returns clean detection results."""

    def __init__(self, model_path="models/yolo11n.pt", conf=0.4):
        self.model = YOLO(model_path)  # downloads the weights on first run
        self.conf = conf

    def detect(self, frame):
        """Returns a list of dicts: bbox, confidence, class_id, label."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "label": self.model.names[class_id],
            })
        return detections