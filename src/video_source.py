import cv2


class VideoSource:
    """Wraps a webcam or video file behind one simple interface."""

    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0

    def read(self):
        """Returns (success, frame). success is False at the end of a video."""
        return self.cap.read()

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()