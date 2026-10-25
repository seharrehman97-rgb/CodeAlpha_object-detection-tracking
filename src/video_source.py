import cv2


class VideoSource:
    """Wraps a webcam or video file and optionally resizes frames."""

    def __init__(self, source=0, target_width=None):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if target_width and orig_w > target_width:
            self.width = target_width
            self.height = int(orig_h * target_width / orig_w)
        else:
            self.width, self.height = orig_w, orig_h
        self._resize = (self.width, self.height) != (orig_w, orig_h)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0

    def read(self):
        """Returns (success, frame). success is False at the end of a video."""
        ok, frame = self.cap.read()
        if ok and self._resize:
            frame = cv2.resize(frame, (self.width, self.height))
        return ok, frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()