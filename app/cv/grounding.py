"""Text -> region grounding via YOLO-World (open-vocabulary detection). Turns a noun
phrase like "the flower" into a bounding box; MobileSAM then turns that box into a clean
mask (Grounded-SAM). Runs on CPU (the deploy target); the box is handed to SAM.

ponytail: torch was resolved as a CUDA build, so we force CPU (`to("cpu")` + device="cpu")
— matches the no-GPU deploy target and avoids YOLO-World's set_classes device-mismatch bug.
A CPU-only torch wheel would shrink the image a lot; pin it if image size matters.
"""
import numpy as np

from app.config import settings

# Detections below this confidence are treated as "not found" (rejects absent objects,
# which YOLO-World otherwise reports as a low-confidence whole-image box).
_MIN_CONF = 0.10


class TextGrounder:
    def __init__(self, model_path: str | None = None):
        from ultralytics import YOLOWorld  # imported lazily; pulls torch

        self.model = YOLOWorld(model_path or settings.grounder_model_path)
        self.model.to("cpu")
        # Warm the CLIP text encoder at startup (it lazy-downloads on first set_classes), so the
        # first user request isn't slow. Fail-soft: no internet at boot shouldn't break the app.
        try:
            self.model.set_classes(["object"])
        except Exception:
            pass

    def detect(self, image_bgr: np.ndarray, phrase: str, conf: float = _MIN_CONF):
        """Return the highest-confidence box (x1, y1, x2, y2) as ints for `phrase`, or None."""
        phrase = (phrase or "").strip()
        if not phrase:
            return None
        self.model.set_classes([phrase])
        boxes = self.model.predict(image_bgr, verbose=False, conf=conf, device="cpu")[0].boxes
        if len(boxes) == 0:
            return None
        best = int(boxes.conf.argmax())
        return tuple(round(v) for v in boxes.xyxy[best].tolist())
