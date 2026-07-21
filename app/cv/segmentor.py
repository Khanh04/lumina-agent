"""Dual-engine mask router: MediaPipe selfie-seg for people, OpenCV heuristics for
scenery. Returns Gaussian-feathered (h, w, 1) float32 masks in [0, 1] for soft blending.
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from app.config import settings
from app.schemas import TargetType

_MP_TARGETS = {"subject", "background", "face"}


def _feather(mask: np.ndarray) -> np.ndarray:
    """Gaussian-feather a (h, w) float mask and return it as (h, w, 1) float32."""
    feathered = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    return np.expand_dims(feathered, axis=-1)


class CVSegmentor:
    def __init__(self, model_path: str | None = None):
        # MediaPipe Tasks selfie segmenter (current API). Fast CPU TFLite pipeline,
        # instantiated once and reused. Outputs one foreground-confidence mask in [0, 1].
        options = vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path or settings.segmenter_model_path),
            output_confidence_masks=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def get_mask(self, image: np.ndarray, target: TargetType) -> np.ndarray:
        h, w, _ = image.shape

        if target == "global":
            return np.ones((h, w, 1), dtype=np.float32)

        if target in _MP_TARGETS:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            mask = self.segmenter.segment(mp_image).confidence_masks[0].numpy_view()  # (h, w, 1) foreground prob
            mask = np.asarray(mask, dtype=np.float32)[:, :, 0]
            if target == "background":
                mask = 1.0 - mask
            # ponytail: "face" reuses the whole-person mask; upgrade to a FaceLandmarker/face
            # detector bbox if true face-only targeting is needed.

        elif target == "sky":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            _, low_sat = cv2.threshold(hsv[:, :, 1], 80, 255, cv2.THRESH_BINARY_INV)
            combined = cv2.bitwise_and(bright, low_sat)
            y_gradient = np.linspace(1.0, 0.0, h)[:, None]  # sky is up top
            mask = (combined.astype(np.float32) / 255.0) * y_gradient

        elif target == "radial_center":
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - w / 2) ** 2 + (Y - h / 2) ** 2)
            max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
            mask = np.clip(1.0 - dist / max_dist, 0, 1).astype(np.float32)

        else:  # unreachable given TargetType, but keep the engine safe
            mask = np.ones((h, w), dtype=np.float32)

        return _feather(mask)


if __name__ == "__main__":
    # ponytail: self-check the OpenCV routes (no MediaPipe needed) — shape, dtype, range.
    seg = object.__new__(CVSegmentor)  # skip __init__ so we don't load the TFLite model
    img = np.random.randint(0, 255, (30, 40, 3), np.uint8)
    for t in ("global", "sky", "radial_center"):
        m = CVSegmentor.get_mask(seg, img, t)
        assert m.shape == (30, 40, 1), (t, m.shape)
        assert m.dtype == np.float32 and 0.0 <= m.min() and m.max() <= 1.0, t
    # radial mask is brightest at the center, dark at a corner
    r = CVSegmentor.get_mask(seg, img, "radial_center")
    assert r[15, 20, 0] > r[0, 0, 0]
    print("segmentor self-check ok")
