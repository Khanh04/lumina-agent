"""Lightweight mathematical telemetry: global luminance stats + regional brightness
under segmentor masks. This is the cheap text the LLM planner reasons over instead of
raw pixels.
"""
import cv2
import numpy as np

from app.schemas import ImageTelemetry, RegionalMetrics

# Global exposure thresholds (carried from the v1 design doc).
UNDEREXPOSED_BELOW = 85.0
OVEREXPOSED_ABOVE = 190.0
# Fraction of the frame the subject mask must cover to count as "a person is present".
SUBJECT_PRESENCE_MIN = 0.03


def _masked_mean(gray: np.ndarray, mask: np.ndarray) -> float | None:
    """Feather-weighted mean luminance under a (h, w, 1) mask; None if the region is empty."""
    m = mask[..., 0]
    total = float(m.sum())
    if total < 1.0:
        return None
    return round(float((gray.astype(np.float32) * m).sum() / total), 2)


class CVAnalyzer:
    def __init__(self, segmentor):
        self.segmentor = segmentor

    def analyze(self, image_bytes: bytes) -> tuple[np.ndarray, ImageTelemetry]:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("could not decode image bytes")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        subject_mask = self.segmentor.get_mask(img, "subject")
        coverage = float((subject_mask[..., 0] > 0.5).mean())

        regional = RegionalMetrics(
            subject_brightness=_masked_mean(gray, subject_mask),
            background_brightness=_masked_mean(gray, self.segmentor.get_mask(img, "background")),
            sky_brightness=_masked_mean(gray, self.segmentor.get_mask(img, "sky")),
            has_human_subject=coverage >= SUBJECT_PRESENCE_MIN,
        )

        telemetry = ImageTelemetry(
            mean_brightness=round(brightness, 2),
            contrast_std=round(float(np.std(gray)), 2),
            sharpness_laplacian=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
            is_underexposed=brightness < UNDEREXPOSED_BELOW,
            is_overexposed=brightness > OVEREXPOSED_ABOVE,
            regional=regional,
        )
        return img, telemetry


if __name__ == "__main__":
    # ponytail: self-check global stats + masked-mean with a stub segmentor (no MediaPipe).
    class _Seg:
        def get_mask(self, im, t):
            h, w, _ = im.shape
            return np.ones((h, w, 1), np.float32) if t != "background" else np.zeros((h, w, 1), np.float32)

    dark = cv2.imencode(".png", np.full((20, 20, 3), 30, np.uint8))[1].tobytes()
    _, tel = CVAnalyzer(_Seg()).analyze(dark)
    assert tel.is_underexposed and not tel.is_overexposed
    assert tel.regional.subject_brightness == 30.0        # full mask -> mean of 30
    assert tel.regional.background_brightness is None       # empty mask -> None
    assert _masked_mean(np.full((5, 5), 100, np.uint8), np.zeros((5, 5, 1), np.float32)) is None
    print("analyzer self-check ok")
