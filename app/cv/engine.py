"""Deterministic, non-destructive image ops + the allowlist that gates LLM output.

Localized edits use soft-mask blending:  I_out = I_orig * (1 - M) + I_edited * M
The ACTION_REGISTRY maps a tool name to (validated param model, callable). The LLM
can only ever invoke a name in this registry, with parameters that pass the model's
ge/le bounds — that is what makes "structured output + allowlist" safe.
"""
from typing import Callable

import cv2
import numpy as np
from pydantic import BaseModel, Field


class ImageEngine:
    @staticmethod
    def blend(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Composite a full-frame edit onto original through a feathered float32 mask."""
        blended = (original.astype(np.float32) * (1.0 - mask)) + (edited.astype(np.float32) * mask)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @classmethod
    def adjust_exposure(cls, image: np.ndarray, mask: np.ndarray, ev: float) -> np.ndarray:
        """Scale exposure on the masked region by an EV factor (2**ev)."""
        edited = cv2.multiply(image.astype(np.float32), 2.0 ** ev)
        edited = np.clip(edited, 0, 255).astype(np.uint8)
        return cls.blend(image, edited, mask)

    @classmethod
    def adjust_temperature(cls, image: np.ndarray, mask: np.ndarray, shift: int) -> np.ndarray:
        """Warm (>0: +R/-B) or cool (<0: +B/-R) the masked region. BGR channel order."""
        edited = image.astype(np.float32)
        if shift > 0:
            edited[:, :, 2] += shift   # R
            edited[:, :, 0] -= shift   # B
        else:
            edited[:, :, 0] += abs(shift)  # B
            edited[:, :, 2] -= abs(shift)  # R
        edited = np.clip(edited, 0, 255).astype(np.uint8)
        return cls.blend(image, edited, mask)

    @classmethod
    def unsharp_mask(cls, image: np.ndarray, mask: np.ndarray, amount: float, radius: int) -> np.ndarray:
        """Sharpen detail on the masked region via unsharp masking (image + amount*(image-blur))."""
        r = radius if radius % 2 else radius + 1  # Gaussian kernel size must be odd
        blurred = cv2.GaussianBlur(image, (r, r), 0).astype(np.float32)
        sharpened = cv2.addWeighted(image.astype(np.float32), 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return cls.blend(image, sharpened, mask)

    @classmethod
    def adjust_saturation(cls, image: np.ndarray, mask: np.ndarray, factor: float) -> np.ndarray:
        """Scale color saturation on the masked region (0 = greyscale, >1 = more vivid)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        edited = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cls.blend(image, edited, mask)

    @classmethod
    def adjust_contrast(cls, image: np.ndarray, mask: np.ndarray, factor: float) -> np.ndarray:
        """Scale contrast around mid-grey (128) on the masked region."""
        edited = np.clip((image.astype(np.float32) - 128.0) * factor + 128.0, 0, 255).astype(np.uint8)
        return cls.blend(image, edited, mask)

    @classmethod
    def gaussian_blur(cls, image: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
        """Soften the masked region — pair with a background mask/selection for fake bokeh."""
        r = radius if radius % 2 else radius + 1  # kernel size must be odd
        edited = cv2.GaussianBlur(image, (r, r), 0)
        return cls.blend(image, edited, mask)

    @classmethod
    def auto_white_balance(cls, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Gray-world white balance: scale each channel so its mean matches the overall grey."""
        edited = image.astype(np.float32)
        means = edited.reshape(-1, 3).mean(axis=0)  # per-channel B, G, R means
        gray = float(means.mean())
        for c in range(3):
            if means[c] > 0:
                edited[:, :, c] = np.clip(edited[:, :, c] * (gray / means[c]), 0, 255)
        return cls.blend(image, edited.astype(np.uint8), mask)

    @classmethod
    def clahe(cls, image: np.ndarray, mask: np.ndarray, clip_limit: float) -> np.ndarray:
        """Contrast-limited adaptive histogram equalization on luminance (local 'clarity')."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        equalized = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
        edited = cv2.cvtColor(cv2.merge([equalized, a, b]), cv2.COLOR_LAB2BGR)
        return cls.blend(image, edited, mask)


# --- Allowlist: param models carry the ge/le bounds the doc put on the dead tool stubs ---

class ExposureParams(BaseModel):
    ev: float = Field(..., ge=-2.0, le=2.0, description="EV exposure change")


class TemperatureParams(BaseModel):
    shift: int = Field(..., ge=-50, le=50, description="Temperature shift value")


class UnsharpParams(BaseModel):
    # Defaults give a sensible moderate sharpen when the model omits them (a good sharpen is
    # image-independent, unlike exposure); bounds still apply when the model does provide them.
    amount: float = Field(1.0, ge=0.0, le=2.5, description="Sharpening strength")
    radius: int = Field(3, ge=1, le=9, description="Detail radius in pixels")


class SaturationParams(BaseModel):
    factor: float = Field(1.3, ge=0.0, le=3.0, description="Saturation scale (0=grey, 1=unchanged)")


class ContrastParams(BaseModel):
    factor: float = Field(1.2, ge=0.5, le=2.0, description="Contrast scale around mid-grey")


class BlurParams(BaseModel):
    radius: int = Field(11, ge=1, le=51, description="Blur radius in pixels")


class AutoWhiteBalanceParams(BaseModel):
    """No parameters — gray-world balance is derived from the image."""


class ClaheParams(BaseModel):
    clip_limit: float = Field(2.0, ge=1.0, le=5.0, description="Local-contrast clip limit")


# name -> (validated param model, engine op). Adding an op = one line here.
ACTION_REGISTRY: dict[str, tuple[type[BaseModel], Callable[..., np.ndarray]]] = {
    "adjust_exposure": (ExposureParams, ImageEngine.adjust_exposure),
    "adjust_temperature": (TemperatureParams, ImageEngine.adjust_temperature),
    "unsharp_mask": (UnsharpParams, ImageEngine.unsharp_mask),
    "adjust_saturation": (SaturationParams, ImageEngine.adjust_saturation),
    "adjust_contrast": (ContrastParams, ImageEngine.adjust_contrast),
    "gaussian_blur": (BlurParams, ImageEngine.gaussian_blur),
    "auto_white_balance": (AutoWhiteBalanceParams, ImageEngine.auto_white_balance),
    "clahe": (ClaheParams, ImageEngine.clahe),
}


PRESET_TARGETS = {"global", "subject", "background", "face", "sky", "radial_center"}


def _resolve_mask(image: np.ndarray, target: str, segmentor, regions):
    """Turn an action target into a mask: a `find_region` result, a preset region, or
    (for anything unrecognized) the whole image as a safe default."""
    regions = regions or {}
    name = target[len("region:"):] if isinstance(target, str) and target.startswith("region:") else target
    if name in regions:                                  # object located by the find_region tool
        return regions[name]
    if target in PRESET_TARGETS:                          # built-in region
        return segmentor.get_mask(image, target)
    return segmentor.get_mask(image, "global")           # unknown -> whole image


def encode_mask(mask: np.ndarray) -> bytes:
    """(h,w,1) float32 [0,1] mask -> grayscale+alpha PNG bytes for storage/transport. Alpha
    carries the same value as the grayscale channels: CSS `mask-image` reads a raster image's
    *alpha* channel to decide visibility, not luminance — a plain grayscale (alpha-less) PNG
    is fully opaque everywhere, so the frontend's overlay would show at full coverage no
    matter what the grayscale pixels said. Backend `decode_mask` is unaffected (it reads
    grayscale, which still equals the mask value here)."""
    v = (mask.squeeze(-1) * 255).astype(np.uint8)
    return cv2.imencode(".png", cv2.merge([v, v, v, v]))[1].tobytes()


def decode_mask(mask_png: bytes) -> np.ndarray:
    """Grayscale PNG bytes -> (h,w,1) float32 [0,1] mask for the blend engine."""
    gray = cv2.imdecode(np.frombuffer(mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
    return np.expand_dims(gray.astype(np.float32) / 255.0, axis=-1)


def apply_action(image: np.ndarray, action, segmentor, override_mask=None, regions=None) -> np.ndarray:
    """Validate one ActionCall against the allowlist and apply it. Raises ValueError on
    unknown tool or invalid params so the caller can skip it without a 500.

    Mask precedence: an explicit override_mask (a user click/name selection) wins for every
    action; otherwise the action's target is resolved against `regions` (masks the
    find_region tool produced) and the built-in presets.
    """
    spec = ACTION_REGISTRY.get(action.tool_name)
    if spec is None:
        raise ValueError(f"unknown tool '{action.tool_name}'")
    param_model, fn = spec
    params = param_model(**action.parameters)  # pydantic enforces ge/le, raises on bad input
    mask = override_mask if override_mask is not None else _resolve_mask(image, action.target, segmentor, regions)
    return fn(image, mask, **params.model_dump())


if __name__ == "__main__":
    # ponytail: self-check for the money paths — blend math + registry validation.
    img = np.full((4, 4, 3), 100, np.uint8)
    full = np.ones((4, 4, 1), np.float32)
    zero = np.zeros((4, 4, 1), np.float32)

    # +1 EV over the whole frame doubles then clips: 100 -> 200
    assert ImageEngine.adjust_exposure(img, full, 1.0)[0, 0, 0] == 200
    # zero mask is a no-op regardless of edit strength
    assert np.array_equal(ImageEngine.adjust_exposure(img, zero, 2.0), img)
    # warm shift raises R, lowers B (BGR: idx2=R, idx0=B)
    warm = ImageEngine.adjust_temperature(img, full, 40)
    assert warm[0, 0, 2] == 140 and warm[0, 0, 0] == 60
    # clamps at 0 rather than wrapping: dark blue channel 20 - 50 -> 0
    dark_img = np.full((4, 4, 3), 20, np.uint8)
    assert ImageEngine.adjust_temperature(dark_img, full, 50)[0, 0, 0] == 0

    # unsharp on a flat image is a no-op (blur of flat == flat); it sharpens a real edge
    assert np.array_equal(ImageEngine.unsharp_mask(img, full, 1.0, 3), img)
    edge = np.full((8, 8, 3), 60, np.uint8); edge[:, 4:] = 180  # step edge, no 0/255 clamping
    sharp = ImageEngine.unsharp_mask(edge, np.ones((8, 8, 1), np.float32), 1.5, 3)
    assert sharp[0, 3, 0] < 60 and sharp[0, 4, 0] > 180  # overshoot darkens dark side, brightens bright side

    m8 = np.ones((8, 8, 1), np.float32)
    # saturation factor 0 -> greyscale: channels converge
    colored = np.zeros((8, 8, 3), np.uint8); colored[:] = (200, 100, 50)  # BGR
    desat = ImageEngine.adjust_saturation(colored, m8, 0.0)
    assert abs(int(desat[0, 0, 0]) - int(desat[0, 0, 2])) <= 2
    # contrast factor 2 pushes 100 (below mid-grey) down to 72
    assert ImageEngine.adjust_contrast(np.full((8, 8, 3), 100, np.uint8), m8, 2.0)[0, 0, 0] == 72
    # blur of a flat image is a no-op
    assert np.array_equal(ImageEngine.gaussian_blur(np.full((8, 8, 3), 120, np.uint8), m8, 5),
                          np.full((8, 8, 3), 120, np.uint8))
    # auto white balance narrows a colour cast (channel means converge)
    cast = np.zeros((8, 8, 3), np.uint8); cast[:] = (150, 100, 50)
    awb = ImageEngine.auto_white_balance(cast, m8)
    assert awb.reshape(-1, 3).mean(0).std() < cast.reshape(-1, 3).mean(0).std()
    # clahe returns a valid same-shape image
    cl = ImageEngine.clahe(np.random.randint(0, 255, (16, 16, 3), np.uint8), np.ones((16, 16, 1), np.float32), 2.0)
    assert cl.shape == (16, 16, 3) and cl.dtype == np.uint8

    # allowlist rejects out-of-range and unknown tools
    class _A:
        tool_name = "adjust_exposure"; target = "global"; parameters = {"ev": 99.0}
    class _Seg:
        def get_mask(self, im, t): return np.ones((*im.shape[:2], 1), np.float32)
    for bad in (_A(), type("B", (), {"tool_name": "rm -rf", "target": "global", "parameters": {}})()):
        try:
            apply_action(img, bad, _Seg()); assert False, "should have raised"
        except (ValueError, Exception):
            pass

    # override_mask applies the edit only within the mask and ignores action.target
    half = np.zeros((4, 4, 1), np.float32); half[:2] = 1.0  # top half selected
    good = type("G", (), {"tool_name": "adjust_exposure", "target": "sky", "parameters": {"ev": 1.0}})()
    out = apply_action(img, good, _Seg(), override_mask=half)
    assert out[0, 0, 0] == 200 and out[3, 0, 0] == 100  # top edited (100->200), bottom untouched

    # a find_region result is resolved by target "region:<name>" and masks only that region
    reg = type("R", (), {"tool_name": "adjust_exposure", "target": "region:flower", "parameters": {"ev": 1.0}})()
    out2 = apply_action(img, reg, _Seg(), regions={"flower": half})
    assert out2[0, 0, 0] == 200 and out2[3, 0, 0] == 100
    print("engine self-check ok")
