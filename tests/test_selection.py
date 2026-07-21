"""The click-selection mask codec used by the /select route.

Pure (no Redis, no models): a mask must survive PNG encode -> store -> decode as a
(h, w, 1) float32 mask in [0, 1], and a SAM-style selection over part of the frame must
apply the edit only inside it — the property the /select + _run_turn flow relies on.
"""
import numpy as np

from app.cv.engine import ImageEngine, decode_mask, encode_mask


def test_mask_png_roundtrip_shape_and_range():
    mask = np.zeros((12, 20, 1), np.float32)
    mask[:6] = 1.0  # top half selected
    out = decode_mask(encode_mask(mask))

    assert out.shape == (12, 20, 1)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out[0, 0, 0] > 0.9 and out[11, 0, 0] < 0.1  # selection preserved top vs bottom


def test_decoded_mask_applies_edit_only_in_region():
    img = np.full((8, 8, 3), 100, np.uint8)
    mask = np.zeros((8, 8, 1), np.float32)
    mask[:4] = 1.0  # top half
    decoded = decode_mask(encode_mask(mask))

    out = ImageEngine.adjust_exposure(img, decoded, 1.0)  # +1 EV doubles: 100 -> 200
    assert out[0, 0, 0] == 200  # inside selection: edited
    assert out[7, 0, 0] == 100  # outside selection: untouched
