"""MobileSAM click-to-select, running under onnxruntime (no torch).

The encoder/decoder pre/post-processing (684x1024 canvas, transform matrix, padding point,
mask warp-back) is vendored from samexporter's onnxruntime reference so it matches these
exact exported models. `MobileSAM` adds an app-facing wrapper: encode an image once, then
turn a click point into a feathered (h, w, 1) mask that drops into the blend engine.

Usage: encode the image on upload (cache the embedding in session state), then call
`mask_at(embedding, x_px, y_px)` per click — the heavy encode runs once, decode is cheap.
"""
from copy import deepcopy

import cv2
import numpy as np
import onnxruntime

from app.config import settings


class _SegmentAnythingONNX:
    """Vendored from vietanhdev/samexporter (onnxruntime-only inference path)."""

    def __init__(self, encoder_model_path: str, decoder_model_path: str):
        self.target_size = 1024
        self.input_size = (684, 1024)
        providers = [p for p in onnxruntime.get_available_providers() if p != "TensorrtExecutionProvider"]
        self.encoder_session = onnxruntime.InferenceSession(encoder_model_path, providers=providers)
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_session = onnxruntime.InferenceSession(decoder_model_path, providers=providers)

    @staticmethod
    def get_preprocess_shape(oldh, oldw, long_side_length):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        return (int(oldh * scale + 0.5), int(oldw * scale + 0.5))

    def apply_coords(self, coords, original_size, target_length):
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def encode(self, cv_image):
        original_size = cv_image.shape[:2]
        scale = min(self.input_size[1] / cv_image.shape[1], self.input_size[0] / cv_image.shape[0])
        transform_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        canvas = cv2.warpAffine(cv_image, transform_matrix[:2], (self.input_size[1], self.input_size[0]),
                                flags=cv2.INTER_LINEAR)
        embedding = self.encoder_session.run(None, {self.encoder_input_name: canvas.astype(np.float32)})[0]
        return {"image_embedding": embedding, "original_size": original_size, "transform_matrix": transform_matrix}

    def predict_masks(self, embedding, points, labels):
        onnx_coord = np.concatenate([points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
        onnx_coord = self.apply_coords(onnx_coord, self.input_size, self.target_size).astype(np.float32)
        # map click coords (original pixels) onto the 684x1024 encoder canvas
        onnx_coord = np.concatenate([onnx_coord, np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32)], axis=2)
        onnx_coord = np.matmul(onnx_coord, embedding["transform_matrix"].T)[:, :, :2].astype(np.float32)

        masks, _, _ = self.decoder_session.run(None, {
            "image_embeddings": embedding["image_embedding"],
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array(self.input_size, dtype=np.float32),
        })
        # warp the canvas-space mask back to the original image size
        inv = np.linalg.inv(embedding["transform_matrix"])
        h, w = embedding["original_size"]
        return cv2.warpAffine(masks[0, 0], inv[:2], (w, h), flags=cv2.INTER_LINEAR)


class MobileSAM:
    def __init__(self, encoder_path: str | None = None, decoder_path: str | None = None):
        self.model = _SegmentAnythingONNX(
            encoder_path or settings.sam_encoder_path,
            decoder_path or settings.sam_decoder_path,
        )

    def embed(self, image_bgr: np.ndarray):
        """Run the (heavy) encoder once; returns an embedding to reuse across clicks."""
        return self.model.encode(image_bgr)

    def mask_at(self, embedding, x_px: float, y_px: float) -> np.ndarray:
        """A click point (original-image pixels) -> feathered (h, w, 1) float32 mask in [0, 1]."""
        return self._to_mask(self.model.predict_masks(embedding, np.array([[x_px, y_px]]), np.array([1])))

    def mask_from_box(self, embedding, box) -> np.ndarray:
        """A box (x1, y1, x2, y2 original pixels) -> feathered mask. SAM box prompt = the two
        corner points with labels 2 (top-left) and 3 (bottom-right)."""
        x1, y1, x2, y2 = box
        pts = np.array([[x1, y1], [x2, y2]], dtype=float)
        return self._to_mask(self.model.predict_masks(embedding, pts, np.array([2, 3])))

    @staticmethod
    def _to_mask(mask_logits: np.ndarray) -> np.ndarray:
        binary = (mask_logits > 0).astype(np.float32)  # decoder returns logits; 0 is the SAM threshold
        feathered = cv2.GaussianBlur(binary, (21, 21), 0)
        return np.expand_dims(feathered, axis=-1)
