# System Design Document: Deterministic AI Photo Retouching Agent (`lumina-agent`)

**Version:** 2.0

**Target Environment:** Railway (CPU-bound, low-latency container)

**Primary Goal:** Low-latency (<600ms), cost-efficient (~$0.10 / 1k edits), deterministic photo retouching via a conversational agent using OpenCV, MediaPipe, and LLM orchestration.

---

## 1. Executive Architecture Overview

`lumina-agent` operates on a **Perceive-Plan-Execute** loop. Instead of streaming heavy raw pixels through an expensive vision LLM or generative diffusion model, the pipeline extracts lightweight mathematical telemetry (histogram data, channel means, regional brightness) and passes text metrics to a fast LLM (e.g., Gemini Flash-Lite or GPT-4o mini).

The LLM outputs structured OpenCV tool execution parameters, which are executed locally on CPU using vectorized NumPy and feather-masked spatial blending.

```
+------------------+      1. Upload / Telemetry      +--------------------+
|  User / Client   | ------------------------------> |  FastAPI Endpoint  |
+------------------+                                 +--------------------+
         ^                                                     |
         |                                           2. Extract Telemetry &
         | 5. Blended Result Output                     Generate Masks
         |                                                     v
+------------------+      4. Apply Edits             +--------------------+
| OpenCV Engine    | <------------------------------ | Pydantic AI Agent  |
| (Masked Blending)|   Tool Calls (e.g., target="sky")| (LLM Planner)      |
+------------------+                                 +--------------------+
         ^                                                     |
         +------------------ Session State -------------------+
                                (Redis)

```

---

## 2. Updated Tech Stack & Hardware Target

* **Framework:** FastAPI (Python 3.11+)
* **Agent Framework:** Pydantic AI (Strict JSON output validation, tool routing)
* **Image Processing Engine:** OpenCV (`opencv-python-headless`) + NumPy
* **Segmentation Engine:** Dual Router — Google MediaPipe (People/Portraits) + OpenCV Spatial Heuristics (Sky/Scenery/Radial)
* **State Management:** Redis (Session history, image state stack, active masks)
* **Observability:** Logfire
* **Target Infrastructure:** Railway Docker Container (512MB RAM budget, 1x Shared vCPU)

---

## 3. Detailed Component Specification

### 3.1 Dual-Engine Unified Mask Router (`app/cv/segmentor.py`)

Executes fast, deterministic spatial segmentation without large GPU model weights, keeping the Docker image tiny (<350MB) and execution fast (<20ms).

```python
from typing import Literal
import cv2
import numpy as np
import mediapipe as mp

TargetType = Literal["global", "subject", "background", "face", "sky", "radial_center"]

class CVSegmentor:
    def __init__(self):
        # MediaPipe Selfie Segmentation (Fast CPU TFLite pipeline)
        self.mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    def get_mask(self, image: np.ndarray, target: TargetType) -> np.ndarray:
        """
        Generates a 3-channel float32 mask (0.0 to 1.0) with Gaussian feathering.
        """
        h, w, _ = image.shape

        if target == "global":
            return np.ones((h, w, 1), dtype=np.float32)

        # 1. MediaPipe Route (Portraits & People)
        if target in ["subject", "background", "face"]:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_selfie.process(rgb)
            mask = results.segmentation_mask  # Float values 0.0 to 1.0

            if target == "background":
                mask = 1.0 - mask

        # 2. Heuristic Route (Sky / Landscapes)
        elif target == "sky":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            saturation = hsv[:, :, 1]
            _, low_sat_mask = cv2.threshold(saturation, 80, 255, cv2.THRESH_BINARY_INV)
            
            combined = cv2.bitwise_and(bright_mask, low_sat_mask)
            y_gradient = np.linspace(1.0, 0.0, h)[:, None]
            mask = (combined.astype(np.float32) / 255.0) * y_gradient

        # 3. Analytical Route (Radial / Center Focus)
        elif target == "radial_center":
            Y, X = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            mask = np.clip(1.0 - (dist / max_dist), 0, 1).astype(np.float32)

        else:
            mask = np.ones((h, w), dtype=np.float32)

        # Soft edge feathering for smooth non-destructive blending
        feathered = cv2.GaussianBlur(mask, (21, 21), 0)
        return np.expand_dims(feathered, axis=-1)

```

---

### 3.2 Non-Destructive Mask Blending Engine (`app/cv/engine.py`)

All localized edits execute via soft-mask blending using the core formula:

$$I_{\text{out}} = I_{\text{orig}} \cdot (1 - M) + I_{\text{edited}} \cdot M$$

```python
import cv2
import numpy as np

class ImageEngine:
    @staticmethod
    def blend(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Applies a full-frame edit onto original using a feathered float32 mask."""
        blended = (original.astype(np.float32) * (1.0 - mask)) + (edited.astype(np.float32) * mask)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @classmethod
    def adjust_exposure(cls, image: np.ndarray, ev: float, mask: np.ndarray) -> np.ndarray:
        """Adjusts exposure on selected region using EV scale factor."""
        factor = 2.0 ** ev
        edited = cv2.multiply(image.astype(np.float32), factor)
        edited = np.clip(edited, 0, 255).astype(np.uint8)
        return cls.blend(image, edited, mask)

    @classmethod
    def adjust_temperature(cls, image: np.ndarray, kelvin_shift: int, mask: np.ndarray) -> np.ndarray:
        """Shifts color temperature (warm/cool) on selected region."""
        edited = image.astype(np.float32)
        if kelvin_shift > 0:  # Warm: boost red, reduce blue
            edited[:, :, 2] += kelvin_shift  # R
            edited[:, :, 0] -= kelvin_shift  # B
        else:  # Cool: boost blue, reduce red
            edited[:, :, 0] += abs(kelvin_shift)  # B
            edited[:, :, 2] -= abs(kelvin_shift)  # R
            
        edited = np.clip(edited, 0, 255).astype(np.uint8)
        return cls.blend(image, edited, mask)

```

---

### 3.3 Telemetry & Data Schemas (`app/schemas.py`)

```python
from typing import Optional, Literal
from pydantic import BaseModel, Field

TargetType = Literal["global", "subject", "background", "sky", "radial_center"]

class RegionalMetrics(BaseModel):
    subject_brightness: Optional[float] = Field(None, description="Average brightness of human subject")
    background_brightness: Optional[float] = Field(None, description="Average brightness of background")
    sky_brightness: Optional[float] = Field(None, description="Average brightness of detected sky region")
    has_human_subject: bool = False

class ImageTelemetry(BaseModel):
    mean_brightness: float = Field(..., description="Global luminance (0-255)")
    contrast_std: float = Field(..., description="Standard deviation of luminance")
    sharpness_laplacian: float = Field(..., description="Variance of Laplacian for sharpness detection")
    is_underexposed: bool
    is_overexposed: bool
    regional: RegionalMetrics

```

---

### 3.4 Agent Tool Registrations (`app/agent/runner.py`)

Pydantic AI agent registering tools that explicitly specify the target area:

```python
from pydantic import Field
from pydantic_ai import Agent
from app.schemas import TargetType

agent = Agent(
    'google-gla:gemini-2.5-flash', # or gpt-4o-mini
    system_prompt="""You are a professional photo retouching agent. 
Analyze telemetry metrics and user requests. 
Issue targeted tool calls to enhance the image deterministically."""
)

@agent.tool_plain
def adjust_exposure(
    ev: float = Field(..., ge=-2.0, le=2.0, description="EV exposure change"),
    target: TargetType = Field("global", description="Target region for edit")
) -> str:
    return f"Executed adjust_exposure: ev={ev}, target={target}"

@agent.tool_plain
def adjust_temperature(
    shift: int = Field(..., ge=-50, le=50, description="Temperature shift value"),
    target: TargetType = Field("global", description="Target region for edit")
) -> str:
    return f"Executed adjust_temperature: shift={shift}, target={target}"

```

---

## 4. End-to-End Execution Flow

| Phase | Component | Action | Time Budget |
| --- | --- | --- | --- |
| **1. Ingestion** | FastAPI | Read frame, convert to BGR NumPy array | ~10 ms |
| **2. Telemetry** | OpenCV + MediaPipe | Compute global stats + regional masks/metrics | ~25 ms |
| **3. Plan** | Pydantic AI / LLM | Evaluate request + metrics $\rightarrow$ Output targeted tool call JSON | ~250–350 ms |
| **4. Execute** | ImageEngine | Generate target mask, execute CV operation, blend layer | ~15 ms |
| **5. Store/Return** | Redis + FastAPI | Store state step, encode output JPEG, return response | ~10 ms |
| **Total** |  |  | **~310–410 ms** |

---

## 5. Railway Deployment Configuration

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

# Install minimal OS dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/
RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false && poetry install --no-root --only main

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

```

### Resource Requirements (Railway)

* **RAM Allocation:** 512 MB (Peak usage stays ~200–250 MB under concurrent loads)
* **vCPU Allocation:** 1 Shared vCPU
* **Network:** Standard HTTP/REST + WebSocket stream support for live preview updates
