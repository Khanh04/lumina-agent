# lumina-agent

Deterministic AI photo-retouching service. A fast LLM (Gemini Flash) reads cheap
mathematical telemetry about an image plus your prompt, and emits a structured **recipe**
of targeted edits. A local OpenCV/NumPy engine executes them non-destructively using
feathered region masks. Session state and a per-session image undo-stack live in Redis.

No pixels are generated — every edit is a math operation with validated, bounded parameters.

**Region targeting**, in order of specificity:
- **Presets** — `global`, `subject`, `background`, `face` (MediaPipe), `sky`, `radial_center` (OpenCV heuristics).
- **Named objects** — the agent has a `find_region` tool: when you mention a specific object
  ("sharpen the flower", "warm the car"), it locates it with **YOLO-World** and masks it with
  **MobileSAM** (Grounded-SAM), then targets the edit there.
- **Manual selection** — in the `/ui`, click an object to select it (MobileSAM); an active
  selection overrides everything for that turn.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (for local runs) and/or Docker
- A Google API key with Gemini access
- `git` on the machine (YOLO-World's CLIP text encoder is fetched via git; the Docker build
  bakes it — see [Deployment](#deployment--footprint))

Copy the env template and add your key:

```bash
cp .env.example .env
# edit .env -> set GOOGLE_API_KEY=...
```

`.env` and the model weights are gitignored; `.env` is never baked into the Docker image.

## Run

### Option A — Docker (app + Redis together)

```bash
docker compose up --build
```

Compose reads `GOOGLE_API_KEY` from `.env`, starts Redis, and downloads all models into the
image. The API is exposed on **http://localhost:8001** (host port per `docker-compose.yml`);
the UI is at **http://localhost:8001/ui**.

### Option B — Local (you supply Redis)

```bash
uv sync                 # install deps (first time only)

# Redis — pick one:
docker run -d -p 6379:6379 redis:7-alpine
# or:  sudo service redis-server start

# fetch the segmentation models once (Docker does this automatically):
mkdir -p app/cv/models
curl -sSL -o app/cv/models/selfie_segmenter.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
# MobileSAM ONNX (click-to-select) — a zip containing the encoder + decoder:
curl -sSL -o /tmp/mobile_sam.zip \
  https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam_20230629.zip
python -c "import zipfile; zipfile.ZipFile('/tmp/mobile_sam.zip').extractall('app/cv/models')"
# YOLO-World open-vocab detector (named-object targeting):
curl -sSL -o app/cv/models/yolov8s-world.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt

uv run uvicorn app.main:app --reload --port 8000   # .env is loaded automatically
```

Local API: **http://localhost:8000**, UI: **http://localhost:8000/ui**. On first startup
ultralytics installs its CLIP fork via git (needs network once). Adjust the port in the
examples below to match how you started it.

## UI

Two front-ends share the same pipeline:

- **`/`** — the dedicated **React** app (`frontend/`), a cinematic full-bleed editor with a
  before/after seam, click-to-select, a version timeline, and a live "recipe" HUD. The Docker
  image bakes the built assets and FastAPI serves them at `/`. For local dev, run it hot:
  ```bash
  cd frontend && npm install && npm run dev    # http://localhost:5173, proxies /api to :8000
  ```
- **`/ui`** — the original Gradio app (kept during the transition), described below.

### Gradio (`/ui`)

A Gradio app for interactive editing:
- **Upload** a photo (left "Original" pane).
- **Click an object** to select it — the green highlight shows the mask; edits then apply
  only there. Or just name the object in your instruction and let the agent find it.
- **Instruction** box + **Apply** → the result appears on the right; **Original** and
  **Result** sit side-by-side as a before/after.
- **History strip** — every step is a thumbnail; click one to revert to it.
- **Undo** (one step) and **Clear selection**.

## API

Interactive docs at `/docs`. Three endpoints:

| Method | Path | Body | Purpose |
|---|---|---|---|
| POST | `/api/v1/sessions/create` | `file` (image) | Create a session without editing (returns `session_id` + image) |
| POST | `/api/v1/sessions/start` | `file` (image) + `prompt` | Create a session and run the first edit |
| POST | `/api/v1/sessions/{id}/chat` | `prompt` | Continue editing the current image |
| POST | `/api/v1/sessions/{id}/undo` | — | Restore the previous image (returns it) |
| POST | `/api/v1/sessions/{id}/revert` | `{step}` (JSON) | Roll the stack back to a version (returns that image) |
| POST | `/api/v1/sessions/{id}/select` | `{x, y}` (JSON) | Click-select a region (MobileSAM); returns a mask, applied to the next edit |
| POST | `/api/v1/sessions/{id}/select/clear` | — | Drop the active click-selection |

### Example

```bash
BASE=http://localhost:8000   # or :8001 for Docker

# start a session (returns JSON with session_id + processed_image_base64)
curl -s -F file=@your_photo.jpg -F prompt="brighten the subject and warm it up" \
  $BASE/api/v1/sessions/start | tee /tmp/resp.json >/dev/null

SID=$(python3 -c "import json;print(json.load(open('/tmp/resp.json'))['session_id'])")

# continue the conversation (the agent can target named objects, e.g. "sharpen the flower")
curl -s -X POST -F prompt="now cool the sky" $BASE/api/v1/sessions/$SID/chat >/dev/null

# undo the last edit
curl -s -X POST $BASE/api/v1/sessions/$SID/undo

# save the processed image from a response
python3 -c "import json,base64;open('out.jpg','wb').write(base64.b64decode(json.load(open('/tmp/resp.json'))['processed_image_base64']))"
```

The response also includes the `recipe` (what the agent planned), the `telemetry` it
reasoned over, `execution_time_ms`, and `skipped` (any actions rejected by the allowlist).

## Available edits

The agent may only emit these (bounds enforced before execution). Each `target` is a preset
region or a `region:<name>` produced by the `find_region` tool:

| Tool | Parameters |
|---|---|
| `adjust_exposure` | `ev` ∈ [-2.0, 2.0] |
| `adjust_temperature` | `shift` ∈ [-50, 50] |
| `unsharp_mask` | `amount` ∈ [0.0, 2.5], `radius` ∈ [1, 9] |
| `adjust_saturation` | `factor` ∈ [0.0, 3.0] |
| `adjust_contrast` | `factor` ∈ [0.5, 2.0] |
| `gaussian_blur` | `radius` ∈ [1, 51] (blur a `background`/named region for fake bokeh) |
| `auto_white_balance` | — |
| `clahe` | `clip_limit` ∈ [1.0, 5.0] |

## Deployment & footprint

CPU-only — no GPU anywhere (SAM runs on onnxruntime, YOLO-World is forced to CPU,
MediaPipe/OpenCV are CPU). Provision accordingly:

- **RAM**: ~1.5 GB resident with all models loaded (torch + CLIP + MobileSAM + MediaPipe +
  Gradio). Use a plan with **≥2 GB** (the original 512 MB target predates SAM/YOLO-World).
- **Image**: ~2 GB. torch is pinned to the **CPU-only** wheel (see `[tool.uv.sources]` in
  `pyproject.toml`) — without that it pulls ~4.5 GB of unused CUDA.
- **Build**: needs network + `git`. The Dockerfile downloads the model weights and runs a
  warm-up that bakes YOLO-World's `clip` module and CLIP weights into the image, so runtime
  needs no git/network for grounding.

## Tests

```bash
uv run pytest
```

Covers the engine blend math + allowlist (incl. region/override masks), the OpenCV mask
routes, and the telemetry analyzer — none require Redis, model weights, or an API key.

## Notes

- **First request is slow**: models initialize on startup and the Gemini call adds latency.
- **503 / 429 in a response** is Gemini throttling (overloaded / rate-limited), not a bug —
  retry, or set `MODEL_NAME` in `.env` to another flash model.
- **Model**: `google:gemini-flash-lite-latest` by default (`gemini-2.5-flash` is retired for
  new keys, `flash-latest` is often throttled). Override with `MODEL_NAME`; switch providers
  by changing it + the matching key.
