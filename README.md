# lumina-agent

Deterministic AI photo-retouching service. A fast LLM (Gemini Flash) reads cheap
mathematical telemetry about an image plus your prompt, and emits a structured **recipe**
of targeted edits. A local OpenCV/NumPy engine executes them non-destructively using
feathered region masks (MediaPipe for people, OpenCV heuristics for sky/radial). Session
state and a per-session image undo-stack live in Redis.

No pixels are generated — every edit is a math operation with validated, bounded parameters.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (for local runs) and/or Docker
- A Google API key with Gemini access

Copy the env template and add your key:

```bash
cp .env.example .env
# edit .env -> set GOOGLE_API_KEY=...
```

`.env` (and the downloaded model) are gitignored and never baked into the Docker image.

## Run

### Option A — Docker (app + Redis together)

```bash
docker compose up --build
```

Compose reads `GOOGLE_API_KEY` from `.env`, starts Redis, and downloads the MediaPipe
model into the image. The API is exposed on **http://localhost:8001** (host port per
`docker-compose.yml`).

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
# YOLO-World open-vocab detector (select-by-name):
curl -sSL -o app/cv/models/yolov8s-world.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt

uv run uvicorn app.main:app --reload --port 8000   # .env is loaded automatically
```

Local API: **http://localhost:8000**. Adjust the port below to match how you started it.

## API

Interactive docs at `/docs`. Three endpoints:

| Method | Path | Body | Purpose |
|---|---|---|---|
| POST | `/api/v1/sessions/start` | `file` (image) + `prompt` | Create a session and run the first edit |
| POST | `/api/v1/sessions/{id}/chat` | `prompt` | Continue editing the current image |
| POST | `/api/v1/sessions/{id}/undo` | — | Restore the previous image from the stack |

### Example

```bash
BASE=http://localhost:8000   # or :8001 for Docker

# start a session (returns JSON with session_id + processed_image_base64)
curl -s -F file=@your_photo.jpg -F prompt="brighten the subject and warm it up" \
  $BASE/api/v1/sessions/start | tee /tmp/resp.json >/dev/null

SID=$(python3 -c "import json;print(json.load(open('/tmp/resp.json'))['session_id'])")

# continue the conversation
curl -s -X POST -F prompt="now cool the sky" $BASE/api/v1/sessions/$SID/chat >/dev/null

# undo the last edit
curl -s -X POST $BASE/api/v1/sessions/$SID/undo

# save the processed image from a response
python3 -c "import json,base64;open('out.jpg','wb').write(base64.b64decode(json.load(open('/tmp/resp.json'))['processed_image_base64']))"
```

The response also includes the `recipe` (what the agent planned), the `telemetry` it
reasoned over, `execution_time_ms`, and `skipped` (any actions rejected by the allowlist).

## Tests

```bash
uv run pytest
```

Covers the engine blend math + allowlist, the OpenCV mask routes, and the telemetry
analyzer — none require Redis, MediaPipe weights, or an API key.

## Notes

- **First request is slow**: MediaPipe initializes on startup and the Gemini call adds latency.
- **503 / 429 in a response** is Gemini throttling (overloaded / rate-limited), not a bug —
  retry, or set `MODEL_NAME` in `.env` to another flash model (e.g. `google:gemini-2.0-flash`).
- **Model**: `google:gemini-flash-latest` by default (the doc's `gemini-2.5-flash` is retired
  for new keys). Override with `MODEL_NAME`; switch providers by changing it + the matching key.

## Available edits

The agent may only emit these (bounds enforced before execution):

| Tool | Parameters | Targets |
|---|---|---|
| `adjust_exposure` | `ev` ∈ [-2.0, 2.0] | `global`, `subject`, `background`, `face`, `sky`, `radial_center` |
| `adjust_temperature` | `shift` ∈ [-50, 50] | same |
| `unsharp_mask` | `amount` ∈ [0.0, 2.5], `radius` ∈ [1, 9] | same |
| `adjust_saturation` | `factor` ∈ [0.0, 3.0] | same |
| `adjust_contrast` | `factor` ∈ [0.5, 2.0] | same |
| `gaussian_blur` | `radius` ∈ [1, 51] | same (great with `background` or a click-selection) |
| `auto_white_balance` | — | same |
| `clahe` | `clip_limit` ∈ [1.0, 5.0] | same |

In the `/ui` you can restrict edits to one object two ways: **click it** (MobileSAM, onnxruntime),
or **type its name** in "Select by name" (YOLO-World finds it → MobileSAM masks it — "Grounded-SAM").
While a selection is active every edit applies only within it, overriding the agent's region choice.
Use **Clear selection** to deselect, or upload a new image to reset.
