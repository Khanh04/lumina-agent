FROM python:3.11-slim

# Minimal OS deps for OpenCV + MediaPipe. The MediaPipe Tasks C bindings need EGL/GLES
# (libGLESv2.so.2), which aren't in python:3.11-slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# uv for dependency management (replaces the doc's poetry).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . /app

# Fetch segmentation models (kept out of git): MediaPipe selfie-seg (preset regions) +
# MobileSAM ONNX encoder/decoder (click-to-select), unzipped into the models dir.
ADD https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite \
    /app/app/cv/models/selfie_segmenter.tflite
ADD https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam_20230629.zip \
    /tmp/mobile_sam.zip
RUN python -c "import zipfile; zipfile.ZipFile('/tmp/mobile_sam.zip').extractall('/app/app/cv/models')" \
    && rm /tmp/mobile_sam.zip
# YOLO-World open-vocab detector (text -> region).
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt \
    /app/app/cv/models/yolov8s-world.pt

RUN uv sync --frozen --no-dev

# Bake YOLO-World's text encoder into the image: ultralytics installs its `clip` fork (via git)
# and downloads the CLIP weights on first set_classes(). Doing it here means no runtime git/network.
RUN uv run python -c "from ultralytics import YOLOWorld; YOLOWorld('/app/app/cv/models/yolov8s-world.pt').set_classes(['object'])"

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
