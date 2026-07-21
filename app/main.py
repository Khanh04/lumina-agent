import base64
import io
import time
import uuid
from functools import lru_cache

import cv2
import numpy as np
import logfire
import pillow_heif
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

# Let Pillow decode HEIC/AVIF (one call covers both in pillow-heif 1.x) — OpenCV-headless can't.
pillow_heif.register_heif_opener()

from app.agent.runner import RetouchAgentRunner, RetouchDeps
from app.config import settings
from app.cv.analyzer import CVAnalyzer
from app.cv.engine import apply_action, decode_mask, encode_mask
from app.cv.grounding import TextGrounder
from app.cv.sam import MobileSAM
from app.cv.segmentor import CVSegmentor
from app.schemas import RetouchResponse
from app.services.session import RedisSessionManager

# Prefix that tells the agent an explicit region is selected (kept in sync with app/ui.py).
_SEL_NOTE = "[A region is selected; apply edits only within it — the target field is ignored] "

app = FastAPI(title="Lumina Agent API", version="2.0")

# Only needed when the frontend is hosted on a separate origin. Dev uses a Vite proxy and the
# default deploy serves the built UI from FastAPI itself — both same-origin, so leave unset.
if settings.cors_allow_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_fastapi(app)
logfire.instrument_pydantic_ai()

# Singletons — models load once and are shared across the API and the UI.
segmentor = CVSegmentor()          # MediaPipe preset-region masks
analyzer = CVAnalyzer(segmentor)
agent_runner = RetouchAgentRunner()
sam = MobileSAM()                  # MobileSAM (ONNX) — click/box -> mask
grounder = TextGrounder()          # YOLO-World — text -> box (for the find_region tool)
session_mgr = RedisSessionManager(settings.redis_url, settings.session_ttl_seconds)


class SelectRequest(BaseModel):
    x: int  # click point, original-image pixel coordinates
    y: int


class RevertRequest(BaseModel):
    step: int  # target stack index (0 = original image)


@lru_cache(maxsize=8)  # ponytail: process-local LRU; fine single-worker, re-embeds after restart
def _sam_embedding(img_bytes: bytes):
    """SAM image encoding is the heavy step; cache it by image bytes across clicks."""
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return sam.embed(bgr)


def _decode_upload(contents: bytes) -> tuple[np.ndarray, bytes]:
    """Return (BGR image, cv2-decodable bytes). JPEG/PNG/WebP pass through untouched; AVIF/HEIC
    (which OpenCV-headless can't read) are transcoded to PNG so every downstream decode — the
    analyzer, undo/revert — succeeds on the stored bytes. 400 if nothing can read it."""
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img, contents
    try:
        rgb = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image — use JPEG, PNG, WebP, AVIF, or HEIC.")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    return bgr, cv2.imencode(".png", bgr)[1].tobytes()


async def _current_image_b64(session_id: str) -> str:
    """Current image, normalized to JPEG base64 (undo/revert may restore a non-JPEG original)."""
    current, _ = await session_mgr.get_current(session_id)
    if not current:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    img, _ = _decode_upload(current)
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode("utf-8")


async def _run_turn(session_id: str, prompt: str) -> RetouchResponse:
    start = time.perf_counter()
    current_bytes, messages = await session_mgr.get_current(session_id)
    if not current_bytes:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    img, telemetry = analyzer.analyze(current_bytes)
    deps = RetouchDeps(telemetry=telemetry, image=img, grounder=grounder, sam=sam)

    # An active click-selection overrides the agent's target and applies to every action.
    sel_png = await session_mgr.get_selection(session_id)
    override_mask = decode_mask(sel_png) if sel_png else None
    run_prompt = _SEL_NOTE + prompt if override_mask is not None else prompt

    result = await agent_runner.agent.run(run_prompt, deps=deps, message_history=messages)
    recipe = result.output

    processed = img.copy()
    skipped: list[str] = []
    for action in recipe.actions:
        try:
            processed = apply_action(processed, action, segmentor, override_mask=override_mask, regions=deps.regions)
        except Exception as exc:  # unknown tool or out-of-range params -> skip, never 500
            skipped.append(f"{action.tool_name}: {exc}")
            logfire.warn("skipped invalid action", tool=action.tool_name, error=str(exc))

    ok, buffer = cv2.imencode(".jpg", processed)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")
    encoded = buffer.tobytes()
    await session_mgr.push_edit(session_id, encoded, result.all_messages())

    return RetouchResponse(
        session_id=session_id,
        recipe=recipe,
        telemetry=telemetry,
        processed_image_base64=base64.b64encode(encoded).decode("utf-8"),
        execution_time_ms=round((time.perf_counter() - start) * 1000, 2),
        skipped=skipped,
    )


@app.post("/api/v1/sessions/create")
async def create_session(file: UploadFile = File(...)):
    """Create a session from an image without editing it — lets the UI upload, then click-select
    a region, before the first instruction. `chat` handles every edit from turn one."""
    session_id = str(uuid.uuid4())
    img, canonical = _decode_upload(await file.read())  # transcodes AVIF/HEIC; 400 if unreadable
    await session_mgr.create_session(session_id, canonical)
    return {
        "session_id": session_id,
        "image_base64": base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode("utf-8"),
    }


@app.post("/api/v1/sessions/start", response_model=RetouchResponse)
async def start_session(file: UploadFile = File(...), prompt: str = Form(...)):
    session_id = str(uuid.uuid4())
    _, canonical = _decode_upload(await file.read())  # transcodes AVIF/HEIC; 400 if unreadable
    await session_mgr.create_session(session_id, canonical)
    return await _run_turn(session_id, prompt)


@app.post("/api/v1/sessions/{session_id}/chat", response_model=RetouchResponse)
async def process_chat_message(session_id: str, prompt: str = Form(...)):
    return await _run_turn(session_id, prompt)


@app.post("/api/v1/sessions/{session_id}/undo")
async def undo_last_edit(session_id: str):
    if not await session_mgr.undo(session_id):
        raise HTTPException(status_code=400, detail="Nothing to undo.")
    return {"status": "success", "processed_image_base64": await _current_image_b64(session_id)}


@app.post("/api/v1/sessions/{session_id}/revert")
async def revert_to_step(session_id: str, req: RevertRequest):
    if not await session_mgr.revert_to(session_id, req.step):
        raise HTTPException(status_code=400, detail="Step out of range.")
    return {"status": "success", "processed_image_base64": await _current_image_b64(session_id)}


@app.post("/api/v1/sessions/{session_id}/select")
async def select_region(session_id: str, req: SelectRequest):
    current_bytes, _ = await session_mgr.get_current(session_id)
    if not current_bytes:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    mask = sam.mask_at(_sam_embedding(current_bytes), req.x, req.y)  # (h,w,1) float32 [0,1]
    mask_png = encode_mask(mask)
    await session_mgr.set_selection(session_id, mask_png)
    return {"mask_base64": base64.b64encode(mask_png).decode("utf-8")}


@app.post("/api/v1/sessions/{session_id}/select/clear")
async def clear_region_selection(session_id: str):
    await session_mgr.clear_selection(session_id)
    return {"status": "success"}


# Gradio UI at /ui, sharing the same pipeline singletons as the API above.
import gradio as gr  # noqa: E402

from app.ui import build_ui  # noqa: E402

app = gr.mount_gradio_app(
    app, build_ui(segmentor, analyzer, agent_runner, sam, grounder), path="/ui",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
)

# The built React UI at / (produced by `npm run build`; the Docker image bakes it). Mounted
# LAST so /api/v1 and /ui match first. Absent in local dev unless you've run the build — the
# dev flow is `npm run dev` on :5173 proxying to this API.
import os  # noqa: E402

_UI_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_UI_DIST):
    from fastapi.staticfiles import StaticFiles  # noqa: E402

    app.mount("/", StaticFiles(directory=_UI_DIST, html=True), name="ui")
