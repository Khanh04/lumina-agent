import base64
import time
import uuid

import cv2
import logfire
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.agent.runner import RetouchAgentRunner, RetouchDeps
from app.config import settings
from app.cv.analyzer import CVAnalyzer
from app.cv.engine import apply_action
from app.cv.grounding import TextGrounder
from app.cv.sam import MobileSAM
from app.cv.segmentor import CVSegmentor
from app.schemas import RetouchResponse
from app.services.session import RedisSessionManager

app = FastAPI(title="Lumina Agent API", version="2.0")

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


async def _run_turn(session_id: str, prompt: str) -> RetouchResponse:
    start = time.perf_counter()
    current_bytes, messages = await session_mgr.get_current(session_id)
    if not current_bytes:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    img, telemetry = analyzer.analyze(current_bytes)
    deps = RetouchDeps(telemetry=telemetry, image=img, grounder=grounder, sam=sam)
    result = await agent_runner.agent.run(prompt, deps=deps, message_history=messages)
    recipe = result.output

    processed = img.copy()
    skipped: list[str] = []
    for action in recipe.actions:
        try:
            processed = apply_action(processed, action, segmentor, regions=deps.regions)
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


@app.post("/api/v1/sessions/start", response_model=RetouchResponse)
async def start_session(file: UploadFile = File(...), prompt: str = Form(...)):
    session_id = str(uuid.uuid4())
    contents = await file.read()
    await session_mgr.create_session(session_id, contents)
    return await _run_turn(session_id, prompt)


@app.post("/api/v1/sessions/{session_id}/chat", response_model=RetouchResponse)
async def process_chat_message(session_id: str, prompt: str = Form(...)):
    return await _run_turn(session_id, prompt)


@app.post("/api/v1/sessions/{session_id}/undo")
async def undo_last_edit(session_id: str):
    if not await session_mgr.undo(session_id):
        raise HTTPException(status_code=400, detail="Nothing to undo.")
    return {"status": "success", "message": "Reverted last edit."}


# Gradio UI at /ui, sharing the same pipeline singletons as the API above.
import gradio as gr  # noqa: E402

from app.ui import build_ui  # noqa: E402

app = gr.mount_gradio_app(
    app, build_ui(segmentor, analyzer, agent_runner, sam, grounder), path="/ui",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
)
