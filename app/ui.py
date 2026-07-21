"""Gradio UI mounted on the FastAPI app at /ui. Reuses the same pipeline singletons as
the REST API; per-browser session state (image stack + message history + active click
selection + cached SAM embedding) lives in gr.State.

Layout: the left "Original" pane is the fixed reference you upload to and click on
(selection highlighted in place); the right "Result" pane shows the evolving edit — a
clear side-by-side before/after. A clickable History strip reverts to any step. New-image
detection uses `.upload` (only fires on real uploads) so redrawing the selection highlight
on the Original never triggers a session reset.
"""
import cv2
import numpy as np
import gradio as gr

from app.agent.runner import RetouchDeps
from app.cv.engine import apply_action

_SEL_NOTE = "[A region is selected; apply edits only within it — the target field is ignored] "


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Highlight the selected region (green, 50%) on the image; return RGB for display."""
    highlight = np.zeros_like(bgr, dtype=np.float32)
    highlight[:] = (0, 255, 0)  # BGR green
    a = 0.5 * mask
    return _rgb((bgr.astype(np.float32) * (1 - a) + highlight * a).clip(0, 255).astype(np.uint8))


def _gallery(stack):
    """History thumbnails, captioned by step (0 = original)."""
    return [(_rgb(s), "original" if i == 0 else f"step {i}") for i, s in enumerate(stack)]


def _format(recipe, telemetry, skipped) -> str:
    lines = [f"**{recipe.summary}**", "", f"_telemetry: brightness {telemetry.mean_brightness}, "
             f"contrast {telemetry.contrast_std}, sharpness {telemetry.sharpness_laplacian}_", ""]
    for a in recipe.actions:
        lines.append(f"- `{a.tool_name}` → **{a.target}** {a.parameters} — {a.rationale}")
    if not recipe.actions:
        lines.append("- _(no edits proposed)_")
    if skipped:
        lines += [""] + [f"- ⚠️ skipped: {s}" for s in skipped]
    return "\n".join(lines)


def build_ui(segmentor, analyzer, agent_runner, sam, grounder) -> gr.Blocks:
    def on_upload(upload_rgb):
        # Only fires on a real user upload: start a fresh session and encode once for SAM.
        if upload_rgb is None:
            return None, [], "", [], [], None, None
        bgr = cv2.cvtColor(upload_rgb, cv2.COLOR_RGB2BGR)
        embedding = sam.embed(bgr)
        stack = [bgr]
        note = "Ready. **Click an object** to retouch just that region, or describe an edit for the whole image."
        return _rgb(bgr), _gallery(stack), note, stack, [], None, embedding

    def on_select(stack, embedding, evt: gr.SelectData):
        stack = list(stack or [])
        if not stack or embedding is None:
            return gr.update(), "Upload an image first.", None
        x, y = evt.index  # original-image pixel coords
        mask = sam.mask_at(embedding, x, y)
        note = "Region selected — describe an edit and it applies here. **Clear selection** to deselect."
        return _overlay(stack[0], mask), note, mask  # highlight on the Original pane itself

    def clear_sel(stack):
        stack = list(stack or [])
        original = _rgb(stack[0]) if stack else None
        return original, "Selection cleared — edits apply to the region the agent chooses.", None

    async def run_turn(prompt, stack, messages, selection):
        stack = list(stack or [])
        if not stack:
            return gr.update(), gr.update(), "Upload an image first.", stack, messages
        current = stack[-1]
        png = cv2.imencode(".png", current)[1].tobytes()
        img, telemetry = analyzer.analyze(png)

        run_prompt = prompt or "Auto-enhance this photo."
        if selection is not None:
            run_prompt = _SEL_NOTE + (prompt or "Enhance the selected region.")
        deps = RetouchDeps(telemetry=telemetry, image=current, grounder=grounder, sam=sam)
        result = await agent_runner.agent.run(run_prompt, deps=deps, message_history=messages or [])
        recipe = result.output

        processed, skipped = img.copy(), []
        for a in recipe.actions:
            try:
                processed = apply_action(processed, a, segmentor, override_mask=selection, regions=deps.regions)
            except Exception as exc:  # allowlist rejection -> skip, never crash the UI
                skipped.append(f"{a.tool_name}: {exc}")

        stack.append(processed)
        return _rgb(processed), _gallery(stack), _format(recipe, telemetry, skipped), stack, result.all_messages()

    def undo(stack, messages):
        stack = list(stack or [])
        if len(stack) < 2:
            return gr.update(), gr.update(), "Nothing to undo.", stack, messages
        stack.pop()
        messages = (messages or [])[:-2]
        return _rgb(stack[-1]), _gallery(stack), "Reverted last edit.", stack, messages

    def on_history_select(stack, messages, evt: gr.SelectData):
        # Click a history thumbnail to revert to that step (each edit = 1 stack frame + 2 messages).
        stack = list(stack or [])
        idx = evt.index
        if not stack or idx >= len(stack):
            return gr.update(), gr.update(), gr.update(), stack, messages
        stack = stack[: idx + 1]
        messages = (messages or [])[: 2 * idx]
        label = "original" if idx == 0 else f"step {idx}"
        return _rgb(stack[-1]), _gallery(stack), f"Reverted to {label}.", stack, messages

    with gr.Blocks(title="Lumina Agent") as demo:
        gr.Markdown("# ✦ Lumina Agent\nDeterministic AI photo retouching. **Click an object** on the left to retouch "
                    "just that part, or describe an edit for the whole image — the agent plans and applies it.")
        stack_state, msg_state = gr.State([]), gr.State([])
        sel_state, emb_state = gr.State(None), gr.State(None)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                src = gr.Image(type="numpy", sources=["upload"], height=440,
                               label="Original — click an object to select it")
            with gr.Column(scale=1):
                result = gr.Image(type="numpy", interactive=False, height=440, label="Result")

        prompt = gr.Textbox(label="Instruction", autofocus=True,
                            placeholder="e.g. sharpen the flower · brighten the subject · blur the background")
        with gr.Row():
            run_btn = gr.Button("✨ Apply edit", variant="primary", scale=2)
            undo_btn = gr.Button("↩ Undo")
            clear_btn = gr.Button("✕ Clear selection")

        info = gr.Markdown()
        history = gr.Gallery(label="History — click a step to revert", columns=8, height=120,
                             object_fit="cover", allow_preview=False)

        src.upload(on_upload, [src], [result, history, info, stack_state, msg_state, sel_state, emb_state])
        src.select(on_select, [stack_state, emb_state], [src, info, sel_state])
        clear_btn.click(clear_sel, [stack_state], [src, info, sel_state])
        run_btn.click(run_turn, [prompt, stack_state, msg_state, sel_state],
                      [result, history, info, stack_state, msg_state])
        undo_btn.click(undo, [stack_state, msg_state],
                       [result, history, info, stack_state, msg_state])
        history.select(on_history_select, [stack_state, msg_state],
                       [result, history, info, stack_state, msg_state])

    return demo
