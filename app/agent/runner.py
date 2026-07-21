"""The LLM planner. Emits a validated RetouchRecipe (structured output) and can call one
tool, find_region, to locate a specific object the user named and register it as an
editable region. The system prompt documents the ops/ranges the allowlist in engine.py
accepts; masks are produced by the engine (presets), the find_region tool (named objects),
or a UI click/name selection (override).
"""
from dataclasses import dataclass, field

import numpy as np
from pydantic_ai import Agent, RunContext

from app.config import settings
from app.schemas import ImageTelemetry, RetouchRecipe


@dataclass
class RetouchDeps:
    """Per-run context the agent (and its tools) operate on."""
    telemetry: ImageTelemetry
    image: np.ndarray                       # BGR working image, for grounding
    grounder: object                        # TextGrounder (YOLO-World)
    sam: object                             # MobileSAM
    regions: dict = field(default_factory=dict)   # name -> mask, filled by find_region
    embedding: object = None                # cached SAM embedding (computed on first find)


_SYSTEM_PROMPT = """You are a professional, deterministic photo-retouching agent.

You are given mathematical telemetry about an image and a user request. Plan a set of
targeted edits and return them as a structured recipe. Do not invent tools or targets.

Available edit ops (use these exact tool_name values and parameter keys):
- adjust_exposure  parameters: {"ev": float in [-2.0, 2.0]}       # + brightens, - darkens
- adjust_temperature  parameters: {"shift": int in [-50, 50]}      # + warms, - cools
- unsharp_mask  parameters: {"amount": float in [0.0, 2.5], "radius": int in [1, 9]}  # sharpen detail/crispness
- adjust_saturation  parameters: {"factor": float in [0.0, 3.0]}   # 0=greyscale, 1=unchanged, >1=more vivid
- adjust_contrast  parameters: {"factor": float in [0.5, 2.0]}     # <1 flatter, >1 punchier
- gaussian_blur  parameters: {"radius": int in [1, 51]}            # soften; on a region = blur it (e.g. background bokeh)
- auto_white_balance  parameters: {}                                # neutralize a colour cast (gray-world)
- clahe  parameters: {"clip_limit": float in [1.0, 5.0]}           # local contrast / "clarity"

Each action's "target" is the region it applies to. Preset targets:
"global", "subject", "background", "face", "sky", "radial_center". Default to "global".

If the user asks to edit a SPECIFIC named object that is not a preset (e.g. "the flower",
"the red car", "her dress"), first call the find_region tool with that description. It
returns a target string like "region:flower" — put that exact string in the target of the
actions that should affect only that object. If find_region reports it wasn't found, fall
back to the most sensible preset (usually "global") and say so in the summary.

Keep edits minimal and justified. Put a short reason in each action's rationale, and a
one-line overall summary. If no edit is warranted, return an empty actions list.
"""


class RetouchAgentRunner:
    def __init__(self, model_name: str | None = None):
        self.agent = Agent(
            model_name or settings.model_name,
            deps_type=RetouchDeps,
            output_type=RetouchRecipe,
            system_prompt=_SYSTEM_PROMPT,
        )

        @self.agent.system_prompt
        def inject_telemetry(ctx: RunContext[RetouchDeps]) -> str:
            t = ctx.deps.telemetry
            return (
                f"Telemetry — mean_brightness={t.mean_brightness}, contrast_std={t.contrast_std}, "
                f"sharpness={t.sharpness_laplacian}, underexposed={t.is_underexposed}, "
                f"overexposed={t.is_overexposed}. Regional: {t.regional.model_dump()}."
            )

        @self.agent.tool
        def find_region(ctx: RunContext[RetouchDeps], description: str) -> str:
            """Locate a specific object the user referred to and register it as an editable
            region. Call this before issuing edits that should apply only to that object.

            Args:
                description: A short noun phrase for the object, e.g. "flower", "red car", "hat".

            Returns the target string (e.g. "region:flower") to place on the relevant actions,
            or a message if the object could not be found.
            """
            deps = ctx.deps
            box = deps.grounder.detect(deps.image, description)
            if box is None:
                return f"Could not find '{description}'. Use a preset target such as 'global' instead."
            if deps.embedding is None:
                deps.embedding = deps.sam.embed(deps.image)  # encode once per run, only if grounding is used
            name = description.strip().lower().replace(" ", "_")
            deps.regions[name] = deps.sam.mask_from_box(deps.embedding, box)
            return f"Found '{description}'. Set target to 'region:{name}' on the actions that should edit it."
