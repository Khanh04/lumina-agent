from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Single source of truth for edit targets. segmentor.py imports this rather than
# redefining it (the v2 doc had two divergent copies — one with "face", one without).
TargetType = Literal["global", "subject", "background", "face", "sky", "radial_center", "selection"]


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


class ActionCall(BaseModel):
    tool_name: str = Field(..., description="Name of the edit op, e.g. 'adjust_exposure'")
    target: str = Field(
        "global",
        description="Region to edit: a preset (global, subject, background, face, sky, radial_center) "
        "or a 'region:<name>' string returned by the find_region tool for a specific named object.",
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Op-specific numeric parameters")
    rationale: str = Field(..., description="Why this edit was chosen")


class RetouchRecipe(BaseModel):
    summary: str = Field(..., description="One-line summary of the planned edits")
    actions: list[ActionCall] = Field(default_factory=list)


class RetouchResponse(BaseModel):
    session_id: str
    recipe: RetouchRecipe
    telemetry: ImageTelemetry
    processed_image_base64: str
    execution_time_ms: float
    # Any actions the model emitted that failed validation and were skipped (never a 500).
    skipped: list[str] = Field(default_factory=list)
