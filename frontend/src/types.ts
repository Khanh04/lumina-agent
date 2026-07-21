// Mirror of app/schemas.py (RetouchResponse and friends).

export interface RegionalMetrics {
  subject_brightness: number | null;
  background_brightness: number | null;
  sky_brightness: number | null;
  has_human_subject: boolean;
}

export interface ImageTelemetry {
  mean_brightness: number;
  contrast_std: number;
  sharpness_laplacian: number;
  is_underexposed: boolean;
  is_overexposed: boolean;
  regional: RegionalMetrics;
}

export interface ActionCall {
  tool_name: string;
  target: string;
  parameters: Record<string, number | string>;
  rationale: string;
}

export interface RetouchRecipe {
  summary: string;
  actions: ActionCall[];
}

export interface RetouchResponse {
  session_id: string;
  recipe: RetouchRecipe;
  telemetry: ImageTelemetry;
  processed_image_base64: string;
  image_format: "jpeg" | "png";
  execution_time_ms: number;
  skipped: string[];
}

// One entry in the client-side version history. Index 0 is the original upload (no recipe).
export interface Version {
  url: string; // data URL of the image at this step
  recipe?: RetouchRecipe;
  telemetry?: ImageTelemetry;
  skipped?: string[];
}
