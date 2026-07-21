import { useState } from "react";
import type { Version } from "../types";
import type { ActionCall } from "../types";

// Compact meter-style label for one action, e.g. "EV +0.40 ▸SUBJECT".
const TOOL_LABEL: Record<string, string> = {
  adjust_exposure: "EV",
  adjust_temperature: "TEMP",
  unsharp_mask: "SHARP",
  adjust_saturation: "SAT",
  adjust_contrast: "CON",
  gaussian_blur: "BLUR",
  auto_white_balance: "AWB",
  clahe: "CLAHE",
};

function firstParam(a: ActionCall): string {
  const v = Object.values(a.parameters)[0];
  if (typeof v === "number") return v > 0 ? `+${v}` : `${v}`;
  return v != null ? `${v}` : "";
}

function region(target: string): string {
  const name = target.startsWith("region:") ? target.slice(7) : target;
  return name === "global" ? "" : `▸${name.toUpperCase()}`;
}

function compact(a: ActionCall): string {
  return [TOOL_LABEL[a.tool_name] ?? a.tool_name, firstParam(a), region(a.target)].filter(Boolean).join(" ");
}

export default function RecipeHUD({ version }: { version: Version }) {
  const [open, setOpen] = useState(false);
  if (!version.recipe) return null;
  const { recipe, telemetry, skipped } = version;

  return (
    <div className="hud">
      <div className="hud-strip" onClick={() => setOpen((o) => !o)} title="Show recipe & telemetry">
        {recipe.actions.length === 0 && <span>no edits</span>}
        {recipe.actions.map((a, i) => (
          <span key={i}>
            {i > 0 && <span className="sep"> · </span>}
            {compact(a)}
          </span>
        ))}
        <span className="sep">{open ? "▴" : "▾"}</span>
      </div>
      {open && (
        <div className="hud-body">
          <div className="hud-summary">{recipe.summary}</div>
          {recipe.actions.map((a, i) => (
            <div className="row" key={i}>
              <span className="k">{compact(a)}</span>
              <span>{a.rationale}</span>
            </div>
          ))}
          {telemetry && (
            <>
              <div className="row">
                <span className="k">brightness</span>
                <span>{telemetry.mean_brightness}</span>
              </div>
              <div className="row">
                <span className="k">contrast</span>
                <span>{telemetry.contrast_std}</span>
              </div>
              <div className="row">
                <span className="k">sharpness</span>
                <span>{telemetry.sharpness_laplacian}</span>
              </div>
            </>
          )}
          {skipped?.map((s, i) => (
            <div className="row warn" key={`s${i}`}>
              <span>⚠ skipped</span>
              <span>{s}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
