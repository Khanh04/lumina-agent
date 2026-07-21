import { useEffect, useRef, useState } from "react";
import { chat, clearSelection, createSession, dataUrl, png, revert, selectRegion } from "./api";
import type { Version } from "./types";
import Stage from "./components/Stage";
import CommandBar from "./components/CommandBar";
import Timeline from "./components/Timeline";
import RecipeHUD from "./components/RecipeHUD";

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [current, setCurrent] = useState(0); // index into versions of the active step
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 4500);
    return () => clearTimeout(t);
  }, [error]);

  async function onFile(file: File) {
    if (!file.type.startsWith("image/")) return;
    setVersions([{ url: URL.createObjectURL(file) }]); // instant local preview
    setCurrent(0);
    setMaskUrl(null);
    setSessionId(null);
    setError(null);
    try {
      const r = await createSession(file);
      setSessionId(r.session_id);
    } catch (e) {
      setError(msg(e));
    }
  }

  async function onApply(prompt: string) {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const r = await chat(sessionId, prompt);
      // A new edit branches off the active step — steps after it (if we'd reverted) are
      // abandoned, matching the backend's push_edit trim-then-append behavior.
      setVersions((v) => [
        ...v.slice(0, current + 1),
        {
          url: dataUrl(r.processed_image_base64, r.image_format),
          recipe: r.recipe,
          telemetry: r.telemetry,
          skipped: r.skipped,
        },
      ]);
      setCurrent((c) => c + 1);
    } catch (e) {
      setError(msg(e));
    } finally {
      setLoading(false);
    }
  }

  async function onSelect(x: number, y: number) {
    if (!sessionId) return;
    try {
      const r = await selectRegion(sessionId, x, y);
      setMaskUrl(png(r.mask_base64));
    } catch (e) {
      setError(msg(e));
    }
  }

  function onClearSelection() {
    setMaskUrl(null);
    if (sessionId) clearSelection(sessionId).catch(() => {});
  }

  async function onRevert(step: number) {
    if (!sessionId) return;
    try {
      await revert(sessionId, step);
      setCurrent(step); // non-destructive — later steps stay in `versions` for redo
    } catch (e) {
      setError(msg(e));
    }
  }

  const empty = versions.length === 0;
  const active = versions[current];
  const before = current > 0 ? versions[current - 1].url : null;

  return (
    <div
      className="app"
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
      }}
    >
      {!empty && (
        <Timeline
          count={versions.length}
          current={current}
          onRevert={onRevert}
          onUndo={() => onRevert(current - 1)}
          canUndo={current > 0}
          download={active?.url ?? null}
          onNewPhoto={() => fileInput.current?.click()}
        />
      )}

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />

      <div className="stage">
        {empty ? (
          <div className={`dropzone ${over ? "over" : ""}`}>
            <h1>Lumina</h1>
            <p>
              Deterministic photo retouching. Drop a photo, then describe an edit — the agent plans bounded, math-only
              adjustments and shows exactly what it changed.
            </p>
            <button className="pick" onClick={() => fileInput.current?.click()}>
              Drop a photo · or browse
            </button>
          </div>
        ) : (
          <>
            <Stage afterUrl={active.url} beforeUrl={before} maskUrl={maskUrl} loading={loading} onSelect={onSelect} />
            {active && <RecipeHUD version={active} />}
            <CommandBar
              loading={loading}
              selectionActive={maskUrl !== null}
              onApply={onApply}
              onClearSelection={onClearSelection}
            />
          </>
        )}
        {error && (
          <div className="toast" onClick={() => setError(null)}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
