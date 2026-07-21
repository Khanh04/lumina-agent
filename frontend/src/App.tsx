import { useEffect, useRef, useState } from "react";
import { chat, clearSelection, createSession, jpeg, png, revert, selectRegion } from "./api";
import type { Version } from "./types";
import Stage from "./components/Stage";
import CommandBar from "./components/CommandBar";
import Timeline from "./components/Timeline";
import RecipeHUD from "./components/RecipeHUD";

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
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
      setVersions((v) => [
        ...v,
        { url: jpeg(r.processed_image_base64), recipe: r.recipe, telemetry: r.telemetry, skipped: r.skipped },
      ]);
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
      setVersions((v) => v.slice(0, step + 1));
    } catch (e) {
      setError(msg(e));
    }
  }

  const empty = versions.length === 0;
  const current = versions[versions.length - 1];
  const before = versions.length > 1 ? versions[versions.length - 2].url : null;

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
          onRevert={onRevert}
          onUndo={() => onRevert(versions.length - 2)}
          canUndo={versions.length > 1}
          download={current?.url ?? null}
        />
      )}

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
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
            />
          </div>
        ) : (
          <>
            <Stage afterUrl={current.url} beforeUrl={before} maskUrl={maskUrl} loading={loading} onSelect={onSelect} />
            {current && <RecipeHUD version={current} />}
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
