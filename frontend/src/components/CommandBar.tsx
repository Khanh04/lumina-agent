import { useState } from "react";

interface Props {
  loading: boolean;
  selectionActive: boolean;
  onApply: (prompt: string) => void;
  onClearSelection: () => void;
}

export default function CommandBar({ loading, selectionActive, onApply, onClearSelection }: Props) {
  const [text, setText] = useState("");

  function submit() {
    const p = text.trim();
    if (!p || loading) return;
    onApply(p);
    setText("");
  }

  return (
    <div className="commandbar">
      {selectionActive && (
        <div className="chip">
          region selected — edits apply here
          <button onClick={onClearSelection} title="Clear selection">
            ✕
          </button>
        </div>
      )}
      <div className="commandbar-row">
        <span className="glyph">⌘</span>
        <input
          autoFocus
          value={text}
          placeholder={selectionActive ? "brighten this · warm it · sharpen it" : "describe an edit — e.g. brighten the subject, blur the background"}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
        <button className="apply" onClick={submit} disabled={loading || !text.trim()}>
          {loading ? "Developing…" : "Apply"}
        </button>
      </div>
    </div>
  );
}
