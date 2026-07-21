import { Fragment } from "react";

interface Props {
  count: number; // number of versions (original + edits)
  current: number; // index of the active step
  onRevert: (step: number) => void;
  onUndo: () => void;
  canUndo: boolean;
  download: string | null; // current image data URL, or null
}

export default function Timeline({ count, current, onRevert, onUndo, canUndo, download }: Props) {
  return (
    <div className="timeline">
      <span className="wordmark">
        Lu<b>·</b>mina
      </span>
      <div className="timeline-track">
        {Array.from({ length: count }).map((_, i) => (
          <Fragment key={i}>
            {i > 0 && <span className="node-link" />}
            <button
              className={`node ${i === current ? "current" : ""}`}
              title={i === 0 ? "original" : `step ${i}`}
              onClick={() => i !== current && onRevert(i)}
            />
          </Fragment>
        ))}
      </div>
      <div className="tools">
        <button className="icon-btn" onClick={onUndo} disabled={!canUndo}>
          ⌫ undo
        </button>
        {download && (
          <a className="icon-btn" href={download} download="lumina.jpg">
            ↓ save
          </a>
        )}
      </div>
    </div>
  );
}
