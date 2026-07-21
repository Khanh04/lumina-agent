import { useRef, useState } from "react";

interface Props {
  afterUrl: string;
  beforeUrl: string | null; // previous version, revealed by the seam; null for the original
  maskUrl: string | null; // active click-selection, tinted amber
  loading: boolean;
  onSelect: (x: number, y: number) => void; // original-image pixel coords
}

export default function Stage({ afterUrl, beforeUrl, maskUrl, loading, onSelect }: Props) {
  const [seam, setSeam] = useState(50); // percent from left
  const [dragging, setDragging] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const draggingSeam = useRef(false);

  function click(e: React.MouseEvent<HTMLImageElement>) {
    if (draggingSeam.current) return;
    const img = e.currentTarget;
    const r = img.getBoundingClientRect();
    const x = Math.round(((e.clientX - r.left) / r.width) * img.naturalWidth);
    const y = Math.round(((e.clientY - r.top) / r.height) * img.naturalHeight);
    onSelect(x, y);
  }

  function startSeamDrag(e: React.PointerEvent) {
    e.preventDefault();
    draggingSeam.current = true;
    setDragging(true);
    const move = (ev: PointerEvent) => {
      const r = wrapRef.current!.getBoundingClientRect();
      setSeam(Math.min(100, Math.max(0, ((ev.clientX - r.left) / r.width) * 100)));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      setDragging(false);
      setTimeout(() => (draggingSeam.current = false), 0); // let the trailing click be ignored
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  return (
    <div className="stage-wrap" ref={wrapRef}>
      <img className="base" src={afterUrl} onClick={click} alt="edited" />
      {beforeUrl && (
        <>
          <img
            className="before"
            src={beforeUrl}
            alt="original"
            style={{ clipPath: `inset(0 ${100 - seam}% 0 0)` }}
          />
          <div
            className={`seam ${dragging ? "dragging" : ""}`}
            style={{ left: `${seam}%` }}
            onPointerDown={startSeamDrag}
          >
            <span className="seam-grip" />
          </div>
        </>
      )}
      {maskUrl && <div className="mask" style={{ WebkitMaskImage: `url(${maskUrl})`, maskImage: `url(${maskUrl})` }} />}
      {loading && <div className="scan" />}
      {beforeUrl && <div className="corner-label">before · after</div>}
    </div>
  );
}
