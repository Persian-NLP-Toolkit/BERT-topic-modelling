import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import type { CloudOptions, PlacedWord, Word } from "../types";
import { ensureFont, FONT_FAMILY, layoutCloud } from "../lib/layout";
import { buildSvg, drawToCanvas, type ExportInput } from "../lib/export";

export interface WordCloudHandle {
  toCanvas: (scale?: number) => HTMLCanvasElement | null;
  toSvg: () => string | null;
}

interface Props {
  words: Word[];
  options: CloudOptions;
  onWordClick: (text: string) => void;
  onPlaced?: (placed: number) => void;
  ref?: Ref<WordCloudHandle>;
}

export function WordCloud({ words, options, onWordClick, onPlaced, ref }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [placed, setPlaced] = useState<PlacedWord[]>([]);
  const [busy, setBusy] = useState(false);

  // Track the stage size (debounced) so the cloud re-flows on resize.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    let timer: number | undefined;
    const ro = new ResizeObserver(([entry]) => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        const { width, height } = entry.contentRect;
        setSize({ width: Math.round(width), height: Math.round(height) });
      }, 120);
    });
    ro.observe(el);
    return () => {
      window.clearTimeout(timer);
      ro.disconnect();
    };
  }, []);

  // Re-run the layout whenever data, options or size change.
  useEffect(() => {
    if (!size.width || !size.height || words.length === 0) {
      setPlaced([]);
      return;
    }
    let handle: ReturnType<typeof layoutCloud> | null = null;
    let alive = true;
    setBusy(true);
    ensureFont(options.fontWeight).then(() => {
      if (!alive) return;
      handle = layoutCloud(words, options, size.width, size.height);
      handle.promise.then((out) => {
        if (!alive) return;
        setPlaced(out);
        setBusy(false);
        onPlaced?.(out.length);
      });
    });
    return () => {
      alive = false;
      handle?.cancel();
    };
    // onPlaced is intentionally not a dependency (callers pass inline functions)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [words, options, size.width, size.height]);

  const exportInput = (): ExportInput => ({
    words: placed,
    width: size.width,
    height: size.height,
    background: options.background,
    fontWeight: options.fontWeight,
  });

  useImperativeHandle(ref, () => ({
    toCanvas: (scale = 2) => (placed.length ? drawToCanvas(exportInput(), scale) : null),
    toSvg: () => (placed.length ? buildSvg(exportInput()) : null),
  }));

  const checker = options.background === "transparent";

  return (
    <div
      ref={wrapRef}
      className={`stage${checker ? " stage--transparent" : ""}`}
      style={checker ? undefined : { background: options.background }}
    >
      {size.width > 0 && (
        <svg
          className="cloud"
          width={size.width}
          height={size.height}
          viewBox={`0 0 ${size.width} ${size.height}`}
          role="img"
          aria-label="ابر کلمات"
        >
          <g
            transform={`translate(${size.width / 2},${size.height / 2})`}
            fontFamily={`${FONT_FAMILY}, Tahoma, sans-serif`}
            fontWeight={options.fontWeight}
          >
            {placed.map((w, i) => (
              <text
                key={w.text}
                className="cloud-word"
                textAnchor="middle"
                fill={w.color}
                style={{
                  fontSize: `${w.size}px`,
                  transform: `translate(${w.x}px, ${w.y}px) rotate(${w.rotate}deg)`,
                  animationDelay: `${Math.min(i * 12, 600)}ms`,
                }}
                onClick={() => onWordClick(w.text)}
              >
                <title>{`${w.text} — وزن ${w.score.toLocaleString("fa-IR", {
                  maximumFractionDigits: 1,
                })} (کلیک برای حذف)`}</title>
                {w.text}
              </text>
            ))}
          </g>
        </svg>
      )}
      {busy && placed.length === 0 && <div className="stage-hint">در حال چیدن کلمات…</div>}
    </div>
  );
}
