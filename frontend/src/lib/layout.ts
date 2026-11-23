import CloudLayout, { type PlacedSprite } from "d3-cloud";
import type { CloudOptions, PlacedWord, Word } from "../types";
import { colorFor, fontRange, mulberry32, rotationFor, scaleSizes } from "./cloud";

export const FONT_FAMILY = "Vazirmatn";
/** Words placed per task; yielding between chunks keeps the UI responsive (setTimeout also runs in background tabs, unlike rAF). */
const CHUNK = 25;
/** A word that does not fit is retried this many times, a bit smaller each time, before it is dropped. */
const SHRINK_STEPS = 4;
const SHRINK_FACTOR = 0.82;
/** Fraction of the stage the finished cloud may cover, and the most it may be enlarged to get there. */
const FIT_FILL = 0.94;
const FIT_MAX_ZOOM = 1.6;

interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Uniformly scale the laid-out words so the cloud fills the stage and is centred.
 * Scaling every position and size by the same factor keeps the layout collision-free.
 */
export function fitToStage(
  words: PlacedWord[],
  boxes: Box[],
  width: number,
  height: number,
  fill = FIT_FILL,
  maxZoom = FIT_MAX_ZOOM,
): PlacedWord[] {
  if (words.length === 0) return words;
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
  words.forEach((w, i) => {
    const b = boxes[i];
    x0 = Math.min(x0, w.x + b.x0);
    y0 = Math.min(y0, w.y + b.y0);
    x1 = Math.max(x1, w.x + b.x1);
    y1 = Math.max(y1, w.y + b.y1);
  });
  const bw = x1 - x0;
  const bh = y1 - y0;
  if (bw <= 0 || bh <= 0) return words;
  const k = Math.max(0.5, Math.min(maxZoom, (width * fill) / bw, (height * fill) / bh));
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const r = (v: number) => Math.round(v * 100) / 100;
  return words.map((w) => ({
    ...w,
    x: r((w.x - cx) * k),
    y: r((w.y - cy) * k),
    size: r(w.size * k),
  }));
}

interface Datum {
  text: string;
  score: number;
  color: string;
}

export interface LayoutHandle {
  promise: Promise<PlacedWord[]>;
  cancel: () => void;
}

/** Make sure the web font (incl. the Arabic-script subset) is ready before measuring. */
export async function ensureFont(weight: number): Promise<void> {
  if (typeof document === "undefined" || !document.fonts) return;
  try {
    await document.fonts.load(`${weight} 32px "${FONT_FAMILY}"`, "ابر کلمات");
  } catch {
    /* fall back to whatever is available */
  }
}

export function layoutCloud(
  words: Word[],
  options: CloudOptions,
  width: number,
  height: number,
): LayoutHandle {
  const [minSize, maxSize] = fontRange(width, height, options.fontScale);
  const sizes = scaleSizes(
    words.map((w) => w.weight),
    minSize,
    maxSize,
  );

  const layout = new CloudLayout()
    .size([width, height])
    .overflow(false)
    .strategy(options.spiral)
    .random(mulberry32(options.seed));

  // Largest first gives the densest, most readable packing.
  const order = words.map((_, i) => i).sort((a, b) => sizes[b] - sizes[a]);

  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const promise = new Promise<PlacedWord[]>((resolve) => {
    const placed: PlacedWord[] = [];
    const boxes: Box[] = [];
    let next = 0;

    const step = () => {
      if (cancelled) return;
      const end = Math.min(next + CHUNK, order.length);
      for (; next < end; next++) {
        const i = order[next];
        const w = words[i];
        const datum: Datum = { text: w.text, score: w.weight, color: colorFor(i, options.palette, options.background) };
        const rotate = rotationFor(options.rotation, w.text, options.seed);
        // Try the intended size first, then progressively smaller ones so that
        // fewer words are dropped and gaps get filled.
        let p: PlacedSprite<Datum, never> | null = null;
        let size = sizes[i];
        for (let attempt = 0; attempt <= SHRINK_STEPS && !p; attempt++) {
          const sprite = layout.getSprite(w.text, {
            font: FONT_FAMILY,
            fontWeight: options.fontWeight,
            size,
            rotate,
            padding: options.padding,
            datum,
          });
          if (sprite) p = layout.place(sprite);
          const smaller = Math.round(size * SHRINK_FACTOR);
          if (smaller < minSize * 0.8) break;
          size = smaller;
        }
        if (!p) continue;
        placed.push({
          text: p.datum.text,
          score: p.datum.score,
          color: p.datum.color,
          size: p.size,
          x: p.x,
          y: p.y,
          rotate: p.rotate,
        });
        boxes.push({ x0: p.x0, y0: p.y0, x1: p.x1, y1: p.y1 });
      }
      if (next < order.length) timer = setTimeout(step, 0);
      else resolve(fitToStage(placed, boxes, width, height));
    };
    step();
  });

  return {
    promise,
    cancel: () => {
      cancelled = true;
      clearTimeout(timer);
    },
  };
}
