import CloudLayout from "d3-cloud";
import type { CloudOptions, PlacedWord, Word } from "../types";
import { colorFor, fontRange, mulberry32, rotationFor, scaleSizes } from "./cloud";

export const FONT_FAMILY = "Vazirmatn";
/** Words placed per task; yielding between chunks keeps the UI responsive (setTimeout also runs in background tabs, unlike rAF). */
const CHUNK = 25;

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
    let next = 0;

    const step = () => {
      if (cancelled) return;
      const end = Math.min(next + CHUNK, order.length);
      for (; next < end; next++) {
        const i = order[next];
        const w = words[i];
        const datum: Datum = { text: w.text, score: w.weight, color: colorFor(i, options.palette, options.background) };
        const sprite = layout.getSprite(w.text, {
          font: FONT_FAMILY,
          fontWeight: options.fontWeight,
          size: sizes[i],
          rotate: rotationFor(options.rotation, w.text, options.seed),
          padding: options.padding,
          datum,
        });
        if (!sprite) continue;
        const p = layout.place(sprite);
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
      }
      if (next < order.length) timer = setTimeout(step, 0);
      else resolve(placed);
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
