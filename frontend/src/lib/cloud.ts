import type { PaletteName, RotationMode, Word } from "../types";

export const PALETTES: Record<PaletteName, { label: string; colors: string[] }> = {
  vivid: {
    label: "رنگارنگ",
    colors: ["#e63946", "#2a9d8f", "#f4a261", "#3a86ff", "#8338ec", "#ff006e", "#264653", "#e9a800"],
  },
  ocean: {
    label: "اقیانوس",
    colors: ["#03045e", "#0077b6", "#0096c7", "#00b4d8", "#023e8a", "#48cae4"],
  },
  sunset: {
    label: "غروب",
    colors: ["#9d0208", "#d00000", "#dc2f02", "#e85d04", "#f48c06", "#6a040f"],
  },
  forest: {
    label: "جنگل",
    colors: ["#081c15", "#1b4332", "#2d6a4f", "#40916c", "#52b788", "#74c69d"],
  },
  berry: {
    label: "تمشکی",
    colors: ["#5a189a", "#7b2cbf", "#9d4edd", "#c9184a", "#ff4d6d", "#3c096c"],
  },
  mono: {
    label: "تک‌رنگ",
    colors: ["#111827", "#374151", "#4b5563", "#6b7280"],
  },
};

export const ROTATIONS: Record<RotationMode, string> = {
  horizontal: "افقی",
  mixed: "افقی و عمودی",
  diagonal: "مورب",
  random: "تصادفی",
};

/** Small, fast, seedable PRNG. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Rotation of a word, stable for the same (text, seed) so that words keep
 * their angle when the cloud is re-laid out (smooth transitions).
 */
export function rotationFor(mode: RotationMode, text: string, seed: number): number {
  const r = mulberry32(hashString(text) ^ seed)();
  switch (mode) {
    case "horizontal":
      return 0;
    case "mixed":
      return r < 0.75 ? 0 : 90;
    case "diagonal":
      return [-45, 0, 0, 45][Math.floor(r * 4)];
    case "random":
      return Math.round((r * 120 - 60) / 15) * 15;
  }
}

/** Map weights to font sizes using a square-root scale. */
export function scaleSizes(weights: number[], minSize: number, maxSize: number): number[] {
  if (weights.length === 0) return [];
  const roots = weights.map((w) => Math.sqrt(Math.max(w, 0)));
  const lo = Math.min(...roots);
  const hi = Math.max(...roots);
  if (hi === lo) return weights.map(() => Math.round((minSize + maxSize) / 2));
  return roots.map((r) => Math.round(minSize + ((r - lo) / (hi - lo)) * (maxSize - minSize)));
}

export function fontRange(width: number, height: number, fontScale: number): [number, number] {
  const max = Math.max(20, Math.round(Math.min(width, height) * fontScale));
  const min = Math.max(10, Math.round(max * 0.14));
  return [min, max];
}

// ---------- colour contrast ----------
type RGB = [number, number, number];

function hexToRgb(hex: string): RGB | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex([r, g, b]: RGB): string {
  return "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
}

function luminance([r, g, b]: RGB): number {
  const lin = (v: number) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

export function contrastRatio(a: string, b: string): number {
  const ra = hexToRgb(a);
  const rb = hexToRgb(b);
  if (!ra || !rb) return 21;
  const [l1, l2] = [luminance(ra), luminance(rb)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

/** Nudge `color` towards white (dark bg) or black (light bg) until it reaches `min` contrast. */
export function readableColor(color: string, background: string, min = 3): string {
  const bg = background === "transparent" ? "#ffffff" : background;
  const c = hexToRgb(color);
  const b = hexToRgb(bg);
  if (!c || !b || contrastRatio(color, bg) >= min) return color;
  const target: RGB = luminance(b) < 0.4 ? [255, 255, 255] : [0, 0, 0];
  for (let t = 0.1; t <= 1; t += 0.1) {
    const mixed = rgbToHex(c.map((v, i) => v + (target[i] - v) * t) as RGB);
    if (contrastRatio(mixed, bg) >= min) return mixed;
  }
  return rgbToHex(target);
}

export function colorFor(index: number, palette: PaletteName, background = "#ffffff"): string {
  const colors = PALETTES[palette].colors;
  return readableColor(colors[index % colors.length], background);
}

export function visibleWords(words: Word[], hidden: Set<string>, count: number): Word[] {
  return words.filter((w) => !hidden.has(w.text)).slice(0, count);
}

export function fileBaseName(name: string | undefined): string {
  if (!name) return "wordcloud";
  const base = name.replace(/\.[^.]+$/, "").trim();
  return base ? `wordcloud-${base}` : "wordcloud";
}
