import type { PlacedWord } from "../types";
import { FONT_FAMILY } from "./layout";

export interface ExportInput {
  words: PlacedWord[];
  width: number;
  height: number;
  background: string; // "transparent" or a CSS colour
  fontWeight: number;
}

/** Draw the laid-out cloud onto a canvas (the web font is already loaded in the page). */
export function drawToCanvas(input: ExportInput, scale = 2): HTMLCanvasElement {
  const { words, width, height, background, fontWeight } = input;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is not supported");

  ctx.scale(scale, scale);
  if (background !== "transparent") {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
  }
  ctx.translate(width / 2, height / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (const w of words) {
    ctx.save();
    ctx.translate(w.x, w.y);
    ctx.rotate((w.rotate * Math.PI) / 180);
    ctx.font = `${fontWeight} ${w.size}px "${FONT_FAMILY}", Tahoma, sans-serif`;
    ctx.fillStyle = w.color;
    ctx.fillText(w.text, 0, 0);
    ctx.restore();
  }
  return canvas;
}

export function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG export failed"))), "image/png"),
  );
}

export function buildSvg(input: ExportInput): string {
  const { words, width, height, background, fontWeight } = input;
  const esc = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const bg =
    background === "transparent"
      ? ""
      : `<rect width="100%" height="100%" fill="${esc(background)}"/>`;
  const texts = words
    .map(
      (w) =>
        `<text text-anchor="middle" transform="translate(${w.x.toFixed(1)},${w.y.toFixed(1)}) rotate(${w.rotate})" ` +
        `font-size="${w.size}" fill="${w.color}">${esc(w.text)}</text>`,
    )
    .join("\n    ");
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  ${bg}
  <g transform="translate(${width / 2},${height / 2})" font-family="${FONT_FAMILY}, Tahoma, sans-serif" font-weight="${fontWeight}">
    ${texts}
  </g>
</svg>`;
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyPng(canvas: HTMLCanvasElement): Promise<void> {
  if (!navigator.clipboard || typeof ClipboardItem === "undefined") {
    throw new Error("Clipboard is not available in this browser");
  }
  const blob = await canvasToBlob(canvas);
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}
