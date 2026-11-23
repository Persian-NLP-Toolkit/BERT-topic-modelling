import { describe, expect, it } from "vitest";
import { fitToStage } from "./layout";
import type { PlacedWord } from "../types";

const word = (x: number, y: number, size: number): PlacedWord => ({
  text: `w${x}`,
  score: 1,
  color: "#000",
  rotate: 0,
  x,
  y,
  size,
});

describe("fitToStage", () => {
  it("enlarges and centres a small cloud, capped by maxZoom", () => {
    // Cloud spans x: 0..100, y: -10..10 (off-centre, small).
    const out = fitToStage([word(50, 0, 20)], [{ x0: -50, x1: 50, y0: -10, y1: 10 }], 1000, 500, 1, 1.5);
    expect(out[0].x).toBe(0);
    expect(out[0].y).toBe(0);
    expect(out[0].size).toBe(30);
  });

  it("fits the tighter dimension", () => {
    const out = fitToStage([word(0, 0, 10)], [{ x0: -100, x1: 100, y0: -25, y1: 25 }], 300, 300, 1, 5);
    expect(out[0].size).toBe(15); // limited by width: 300 / 200
  });

  it("returns an empty list untouched", () => {
    expect(fitToStage([], [], 100, 100)).toEqual([]);
  });
});
