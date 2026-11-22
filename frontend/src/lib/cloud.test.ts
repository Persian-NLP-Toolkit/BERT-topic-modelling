import { describe, expect, it } from "vitest";
import {
  colorFor,
  contrastRatio,
  fileBaseName,
  fontRange,
  mulberry32,
  PALETTES,
  readableColor,
  rotationFor,
  scaleSizes,
  visibleWords,
} from "./cloud";

describe("scaleSizes", () => {
  it("maps min and max weight to min and max size", () => {
    const sizes = scaleSizes([1, 4, 9], 10, 70);
    expect(sizes[0]).toBe(10);
    expect(sizes[2]).toBe(70);
    expect(sizes[1]).toBe(40);
  });

  it("handles equal weights and empty input", () => {
    expect(scaleSizes([5, 5], 10, 30)).toEqual([20, 20]);
    expect(scaleSizes([], 10, 30)).toEqual([]);
  });
});

describe("rotationFor", () => {
  it("is deterministic per word and seed", () => {
    expect(rotationFor("random", "سلام", 1)).toBe(rotationFor("random", "سلام", 1));
  });

  it("respects the mode", () => {
    expect(rotationFor("horizontal", "x", 3)).toBe(0);
    for (const word of ["الف", "ب", "پ", "ت", "ث"]) {
      expect([0, 90]).toContain(rotationFor("mixed", word, 7));
      expect([-45, 0, 45]).toContain(rotationFor("diagonal", word, 7));
      const r = rotationFor("random", word, 7);
      expect(r).toBeGreaterThanOrEqual(-60);
      expect(r).toBeLessThanOrEqual(60);
    }
  });
});

describe("helpers", () => {
  it("mulberry32 is seedable", () => {
    expect(mulberry32(42)()).toBe(mulberry32(42)());
  });

  it("fontRange scales with canvas", () => {
    const [min, max] = fontRange(1000, 500, 0.2);
    expect(max).toBe(100);
    expect(min).toBe(14);
  });

  it("colorFor cycles through the palette", () => {
    const n = PALETTES.ocean.colors.length;
    expect(colorFor(n, "ocean")).toBe(colorFor(0, "ocean"));
  });

  it("visibleWords hides and limits", () => {
    const words = [
      { text: "a", weight: 3 },
      { text: "b", weight: 2 },
      { text: "c", weight: 1 },
    ];
    expect(visibleWords(words, new Set(["a"]), 1)).toEqual([{ text: "b", weight: 2 }]);
  });

  it("fileBaseName strips the extension", () => {
    expect(fileBaseName("data.csv")).toBe("wordcloud-data");
    expect(fileBaseName(undefined)).toBe("wordcloud");
  });
});

describe("readableColor", () => {
  it("keeps colours that already contrast enough", () => {
    expect(readableColor("#e63946", "#ffffff")).toBe("#e63946");
  });

  it("lightens dark colours on a dark background", () => {
    const c = readableColor("#264653", "#111827");
    expect(c).not.toBe("#264653");
    expect(contrastRatio(c, "#111827")).toBeGreaterThanOrEqual(3);
  });

  it("darkens pale colours on a light background and treats transparent as white", () => {
    const c = readableColor("#e9c46a", "transparent");
    expect(contrastRatio(c, "#ffffff")).toBeGreaterThanOrEqual(3);
  });
});
