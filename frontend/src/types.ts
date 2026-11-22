export interface Word {
  text: string;
  weight: number;
}

export type Method = "frequency" | "bertopic";
export type Ngram = "1" | "2" | "1-2";

export interface AnalyzeParams {
  column: string;
  method: Method;
  ngram: Ngram;
  maxWords: number;
  minCount: number;
  stopwords: string;
}

export interface AnalyzeResponse {
  words: Word[];
  method: Method;
  documents: number;
  tokens: number;
  topics: number;
  warnings: string[];
  columns: string[];
  column: string | null;
}

export interface Health {
  status: string;
  version: string;
  bertopic: boolean;
  max_upload_mb: number;
}

export type PaletteName = "vivid" | "ocean" | "sunset" | "forest" | "berry" | "mono";
export type RotationMode = "horizontal" | "mixed" | "diagonal" | "random";
export type Spiral = "archimedean" | "rectangular";

export interface CloudOptions {
  palette: PaletteName;
  rotation: RotationMode;
  spiral: Spiral;
  /** largest font as a fraction of the shorter side of the canvas */
  fontScale: number;
  fontWeight: 400 | 700 | 900;
  padding: number;
  background: string;
  visibleCount: number;
  seed: number;
}

/** A word after layout; x / y are relative to the canvas centre. */
export interface PlacedWord {
  text: string;
  score: number;
  size: number;
  x: number;
  y: number;
  rotate: number;
  color: string;
}
