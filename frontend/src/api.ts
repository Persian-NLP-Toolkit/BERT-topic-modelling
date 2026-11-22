import type { AnalyzeParams, AnalyzeResponse, Health } from "./types";

const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("، ");
  } catch {
    /* not JSON */
  }
  return `خطای سرور (${res.status})`;
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${BASE}/api/health`);
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function analyze(
  file: File,
  params: AnalyzeParams,
  signal?: AbortSignal,
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  if (params.column) form.append("column", params.column);
  form.append("method", params.method);
  form.append("ngram", params.ngram);
  form.append("max_words", String(params.maxWords));
  form.append("min_count", String(params.minCount));
  if (params.stopwords.trim()) form.append("stopwords", params.stopwords);

  let res: Response;
  try {
    res = await fetch(`${BASE}/api/wordcloud`, { method: "POST", body: form, signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new Error("اتصال به سرور برقرار نشد. آیا bert-wordcloud serve در حال اجراست؟");
  }
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}
