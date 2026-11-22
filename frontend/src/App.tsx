import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { analyze, getHealth } from "./api";
import { FileDrop } from "./components/FileDrop";
import { Icon, type IconName } from "./components/Icon";
import { WordCloud, type WordCloudHandle } from "./components/WordCloud";
import { WordList } from "./components/WordList";
import { fileBaseName, PALETTES, ROTATIONS, visibleWords } from "./lib/cloud";
import { canvasToBlob, copyPng, downloadBlob } from "./lib/export";
import type {
  AnalyzeParams,
  AnalyzeResponse,
  CloudOptions,
  Health,
  PaletteName,
  RotationMode,
} from "./types";

const prefersDark =
  typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;

const DEFAULT_PARAMS: AnalyzeParams = {
  column: "",
  method: "frequency",
  ngram: "1-2",
  maxWords: 200,
  minCount: 2,
  stopwords: "",
};

const DEFAULT_OPTIONS: CloudOptions = {
  palette: "vivid",
  rotation: "horizontal",
  spiral: "archimedean",
  fontScale: 0.14,
  fontWeight: 700,
  padding: 4,
  background: prefersDark ? "#111827" : "#ffffff",
  visibleCount: 120,
  seed: 1,
};

const BACKGROUNDS = [
  { value: "#ffffff", label: "سفید" },
  { value: "#fbf7ef", label: "کرم" },
  { value: "#111827", label: "تیره" },
  { value: "#0f172a", label: "سرمه‌ای" },
  { value: "transparent", label: "شفاف" },
];

const GHOST_WORDS = [
  ["هوش مصنوعی", 44, -10, -88],
  ["ابر کلمات", 34, -170, 92],
  ["زبان فارسی", 30, 175, 95],
  ["داده", 26, -235, -20],
  ["متن", 24, 240, -25],
  ["تحلیل", 20, 15, 122],
  ["موضوع", 18, -170, -105],
  ["کلمه", 16, 165, -110],
] as const;

type Status = { kind: "idle" } | { kind: "loading" } | { kind: "error"; message: string };

const fa = (n: number, digits = 0) => n.toLocaleString("fa-IR", { maximumFractionDigits: digits });

function Panel({ step, icon, title, children }: { step: number; icon: IconName; title: string; children: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-title">
        <span className="step">{fa(step)}</span>
        <h2>{title}</h2>
        <Icon name={icon} />
      </header>
      {children}
    </section>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthChecked, setHealthChecked] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [params, setParams] = useState<AnalyzeParams>(DEFAULT_PARAMS);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [dirty, setDirty] = useState(false);
  const [options, setOptions] = useState<CloudOptions>(DEFAULT_OPTIONS);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [placedCount, setPlacedCount] = useState(0);
  const [toast, setToast] = useState<string | null>(null);
  const cloudRef = useRef<WordCloudHandle>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setHealthChecked(true));
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 2500);
    return () => window.clearTimeout(t);
  }, [toast]);

  const run = useCallback(async (f: File, p: AnalyzeParams) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStatus({ kind: "loading" });
    try {
      const res = await analyze(f, p, ctrl.signal);
      setResult(res);
      setHidden(new Set());
      setDirty(false);
      setStatus({ kind: "idle" });
      if (res.column && res.column !== p.column) {
        setParams((prev) => ({ ...prev, column: res.column ?? "" }));
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setStatus({ kind: "error", message: (err as Error).message });
    }
  }, []);

  const onFile = (f: File) => {
    const next = { ...params, column: "" };
    setFile(f);
    setParams(next);
    setResult(null);
    run(f, next);
  };

  const onSample = async () => {
    const res = await fetch(`${import.meta.env.BASE_URL}sample.csv`);
    const blob = await res.blob();
    onFile(new File([blob], "sample.csv", { type: "text/csv" }));
  };

  // `?sample` in the URL loads the bundled sample file (handy for demos / screenshots)
  const autoSample = useRef(false);
  useEffect(() => {
    if (autoSample.current || !new URLSearchParams(window.location.search).has("sample")) return;
    autoSample.current = true;
    onSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updateParam = <K extends keyof AnalyzeParams>(key: K, value: AnalyzeParams[K]) => {
    setParams((p) => ({ ...p, [key]: value }));
    setDirty(true);
  };

  const updateOption = <K extends keyof CloudOptions>(key: K, value: CloudOptions[K]) =>
    setOptions((o) => ({ ...o, [key]: value }));

  const toggleWord = useCallback((text: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(text)) next.delete(text);
      else next.add(text);
      return next;
    });
  }, []);

  const words = useMemo(
    () => (result ? visibleWords(result.words, hidden, options.visibleCount) : []),
    [result, hidden, options.visibleCount],
  );

  const base = fileBaseName(file?.name);

  const savePng = async () => {
    const canvas = cloudRef.current?.toCanvas(3);
    if (!canvas) return;
    downloadBlob(await canvasToBlob(canvas), `${base}.png`);
  };
  const saveSvg = () => {
    const svg = cloudRef.current?.toSvg();
    if (svg) downloadBlob(new Blob([svg], { type: "image/svg+xml" }), `${base}.svg`);
  };
  const saveJson = () => {
    if (!result) return;
    const data = JSON.stringify(result.words.filter((w) => !hidden.has(w.text)), null, 2);
    downloadBlob(new Blob([data], { type: "application/json" }), `${base}.json`);
  };
  const copyImage = async () => {
    const canvas = cloudRef.current?.toCanvas(2);
    if (!canvas) return;
    try {
      await copyPng(canvas);
      setToast("تصویر در کلیپ‌بورد کپی شد");
    } catch {
      setToast("مرورگر اجازهٔ کپی تصویر را نداد");
    }
  };

  const loading = status.kind === "loading";
  const maxVisible = result ? Math.max(10, result.words.length) : 200;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">
            <Icon name="cloud" size={24} />
          </span>
          <div>
            <h1>ابر کلمات فارسی</h1>
            <p className="muted">فایل متنی را بفرستید، کلمات و عبارات پرتکرار را ببینید و تصویرش را ذخیره کنید</p>
          </div>
        </div>
        <span className={`pill ${health ? "pill--ok" : healthChecked ? "pill--off" : ""}`}>
          {health
            ? `سرور فعال · نسخهٔ ${health.version}${health.bertopic ? " · BERTopic" : ""}`
            : healthChecked
              ? "سرور در دسترس نیست"
              : "در حال اتصال…"}
        </span>
      </header>

      <main className="layout">
        <aside className="sidebar">
          <Panel step={1} icon="file" title="فایل">
            <FileDrop file={file} maxMb={health?.max_upload_mb ?? 20} onFile={onFile} onSample={onSample} />
          </Panel>

          <Panel step={2} icon="sliders" title="تحلیل">
            {result && result.columns.length > 0 && (
              <label className="field">
                <span>ستون متن</span>
                <select value={params.column} onChange={(e) => updateParam("column", e.target.value)}>
                  {result.columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="field">
              <span>روش وزن‌دهی</span>
              <select
                value={params.method}
                onChange={(e) => updateParam("method", e.target.value as AnalyzeParams["method"])}
              >
                <option value="frequency">فراوانی + PMI (سریع)</option>
                <option value="bertopic" disabled={health ? !health.bertopic : false}>
                  BERTopic + ParsBERT{health && !health.bertopic ? " (نصب نیست)" : ""}
                </option>
              </select>
            </label>
            <div className="field">
              <span>نوع عبارت</span>
              <div className="segmented">
                {(
                  [
                    ["1", "تک‌کلمه"],
                    ["2", "دوکلمه‌ای"],
                    ["1-2", "هر دو"],
                  ] as const
                ).map(([v, label]) => (
                  <button
                    key={v}
                    type="button"
                    className={params.ngram === v ? "active" : undefined}
                    onClick={() => updateParam("ngram", v)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="row">
              <label className="field">
                <span>حداکثر کلمات</span>
                <input
                  type="number"
                  min={10}
                  max={1000}
                  value={params.maxWords}
                  onChange={(e) => updateParam("maxWords", Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>حداقل تکرار</span>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={params.minCount}
                  onChange={(e) => updateParam("minCount", Number(e.target.value))}
                />
              </label>
            </div>
            <label className="field">
              <span>کلمات اضافه برای حذف</span>
              <textarea
                rows={2}
                placeholder="با ویرگول یا خط جدید جدا کنید"
                value={params.stopwords}
                onChange={(e) => updateParam("stopwords", e.target.value)}
              />
            </label>
            <button
              type="button"
              className="primary block"
              disabled={!file || loading}
              onClick={() => file && run(file, params)}
            >
              <Icon name="sparkle" />
              {loading ? "در حال تحلیل…" : dirty && result ? "تحلیل دوباره" : "تحلیل"}
            </button>
            {params.method === "bertopic" && (
              <p className="muted small">
                BERTopic در اولین اجرا مدل ParsBERT را دانلود می‌کند و ممکن است چند دقیقه طول بکشد.
              </p>
            )}
          </Panel>

          <Panel step={3} icon="palette" title="ظاهر">
            <div className="field">
              <span>رنگ‌بندی</span>
              <div className="swatches">
                {(Object.keys(PALETTES) as PaletteName[]).map((name) => (
                  <button
                    key={name}
                    type="button"
                    title={PALETTES[name].label}
                    aria-label={PALETTES[name].label}
                    aria-pressed={options.palette === name}
                    className={`swatch${options.palette === name ? " active" : ""}`}
                    onClick={() => updateOption("palette", name)}
                  >
                    {PALETTES[name].colors.slice(0, 5).map((c) => (
                      <span key={c} style={{ background: c }} />
                    ))}
                  </button>
                ))}
              </div>
            </div>
            <label className="field">
              <span>
                تعداد کلمات در ابر <b>{fa(Math.min(options.visibleCount, maxVisible))}</b>
              </span>
              <input
                type="range"
                min={10}
                max={maxVisible}
                value={Math.min(options.visibleCount, maxVisible)}
                onChange={(e) => updateOption("visibleCount", Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>
                اندازهٔ قلم <b>{fa(Math.round(options.fontScale * 100))}٪</b>
              </span>
              <input
                type="range"
                min={0.08}
                max={0.3}
                step={0.01}
                value={options.fontScale}
                onChange={(e) => updateOption("fontScale", Number(e.target.value))}
              />
            </label>
            <div className="row">
              <label className="field">
                <span>چرخش</span>
                <select
                  value={options.rotation}
                  onChange={(e) => updateOption("rotation", e.target.value as RotationMode)}
                >
                  {(Object.keys(ROTATIONS) as RotationMode[]).map((r) => (
                    <option key={r} value={r}>
                      {ROTATIONS[r]}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>چیدمان</span>
                <select
                  value={options.spiral}
                  onChange={(e) => updateOption("spiral", e.target.value as CloudOptions["spiral"])}
                >
                  <option value="archimedean">بیضی</option>
                  <option value="rectangular">مستطیلی</option>
                </select>
              </label>
            </div>
            <div className="row">
              <label className="field">
                <span>ضخامت قلم</span>
                <select
                  value={options.fontWeight}
                  onChange={(e) =>
                    updateOption("fontWeight", Number(e.target.value) as CloudOptions["fontWeight"])
                  }
                >
                  <option value={400}>معمولی</option>
                  <option value={700}>ضخیم</option>
                  <option value={900}>خیلی ضخیم</option>
                </select>
              </label>
              <label className="field">
                <span>پس‌زمینه</span>
                <select value={options.background} onChange={(e) => updateOption("background", e.target.value)}>
                  {BACKGROUNDS.map((b) => (
                    <option key={b.value} value={b.value}>
                      {b.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </Panel>
        </aside>

        <section className="main">
          <div className="panel canvas-panel">
            <div className="toolbar">
              <div className="stats">
                {result ? (
                  <>
                    <span className="chip">
                      <b>{fa(result.documents)}</b> سند
                    </span>
                    <span className="chip">
                      <b>{fa(result.tokens)}</b> توکن
                    </span>
                    {result.method === "bertopic" && (
                      <span className="chip">
                        <b>{fa(result.topics)}</b> موضوع
                      </span>
                    )}
                    <span className="chip">
                      <b>{fa(placedCount)}</b> از {fa(words.length)} کلمه
                    </span>
                  </>
                ) : (
                  <span className="muted">هنوز تحلیلی انجام نشده</span>
                )}
              </div>
              <div className="actions">
                <button
                  type="button"
                  className="icon-btn"
                  disabled={!result}
                  title="چیدمان جدید"
                  onClick={() => updateOption("seed", options.seed + 1)}
                >
                  <Icon name="shuffle" />
                  <span>چیدمان جدید</span>
                </button>
                <button type="button" className="icon-btn" disabled={!result} title="کپی تصویر" onClick={copyImage}>
                  <Icon name="copy" />
                  <span>کپی</span>
                </button>
                <button type="button" className="icon-btn" disabled={!result} title="دانلود SVG" onClick={saveSvg}>
                  <Icon name="code" />
                  <span>SVG</span>
                </button>
                <button type="button" className="icon-btn" disabled={!result} title="دانلود JSON" onClick={saveJson}>
                  <Icon name="braces" />
                  <span>JSON</span>
                </button>
                <button type="button" className="primary" disabled={!result} onClick={savePng}>
                  <Icon name="download" />
                  ذخیرهٔ PNG
                </button>
              </div>
            </div>

            {status.kind === "error" && <div className="alert alert--error">{status.message}</div>}
            {result?.warnings.map((w) => (
              <div key={w} className="alert" dir="ltr">
                {w}
              </div>
            ))}

            <div className={`stage-wrap${loading ? " is-loading" : ""}`}>
              {result ? (
                <WordCloud
                  ref={cloudRef}
                  words={words}
                  options={options}
                  onWordClick={toggleWord}
                  onPlaced={setPlacedCount}
                />
              ) : (
                <div className="empty">
                  <svg className="ghost" viewBox="-300 -140 600 280" aria-hidden>
                    {GHOST_WORDS.map(([t, s, x, y]) => (
                      <text key={t} x={x} y={y} fontSize={s} textAnchor="middle">
                        {t}
                      </text>
                    ))}
                  </svg>
                  <div className="empty-text">
                    <p>
                      یک فایل <b>CSV</b> با ستون متن (مثلاً <code>txtContent</code>) یا یک فایل <b>TXT</b> بفرستید
                    </p>
                    <button type="button" className="primary" onClick={onSample}>
                      <Icon name="sparkle" />
                      امتحان با فایل نمونه
                    </button>
                    <p className="muted small">در ابر کلمات روی هر کلمه کلیک کنید تا حذف شود</p>
                  </div>
                </div>
              )}
              {loading && (
                <div className="overlay">
                  <span className="spinner" />
                  در حال تحلیل متن…
                </div>
              )}
            </div>
          </div>

          {result && (
            <WordList
              words={result.words}
              hidden={hidden}
              onToggle={toggleWord}
              onReset={() => setHidden(new Set())}
            />
          )}
        </section>
      </main>

      <footer className="footer muted">
        BERTopic · ParsBERT · d3-cloud — نسخهٔ خط فرمان: <code>bert-wordcloud generate data.csv</code>
      </footer>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
