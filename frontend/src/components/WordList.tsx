import { useMemo, useState } from "react";
import type { Word } from "../types";

interface Props {
  words: Word[];
  hidden: Set<string>;
  onToggle: (text: string) => void;
  onReset: () => void;
}

export function WordList({ words, hidden, onToggle, onReset }: Props) {
  const [query, setQuery] = useState("");
  const max = words[0]?.weight ?? 1;
  const shown = useMemo(
    () => (query ? words.filter((w) => w.text.includes(query.trim())) : words),
    [words, query],
  );

  return (
    <section className="panel wordlist">
      <header className="panel-head">
        <h2>کلمات ({words.length.toLocaleString("fa-IR")})</h2>
        {hidden.size > 0 && (
          <button type="button" className="link" onClick={onReset}>
            نمایش {hidden.size.toLocaleString("fa-IR")} کلمهٔ حذف‌شده
          </button>
        )}
      </header>
      <input
        className="search"
        type="search"
        placeholder="جستجو…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <ul>
        {shown.map((w) => {
          const off = hidden.has(w.text);
          return (
            <li key={w.text} className={off ? "off" : undefined}>
              <label>
                <input type="checkbox" checked={!off} onChange={() => onToggle(w.text)} />
                <span className="word">{w.text}</span>
                <span className="bar" aria-hidden>
                  <span style={{ width: `${Math.max(3, (w.weight / max) * 100)}%` }} />
                </span>
                <span className="num">
                  {w.weight.toLocaleString("fa-IR", { maximumFractionDigits: 1 })}
                </span>
              </label>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
