import { useRef, useState, type DragEvent } from "react";
import { Icon } from "./Icon";

const ACCEPT = ".csv,.tsv,.txt";

interface Props {
  file: File | null;
  maxMb: number;
  onFile: (file: File) => void;
  onSample: () => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} بایت`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} کیلوبایت`;
  return `${(bytes / 1024 / 1024).toFixed(1)} مگابایت`;
}

export function FileDrop({ file, maxMb, onFile, onSample }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  return (
    <div>
      <button
        type="button"
        className={`drop${over ? " drop--over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <span className="drop-icon">
          <Icon name="upload" size={22} />
        </span>
        {file ? (
          <>
            <strong className="drop-name" dir="auto">{file.name}</strong>
            <span className="muted">{formatSize(file.size)} · برای تغییر کلیک کنید</span>
          </>
        ) : (
          <>
            <strong>فایل را اینجا رها کنید</strong>
            <span className="muted">یا کلیک کنید · CSV، TSV یا TXT تا {maxMb} مگابایت</span>
          </>
        )}
      </button>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
      <button type="button" className="link" onClick={onSample}>
        امتحان با فایل نمونه
      </button>
    </div>
  );
}
