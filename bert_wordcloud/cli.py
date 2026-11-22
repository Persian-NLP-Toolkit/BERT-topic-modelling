"""Command line interface: ``bert-wordcloud generate|serve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .ngrams import parse_ngram_range
from .pipeline import METHODS, compute_frequencies, load_documents
from .preprocess import parse_stopwords


def _cmd_generate(args: argparse.Namespace) -> int:
    stop: List[str] = []
    if args.stopwords:
        stop = parse_stopwords(Path(args.stopwords).read_text(encoding="utf-8"))

    docs = load_documents(args.input, column=args.column)
    if docs.column:
        print(
            f"📄 {len(docs.texts)} documents from column '{docs.column}'",
            file=sys.stderr,
        )
    else:
        print(f"📄 {len(docs.texts)} documents", file=sys.stderr)

    result = compute_frequencies(
        docs.texts,
        method=args.method,
        ngram_range=parse_ngram_range(args.ngram),
        max_words=args.max_words,
        min_count=args.min_count,
        extra_stopwords=stop,
    )
    for w in result.warnings:
        print(f"⚠️  {w}", file=sys.stderr)
    if not result.words:
        print("❌ No terms found – nothing to draw.", file=sys.stderr)
        return 1

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ Saved {args.json}", file=sys.stderr)

    if args.output != "-":
        from .render import render_png

        out = render_png(
            result.frequencies,
            args.output,
            width=args.width,
            height=args.height,
            background=args.background,
            colormap=args.colormap,
            font_path=args.font,
            transparent=args.transparent,
        )
        print(f"✅ Saved {out}", file=sys.stderr)

    if args.top:
        for text, weight in result.words[: args.top]:
            print(f"{weight:10.2f}  {text}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "bert_wordcloud.api:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bert-wordcloud", description="Persian word clouds from text files."
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate a word cloud image from a file")
    g.add_argument(
        "input", help="CSV/TSV (one row per document) or TXT (one line per document)"
    )
    g.add_argument(
        "-o",
        "--output",
        default="wordclouds/wordcloud.png",
        help="PNG output path, or '-' to skip the image (default: %(default)s)",
    )
    g.add_argument(
        "-c", "--column", help="Text column (default: txtContent or auto-detect)"
    )
    g.add_argument(
        "-m",
        "--method",
        choices=METHODS,
        default="frequency",
        help="frequency (fast) or bertopic (topic weighted, needs extras)",
    )
    g.add_argument(
        "-n",
        "--ngram",
        default="1-2",
        help="n-gram range: 1, 2 or 1-2 (default: %(default)s)",
    )
    g.add_argument("--max-words", type=int, default=200)
    g.add_argument("--min-count", type=int, default=2)
    g.add_argument(
        "--stopwords",
        help="File with extra stopwords (one per line or comma separated)",
    )
    g.add_argument("--json", help="Also write the weighted terms to this JSON file")
    g.add_argument("--top", type=int, default=0, help="Print the top N terms to stdout")
    g.add_argument("--width", type=int, default=1600)
    g.add_argument("--height", type=int, default=900)
    g.add_argument("--background", default="white")
    g.add_argument("--colormap", default="tab20", help="Any matplotlib colormap name")
    g.add_argument("--font", help="Path to a TTF font (default: bundled Gandom)")
    g.add_argument("--transparent", action="store_true", help="Transparent background")
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("serve", help="Run the web API (and the built frontend)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
