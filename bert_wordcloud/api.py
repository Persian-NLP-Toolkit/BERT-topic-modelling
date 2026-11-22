"""FastAPI backend used by the React frontend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .ngrams import parse_ngram_range
from .pipeline import bertopic_available, compute_frequencies, load_documents
from .preprocess import parse_stopwords

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "20"))
STATIC_DIR = Path(os.environ.get("STATIC_DIR", Path(__file__).parent / "static"))

app = FastAPI(title="BERT Word Cloud", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "bertopic": bertopic_available(),
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/wordcloud")
async def wordcloud(
    file: UploadFile = File(...),
    column: Optional[str] = Form(None),
    method: Literal["frequency", "bertopic"] = Form("frequency"),
    ngram: str = Form("1-2"),
    max_words: int = Form(200, ge=10, le=1000),
    min_count: int = Form(2, ge=1, le=100),
    stopwords: Optional[str] = Form(None),
) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
    try:
        ngram_range = parse_ngram_range(ngram)
        docs = load_documents(data, filename=file.filename, column=column or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not docs.texts:
        raise HTTPException(400, "The file does not contain any text.")

    result = await run_in_threadpool(
        compute_frequencies,
        docs.texts,
        method=method,
        ngram_range=ngram_range,
        max_words=max_words,
        min_count=min_count,
        extra_stopwords=parse_stopwords(stopwords),
    )
    payload = result.to_dict()
    payload.update({"columns": docs.columns, "column": docs.column})
    return payload


# Serve the built React app (if present) so a single container can host everything.
if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
