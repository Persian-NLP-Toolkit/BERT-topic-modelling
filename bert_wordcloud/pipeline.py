"""Load documents and turn them into word cloud weights."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from .ngrams import NgramRange, top_n, weight_ngrams
from .preprocess import tokenize

log = logging.getLogger(__name__)

DEFAULT_COLUMN = "txtContent"
METHODS = ("frequency", "bertopic")
SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt")


@dataclass
class Documents:
    texts: List[str]
    columns: List[str] = field(default_factory=list)
    column: Optional[str] = None


@dataclass
class WordCloudResult:
    words: List[Tuple[str, float]]
    method: str
    documents: int
    tokens: int
    topics: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def frequencies(self) -> Dict[str, float]:
        return dict(self.words)

    def to_dict(self) -> dict:
        return {
            "words": [{"text": t, "weight": round(w, 4)} for t, w in self.words],
            "method": self.method,
            "documents": self.documents,
            "tokens": self.tokens,
            "topics": self.topics,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _pick_column(df: pd.DataFrame, column: Optional[str]) -> str:
    if column:
        if column not in df.columns:
            raise ValueError(
                f"Column {column!r} not found. Available: {', '.join(map(str, df.columns))}"
            )
        return column
    if DEFAULT_COLUMN in df.columns:
        return DEFAULT_COLUMN
    # every column is read as text (dtype=str), so pick among all of them
    text_cols = list(df.columns)
    if not text_cols:
        raise ValueError("No text column found in the file.")

    def avg_len(c) -> float:
        lengths = df[c].dropna().astype(str).str.len()
        return float(lengths.mean()) if len(lengths) else 0.0

    # the column with the longest average text is most likely the content
    return max(text_cols, key=avg_len)


def load_documents(
    source: Union[str, Path, bytes],
    filename: Optional[str] = None,
    column: Optional[str] = None,
) -> Documents:
    """Read documents from a CSV/TSV (one row per document) or TXT (one line per document)."""
    if isinstance(source, (str, Path)):
        filename = filename or str(source)
        data = Path(source).read_bytes()
    else:
        data = source
    name = (filename or "").lower()
    ext = Path(name).suffix or ".csv"
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {ext!r}; use one of {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".txt":
        text = data.decode("utf-8-sig", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return Documents(texts=lines)

    sep = "\t" if ext == ".tsv" else ","
    try:
        df = pd.read_csv(io.BytesIO(data), sep=sep, encoding="utf-8-sig", dtype=str)
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not parse file: {exc}") from exc
    col = _pick_column(df, column)
    texts = df[col].dropna().astype(str).tolist()
    return Documents(texts=texts, columns=[str(c) for c in df.columns], column=col)


# --------------------------------------------------------------------------- #
# BERTopic
# --------------------------------------------------------------------------- #
def bertopic_available() -> bool:
    try:
        import bertopic  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import umap  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _get_embedder():
    from .embedder import ParsBERTEmbedder

    return ParsBERTEmbedder()


def _topic_groups(texts: List[str]) -> List[Tuple[List[int], float]]:
    """Fit BERTopic and return ``(doc_indices, factor)`` per topic.

    Larger topics (lower id) get a higher factor, matching the original weighting.
    """
    from bertopic import BERTopic
    from umap import UMAP

    embedder = _get_embedder()
    embs = embedder.embed(texts)
    umap_model = UMAP(
        n_neighbors=max(2, min(15, len(texts) - 1)),
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    topic_model = BERTopic(
        language="multilingual",
        embedding_model=embedder,
        umap_model=umap_model,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(texts, embs)
    topic_ids = sorted({t for t in topics if t != -1})
    if not topic_ids:
        return []
    max_tid = max(topic_ids)
    return [
        ([i for i, t in enumerate(topics) if t == tid], float(max_tid - tid + 1))
        for tid in topic_ids
    ]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_frequencies(
    texts: Iterable[str],
    method: str = "frequency",
    ngram_range: NgramRange = (1, 2),
    max_words: int = 200,
    min_count: int = 2,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> WordCloudResult:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    stop = list(extra_stopwords or [])
    raw = [t for t in texts if isinstance(t, str) and t.strip()]
    token_docs = [tokenize(t, stop) for t in raw]
    keep = [i for i, toks in enumerate(token_docs) if toks]
    token_docs = [token_docs[i] for i in keep]
    warnings: List[str] = []

    groups = None
    used = "frequency"
    n_topics = 0
    if method == "bertopic":
        if not bertopic_available():
            warnings.append(
                "BERTopic dependencies are not installed; fell back to frequency mode "
                "(pip install 'bert-wordcloud[bertopic]')."
            )
        elif len(token_docs) < 10:
            warnings.append(
                "Too few documents for topic modelling (need at least 10); "
                "fell back to frequency mode."
            )
        else:
            try:
                groups = _topic_groups([" ".join(t) for t in token_docs])
                if groups:
                    used, n_topics = "bertopic", len(groups)
                else:
                    groups = None
                    warnings.append(
                        "BERTopic found no topics (all documents were outliers); "
                        "fell back to frequency mode."
                    )
            except Exception as exc:  # pragma: no cover - depends on heavy deps
                log.exception("BERTopic failed")
                groups = None
                warnings.append(
                    f"BERTopic failed ({exc}); fell back to frequency mode."
                )

    weights = weight_ngrams(token_docs, ngram_range, min_count=min_count, groups=groups)
    if not weights and min_count > 1:
        weights = weight_ngrams(token_docs, ngram_range, min_count=1, groups=groups)
        if weights:
            warnings.append(
                f"No terms appeared at least {min_count} times; showing all terms."
            )

    return WordCloudResult(
        words=top_n(weights, max_words),
        method=used,
        documents=len(token_docs),
        tokens=sum(len(t) for t in token_docs),
        topics=n_topics,
        warnings=warnings,
    )
