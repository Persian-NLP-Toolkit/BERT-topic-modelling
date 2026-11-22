"""Lightweight Persian text normalisation and tokenisation (no heavy NLP deps)."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

from .stopwords import STOPWORDS

ZWNJ = "\u200c"

_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "\u200f": " ",  # RLM
        "\u200e": " ",  # LRM
        "\u00a0": " ",  # NBSP
    }
)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")  # harakat + tatweel
# Anything that is not a Persian/Arabic letter, a Latin letter or ZWNJ becomes a space.
_NON_LETTER_RE = re.compile(
    r"[^\u0621-\u063A\u0641-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CCa-z\u200C]+"
)
_MULTI_ZWNJ_RE = re.compile(r"\u200C{2,}")


def normalize(text: str) -> str:
    text = str(text).translate(_CHAR_MAP).lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _DIACRITICS_RE.sub("", text)
    text = _NON_LETTER_RE.sub(" ", text)
    text = _MULTI_ZWNJ_RE.sub(ZWNJ, text)
    return text


def tokenize(
    text: str,
    extra_stopwords: Optional[Iterable[str]] = None,
    min_length: int = 2,
) -> List[str]:
    stop: Set[str] = STOPWORDS
    if extra_stopwords:
        stop = STOPWORDS | {normalize(w).strip().strip(ZWNJ) for w in extra_stopwords}
    tokens = []
    for tok in normalize(text).split():
        tok = tok.strip(ZWNJ)
        if len(tok) < min_length or tok in stop:
            continue
        tokens.append(tok)
    return tokens


def parse_stopwords(raw: Optional[str]) -> List[str]:
    """Split a user supplied stopword string (comma / newline / Persian comma separated)."""
    if not raw:
        return []
    return [w.strip() for w in re.split(r"[,\n،;]+", raw) if w.strip()]
