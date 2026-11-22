"""N-gram extraction and weighting (counts, PMI and optional per-topic factors)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

NgramRange = Tuple[int, int]


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    return zip(*(tokens[i:] for i in range(n)))


def parse_ngram_range(value: str) -> NgramRange:
    """Parse "1", "2" or "1-2" into a (min_n, max_n) tuple."""
    parts = str(value).replace(",", "-").split("-")
    try:
        lo, hi = int(parts[0]), int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid n-gram range: {value!r}") from exc
    if lo < 1 or hi < lo or hi > 3:
        raise ValueError(f"n-gram range must be within 1..3, got {value!r}")
    return lo, hi


def weight_ngrams(
    token_docs: List[List[str]],
    ngram_range: NgramRange = (1, 2),
    min_count: int = 2,
    min_pmi: float = 1.0,
    groups: Iterable[Tuple[List[int], float]] | None = None,
) -> Dict[str, float]:
    """Return ``{label: weight}``.

    * unigrams are weighted by their count,
    * longer n-grams by ``count * PMI`` (and dropped when PMI < ``min_pmi``).

    ``groups`` is an optional list of ``(doc_indices, factor)`` pairs (e.g. one
    per BERTopic topic); every group is counted separately and its weights are
    multiplied by ``factor`` before being summed. Without groups all documents
    form a single group with factor 1.
    """
    uni = Counter()
    for toks in token_docs:
        uni.update(toks)
    total = sum(uni.values())
    if total == 0:
        return {}

    if groups is None:
        groups = [(list(range(len(token_docs))), 1.0)]

    lo, hi = ngram_range
    weights: Dict[str, float] = {}
    for doc_ids, factor in groups:
        tf: Counter = Counter()
        for i in doc_ids:
            toks = token_docs[i]
            for n in range(lo, hi + 1):
                tf.update(iter_ngrams(toks, n))

        for gram, cnt in tf.items():
            if cnt < min_count:
                continue
            if len(gram) == 1:
                score = float(cnt)
            else:
                p_joint = cnt / total
                p_ind = math.prod(uni[w] / total for w in gram)
                pmi = math.log2(p_joint / p_ind) if p_ind > 0 else 0.0
                if pmi < min_pmi:
                    continue
                score = cnt * pmi
            label = " ".join(gram)
            weights[label] = weights.get(label, 0.0) + score * factor
    return weights


def top_n(weights: Dict[str, float], n: int) -> List[Tuple[str, float]]:
    return sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
