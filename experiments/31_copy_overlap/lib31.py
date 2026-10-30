"""Exp 31 helpers (pure, tested): token normalisation, copy overlap of a readout with a context window (content tokens
and word n-grams), the shuffled-window baseline (derangement), opposite-class partners for the leak-balanced
deletion, token deletion, rank AUROC, and a paired cluster bootstrap. No tokenizer or model is imported here; the
analysis script passes a `norm_of(token_id) -> str` mapping built from the Qwen tokenizer."""
from __future__ import annotations

import random
import re
from typing import Callable, Sequence

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)
_TAGS = re.compile(r"</?explanation>")


def strip_tags(text: str | None) -> str:
    """Remove the NLA's <explanation> tags (constant across items) for the overlap measures."""
    return _TAGS.sub("", text or "")


def norm_piece(decoded: str) -> str:
    """Normalised form of one decoded token: stripped and lowercased; pure-whitespace tokens keep their raw form so
    a newline token still matches a newline token."""
    s = decoded.strip().lower()
    return s if s else decoded


def is_content(norm: str) -> bool:
    """A content token has at least one letter or digit (punctuation, whitespace and '_' runs are not content)."""
    return bool(_ALNUM.search(norm))


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def ngrams(ws: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def token_overlap(readout_norms: Sequence[str], window_norms: set[str]) -> float | None:
    """Share of the readout's CONTENT tokens whose normalised form occurs in the window; None if no content token."""
    content = [t for t in readout_norms if is_content(t)]
    if not content:
        return None
    return sum(t in window_norms for t in content) / len(content)


def ngram_overlap(readout_words: Sequence[str], window_words: Sequence[str], n: int) -> float | None:
    """Share of the readout's word n-grams (with repetition) that occur among the window's word n-grams."""
    r = ngrams(readout_words, n)
    if not r:
        return None
    w = set(ngrams(window_words, n))
    return sum(g in w for g in r) / len(r)


def derangement(n: int, seed: int) -> list[int]:
    """A random permutation of range(n) with no fixed point (n >= 2), deterministic for `seed`."""
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(i != j for i, j in enumerate(p)):
            return p


def opposite_partners(labels: Sequence[int], seed: int) -> list[int]:
    """For each index, a random index whose label differs (both classes must be present)."""
    labels = list(labels)
    pos = [i for i, v in enumerate(labels) if v]
    neg = [i for i, v in enumerate(labels) if not v]
    if not pos or not neg:
        raise ValueError("both classes are needed")
    rng = random.Random(seed)
    return [rng.choice(neg) if v else rng.choice(pos) for v in labels]


def delete_tokens(ids: Sequence[int], norm_of: Callable[[int], str], banned: set[str]) -> list[int]:
    """Drop every token whose normalised form is in `banned` (punctuation included)."""
    return [t for t in ids if norm_of(t) not in banned]


def auroc(scores: Sequence[float], y: Sequence[int]) -> float:
    """Rank (Mann-Whitney) AUROC with ties counted half; nan if a class is missing."""
    from scipy.stats import rankdata

    s = np.asarray(scores, float)
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_cluster_bootstrap(stats: dict[str, Callable[[np.ndarray], float]], groups: Sequence, n_boot: int = 2000,
                             seed: int = 0) -> dict[str, tuple[float, float]]:
    """Percentile 95% CIs for several statistics of the SAME resampled index sets (resampling whole clusters), so
    differences between conditions computed inside one stat get paired intervals. Each stat takes an index array."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in stats}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        for k, f in stats.items():
            vals[k].append(f(idx))
    out = {}
    for k, v in vals.items():
        a = np.asarray(v, float)
        a = a[~np.isnan(a)]
        out[k] = (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2
    return out


def low_half_mask(values: Sequence[float | None]) -> np.ndarray:
    """True for items at or below the median of the non-None values (None counts as 0 overlap: nothing copied)."""
    v = np.array([0.0 if x is None else float(x) for x in values])
    return v <= np.median(v)
