"""Pure helpers for experiment 11 (mentor follow-ups): consensus over k samples, punctuation context of
a read token, per-stratum paired statistics, layer x position map assembly, and the shared-contradiction
catalogue. No torch, no network; everything here is unit-tested in test_a11_lib.py.

PILOT / EXPLORATORY: every number produced through these helpers is reported as such.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from mats import metrics as M

SENTENCE_END = ".!?"
OTHER_PUNCT = ",;:)\"'’”-–—"
_SENT_BOUNDARY = re.compile(r"[.!?]['\")’”]*(\s|$)")


# ----------------------------------------------------------------------------- consensus (Agam point 2)
def consensus_rate(rows: Sequence[Sequence[str | None]], hit: Callable[[str], bool], min_count: int) -> dict[str, Any]:
    """Multi-sample consensus for a yes/no property of free readouts.

    rows[i] = the k samples of item i (None = failed extraction, ignored).
    Returns single-sample rate, item-level any-rate, and the consensus rate (items whose support
    count, the number of samples with the property, is >= min_count). Kenney's rule is min_count=2 of 8.
    """
    support = [sum(bool(hit(t)) for t in row if t) for row in rows]
    n_samples = sum(sum(1 for t in row if t) for row in rows)
    k = max((len(row) for row in rows), default=0)
    return {
        "n_items": len(rows),
        "k": k,
        "single": (sum(support) / n_samples) if n_samples else float("nan"),
        "items_any": float(np.mean([s > 0 for s in support])) if rows else float("nan"),
        "consensus": float(np.mean([s >= min_count for s in support])) if rows else float("nan"),
        "min_count": min_count,
        "support": support,
    }


def self_consistency(rows: Sequence[Sequence[str | None]], key: Callable[[str], Hashable]) -> dict[str, Any]:
    """Per item: share of samples that agree with the item's majority value of ``key``.
    (Kenney's self-consistency statistic, generalised from yes/no to any hashable key.)"""
    per_item = []
    for row in rows:
        vals = [key(t) for t in row if t]
        if not vals:
            per_item.append(float("nan"))
            continue
        counts: dict[Hashable, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        per_item.append(max(counts.values()) / len(vals))
    return {"per_item": per_item, "mean": float(np.nanmean(per_item)) if per_item else float("nan")}


# ----------------------------------------------------------------------------- punctuation (Agam point 3)
def punct_context(text: str, window: int = 12) -> dict[str, Any]:
    """Classify the read token's context from the text that ENDS at the read token.

    ends:           "stop"  if the text ends with sentence-final punctuation (. ! ?),
                    "punct" if it ends with other punctuation, else "word".
    sentence_start: a sentence boundary occurs within the last ``window`` characters
                    (the read token sits at or just after the start of a sentence).
    """
    t = text.rstrip()
    last = t[-1] if t else ""
    if last in SENTENCE_END:
        ends = "stop"
    elif last in OTHER_PUNCT:
        ends = "punct"
    else:
        ends = "word"
    tail = t[-window:] if window > 0 else ""
    sentence_start = bool(_SENT_BOUNDARY.search(tail))
    return {"ends": ends, "sentence_start": sentence_start, "tail": tail}


# ----------------------------------------------------------------------------- strata
def strata_stats(lo_plus: Sequence[float], lo_minus: Sequence[float], strata: Sequence[str]) -> dict[str, dict[str, float]]:
    """Paired +/- statistics (mean shift, paired win, AUROC) within each stratum label."""
    p = np.asarray(lo_plus, float)
    m = np.asarray(lo_minus, float)
    s = np.asarray(strata)
    out: dict[str, dict[str, float]] = {}
    for label in sorted(set(s.tolist())):
        ix = np.flatnonzero(s == label)
        st = M.paired_stats(p[ix], m[ix])
        out[label] = {"n": int(len(ix)), **st, "shift_sd": float(np.std(p[ix] - m[ix], ddof=1)) if len(ix) > 1 else float("nan")}
    return out


# ----------------------------------------------------------------------------- map (Agam point 5)
def assemble_map(cells: Sequence[dict[str, Any]]) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    """{feature: {(layer, pos): cell}} from a flat list of cells (each with layer, pos, feature)."""
    out: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for c in cells:
        out.setdefault(c["feature"], {})[(int(c["layer"]), int(c["pos"]))] = dict(c)
    return out


def rank_corr(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Spearman rank correlation with NaN pairs dropped."""
    a = np.asarray(x, float)
    b = np.asarray(y, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    r = spearmanr(a[ok], b[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


# ----------------------------------------------------------------------------- catalogue (Agam point 6)
def shared_contradictions(rows: Sequence[dict[str, Any]], readers: Sequence[str],
                          label: str = "contradicted_by_text") -> list[dict[str, Any]]:
    """Items on which EVERY reader in ``readers`` has at least one claim with ``label``.
    Input rows follow experiments/06 data/claims_gemini.jsonl: {reader, family, id, claims:[{claim,label}]}."""
    by_item: dict[tuple[str, str], dict[str, list[str]]] = {}
    for r in rows:
        key = (r["family"], r["id"])
        bad = [c["claim"] for c in r.get("claims", []) if c.get("label") == label]
        by_item.setdefault(key, {})[r["reader"]] = bad
    out = []
    for (fam, iid), per_reader in by_item.items():
        if all(per_reader.get(rd) for rd in readers):
            out.append({"family": fam, "id": iid, **{rd: per_reader[rd] for rd in readers}})
    return out
