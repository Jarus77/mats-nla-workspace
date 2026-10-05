"""Scoring statistics for the prereg (§2 indices, §6.1 statistics, §4.4 FVE guard, H6 trend).

All functions are numpy-only (scipy only for the normal tail) and take plain arrays, so they run
locally on downloaded scores.

Conventions
- A "score" is a reader's paired log-odds LO = log p(a) - log p(b) for one activation (or the
  probe projection). Higher means "more like value a / the +Delta direction".
- ``clusters`` labels the unit of resampling (passage / item). The bootstrap resamples whole
  clusters with replacement (prereg §6.1: item-clustered bootstrap, 2,000 resamples).
- AUROC counts ties as 1/2 (Mann-Whitney U / (n_pos n_neg)).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ArrayLike = Sequence[float] | np.ndarray
DEFAULT_BOOT = 2000


# ----------------------------------------------------------------------------- AUROC


def _rank_avg(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (scipy.stats.rankdata 'average')."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    # boundaries of tie groups
    diff = np.concatenate(([True], xs[1:] != xs[:-1], [True]))
    idx = np.flatnonzero(diff)
    for a, b in zip(idx[:-1], idx[1:]):
        ranks[order[a:b]] = 0.5 * (a + b - 1) + 1.0
    return ranks


def auroc(pos: ArrayLike, neg: ArrayLike) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie). NaN if a class is empty."""
    p = np.asarray(pos, dtype=np.float64).ravel()
    n = np.asarray(neg, dtype=np.float64).ravel()
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    r = _rank_avg(np.concatenate([p, n]))
    u = r[: len(p)].sum() - len(p) * (len(p) + 1) / 2.0
    return float(u / (len(p) * len(n)))


def auroc_labels(scores: ArrayLike, labels: ArrayLike) -> float:
    """AUROC of ``scores`` for binary ``labels`` (1 = positive)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    return auroc(s[y], s[~y])


# ----------------------------------------------------------------------------- bootstrap


def _cluster_index(clusters: ArrayLike) -> list[np.ndarray]:
    c = np.asarray(clusters)
    _, inv = np.unique(c, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: ArrayLike,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    ci: float = 0.95,
    return_samples: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    """Percentile CI of ``stat(row_idx)`` under resampling whole clusters with replacement.

    ``stat`` receives an integer index array into the rows (duplicates allowed) and returns a
    float. NaN resamples (e.g. a class went missing) are dropped from the percentile.
    """
    groups = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    g = len(groups)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, g, g)
        boots[b] = stat(np.concatenate([groups[i] for i in pick]))
    finite = boots[np.isfinite(boots)]
    a = (1.0 - ci) / 2.0
    lo, hi = (float(np.quantile(finite, a)), float(np.quantile(finite, 1 - a))) if len(finite) else (
        float("nan"), float("nan"))
    return (lo, hi, boots) if return_samples else (lo, hi)


@dataclass(frozen=True)
class AurocResult:
    auroc: float
    lo: float
    hi: float
    n: int
    n_clusters: int


def auroc_ci(
    scores: ArrayLike,
    labels: ArrayLike,
    clusters: ArrayLike | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    ci: float = 0.95,
) -> AurocResult:
    """AUROC with a cluster-bootstrap percentile CI (clusters default to rows)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    c = np.arange(len(s)) if clusters is None else np.asarray(clusters)
    if not (len(s) == len(y) == len(c)):
        raise ValueError(f"length mismatch: scores {len(s)}, labels {len(y)}, clusters {len(c)}")
    point = auroc(s[y], s[~y])
    lo, hi = cluster_bootstrap(lambda i: auroc(s[i][y[i]], s[i][~y[i]]), c, n_boot, seed, ci)
    return AurocResult(point, lo, hi, len(s), len(np.unique(c)))


# ----------------------------------------------------------------------------- indices (§2)


def _pm_rows(
    plus: ArrayLike, minus: ArrayLike, clusters: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(plus, dtype=np.float64).ravel()
    m = np.asarray(minus, dtype=np.float64).ravel()
    if len(p) != len(m):
        raise ValueError(f"plus/minus must be paired by item: {len(p)} vs {len(m)}")
    c = np.arange(len(p)) if clusters is None else np.asarray(clusters).ravel()
    if len(c) != len(p):
        raise ValueError("clusters must have one entry per item")
    s = np.concatenate([p, m])
    y = np.concatenate([np.ones(len(p), bool), np.zeros(len(m), bool)])
    return p, m, s, y, np.concatenate([c, c])


def follow_feature(
    lo_plus: ArrayLike,
    lo_minus: ArrayLike,
    clusters: ArrayLike | None = None,
    lo_zero: ArrayLike | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """FollowFeature(R, f, dose): AUROC separating +Delta from -Delta steered versions of the
    same items, from reader R's paired log-odds (chance 0.5).

    ``lo_zero`` (optional, matched null (i)): the 0-dose LO per item, subtracted from both
    versions so the reader's item-level prior drops out of the across-item AUROC.
    Also returns ``paired_win`` = P(LO+ > LO-) within item (ties 1/2) and the mean shift.
    """
    p, m = np.asarray(lo_plus, float).ravel(), np.asarray(lo_minus, float).ravel()
    if lo_zero is not None:
        z = np.asarray(lo_zero, float).ravel()
        p, m = p - z, m - z
    p, m, s, y, c = _pm_rows(p, m, clusters)
    r = auroc_ci(s, y, c, n_boot, seed)
    return {
        "auroc": r.auroc, "lo": r.lo, "hi": r.hi, "n_items": len(p), "n_clusters": r.n_clusters,
        "paired_win": float(np.mean((p > m) + 0.5 * (p == m))),
        "mean_shift": float(np.mean(p - m)),
        "zero_subtracted": lo_zero is not None,
    }


def follow_text(
    lo_text_a: ArrayLike,
    lo_text_b: ArrayLike,
    clusters: ArrayLike | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """FollowText(R, f): AUROC with which R separates texts whose text-implied value is a from
    texts implied b, after both activations were clamped to the same probe projection.
    Inputs are paired (pair k = texts i_k, j_k); clusters default to the pair index."""
    p, m, s, y, c = _pm_rows(lo_text_a, lo_text_b, clusters)
    r = auroc_ci(s, y, c, n_boot, seed)
    return {"auroc": r.auroc, "lo": r.lo, "hi": r.hi, "n_pairs": len(p),
            "n_clusters": r.n_clusters, "paired_win": float(np.mean((p > m) + 0.5 * (p == m)))}


def fraction_of_probe(auroc_reader: float, auroc_probe: float) -> float:
    """(AUROC_R - 0.5) / (AUROC_probe - 0.5); NaN when the probe is at or below chance."""
    if not np.isfinite(auroc_probe) or auroc_probe <= 0.5:
        return float("nan")
    return float((auroc_reader - 0.5) / (auroc_probe - 0.5))


def fraction_of_probe_ci(
    reader_scores: ArrayLike,
    probe_scores: ArrayLike,
    labels: ArrayLike,
    clusters: ArrayLike | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, float]:
    """Fraction of probe-recoverable information on one contrast, with a joint cluster bootstrap
    (reader and probe resampled on the same rows)."""
    r = np.asarray(reader_scores, float)
    q = np.asarray(probe_scores, float)
    y = np.asarray(labels).astype(bool)
    c = np.arange(len(r)) if clusters is None else np.asarray(clusters)

    def stat(i: np.ndarray) -> float:
        return fraction_of_probe(auroc(r[i][y[i]], r[i][~y[i]]), auroc(q[i][y[i]], q[i][~y[i]]))

    a_r, a_q = auroc(r[y], r[~y]), auroc(q[y], q[~y])
    lo, hi = cluster_bootstrap(stat, c, n_boot, seed)
    return {"fraction": fraction_of_probe(a_r, a_q), "lo": lo, "hi": hi, "auroc_reader": a_r,
            "auroc_probe": a_q}


def paired_null_auroc(
    real: tuple[ArrayLike, ArrayLike],
    null: tuple[ArrayLike, ArrayLike],
    clusters: ArrayLike | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, float]:
    """FollowFeature on the real contrast vs a matched null contrast on the SAME items (e.g.
    equal-norm random-direction steering, §6.1 null (ii)). Returns both AUROCs, their
    difference with a cluster-paired bootstrap CI, and a one-sided bootstrap p (diff <= 0)."""
    rp, rm, rs, ry, rc = _pm_rows(*real, clusters)
    np_, nm, ns, ny, _ = _pm_rows(*null, clusters)
    if len(rp) != len(np_):
        raise ValueError("real and null contrasts must cover the same items")
    a_real, a_null = auroc(rs[ry], rs[~ry]), auroc(ns[ny], ns[~ny])

    def diff(i: np.ndarray) -> float:
        return auroc(rs[i][ry[i]], rs[i][~ry[i]]) - auroc(ns[i][ny[i]], ns[i][~ny[i]])

    lo, hi, boots = cluster_bootstrap(diff, rc, n_boot, seed, return_samples=True)
    boots = boots[np.isfinite(boots)]
    p = float((np.sum(boots <= 0) + 1) / (len(boots) + 1)) if a_real != a_null else 1.0
    return {"auroc_real": a_real, "auroc_null": a_null, "diff": a_real - a_null, "diff_lo": lo,
            "diff_hi": hi, "p_one_sided": p}


# ----------------------------------------------------------------------------- trend (H6)


def page_trend_test(data: ArrayLike) -> dict[str, float]:
    """Page's L test for an increasing trend across ordered treatments.

    ``data`` is [n_blocks, k_treatments] with columns in the hypothesised increasing order
    (e.g. rows = items, columns = doses q90 < q99 < 2 q99; values = log-odds shift). Ranks are
    taken within each block (ties averaged). Large-sample normal approximation, one-sided
    (same as ``scipy.stats.page_trend_test(method="asymptotic")``).
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("data must be [n_blocks, k>=2 treatments]")
    n, k = x.shape
    ranks = np.vstack([_rank_avg(row) for row in x])
    big_l = float(np.sum(np.arange(1, k + 1) * ranks.sum(axis=0)))
    mean = n * k * (k + 1) ** 2 / 4.0
    var = n * k**2 * (k + 1) * (k**2 - 1) / 144.0
    z = (big_l - mean) / math.sqrt(var)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {"L": big_l, "z": z, "p": p, "n_blocks": n, "k": k}


# ----------------------------------------------------------------------------- FVE guard (§4.4)


def fve_guard(
    fve_natural: ArrayLike,
    fve_steered: ArrayLike,
    margin: float = 0.05,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """One-sided FVE guard, paired per example (each entry = mean FVE over k samples).

    Void if mean FVE(steered) < mean FVE(natural) - margin, or < the 5th percentile of natural
    per-example FVE. High FVE proves nothing (prereg rev 2, row 13). Pairs with a NaN on either
    side (all samples failed to extract) are dropped and counted.
    """
    a = np.asarray(fve_natural, float).ravel()
    b = np.asarray(fve_steered, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = b - a
    lo, hi = cluster_bootstrap(lambda i: float(d[i].mean()), np.arange(len(d)), n_boot, seed)
    p5 = float(np.quantile(a, 0.05)) if len(a) else float("nan")
    by_margin = bool(b.mean() < a.mean() - margin) if len(a) else True
    by_p5 = bool(b.mean() < p5) if len(a) else True
    return {"mean_natural": float(a.mean()) if len(a) else float("nan"),
            "mean_steered": float(b.mean()) if len(b) else float("nan"),
            "mean_diff": float(d.mean()) if len(d) else float("nan"), "lo": lo, "hi": hi,
            "natural_p5": p5, "void_margin": by_margin, "void_p5": by_p5,
            "void": by_margin or by_p5, "n_pairs": int(ok.sum()), "n_dropped": int((~ok).sum())}
