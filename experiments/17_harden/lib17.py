"""Pure helpers for exp 17 (no torch model code): candidate construction, multi-candidate metrics."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

from mats.metrics import auroc, cluster_bootstrap
from spec17 import VALID_COLS


def candidates(template: dict) -> list:
    """The C filled strings in slot order; J-lens entries (template None) return their word lists."""
    if template["template"] is None:
        return list(template["slots"])
    return [template["template"].replace("{v}", s) for s in template["slots"]]


def true_index(row: str, item: dict) -> int:
    if row == "tb_tablecol":
        col = int(item["meta"]["col"])
        if col not in VALID_COLS:
            raise ValueError(f"table-column item outside the declared bins: col={col} ({item.get('id')})")
        return col - 1
    if row == "tb_paren":
        return {"inside": 0, "outside": 1}[item["value"]]
    raise KeyError(row)


def paren_item_ok(text: str) -> bool:
    """False when a quotation or a square bracket is open at the read token (the distractors would be true)."""
    return text.count('"') % 2 == 0 and text.count("[") <= text.count("]")


def lse_pair_lo(lp: np.ndarray, class_sets: dict[str, list[int]], values: Sequence[str]) -> np.ndarray:
    """[N] log-sum-exp(values[1] candidates) - log-sum-exp(values[0] candidates)."""
    lp = np.asarray(lp, float)
    return logsumexp(lp[:, class_sets[values[1]]], axis=1) - logsumexp(lp[:, class_sets[values[0]]], axis=1)


def _ci(stat, clusters: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:  # noqa: ANN001
    point = float(stat(np.arange(len(clusters))))
    lo, hi = cluster_bootstrap(stat, clusters, n_boot=n_boot, seed=seed)
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def metrics_multi(lp: np.ndarray, truth: np.ndarray, y: np.ndarray, class_sets: dict[str, list[int]],
                  values: Sequence[str], clusters: np.ndarray, n_boot: int, seed: int) -> dict:
    """Multi-candidate metrics for ONE template: lp is [N, C] summed log-probs."""
    lp = np.asarray(lp, float)
    if lp.ndim != 2:
        raise ValueError(f"lp must be [N, C] for one template, got shape {lp.shape}")
    truth, y, clusters = np.asarray(truth), np.asarray(y).astype(bool), np.asarray(clusters)
    n, c = lp.shape
    hit = (lp.argmax(1) == truth).astype(float)
    rank = (lp > lp[np.arange(n), truth][:, None]).sum(1)
    rr = 1.0 / (1.0 + rank)
    contrast = lse_pair_lo(lp, class_sets, values)
    top1 = _ci(lambda i: float(hit[i].mean()), clusters, n_boot, seed)
    top1["chance"] = 1.0 / c
    out = {
        "top1": top1,
        "mrr": _ci(lambda i: float(rr[i].mean()), clusters, n_boot, seed),
        "auroc_lse": _ci(lambda i: auroc(contrast[i][y[i]], contrast[i][~y[i]]), clusters, n_boot, seed),
        "spearman_argmax": None,
        "n": int(n),
    }
    am = lp.argmax(1)
    if len(np.unique(truth)) >= 3 and len(np.unique(am)) >= 2:
        out["spearman_argmax"] = float(spearmanr(am, truth).statistic)
    return out
