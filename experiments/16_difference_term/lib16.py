"""Pure helpers for exp 16: partner pairing, decomposition, swap construction, follow-AUROC."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from spec16 import VARIANTS


def partners(y: np.ndarray, clusters: Sequence[str], seed: int) -> np.ndarray:
    """j(i): a random item of the other class from a different cluster. Raises if impossible."""
    y = np.asarray(y)
    cl = np.asarray(clusters)
    rng = np.random.default_rng(seed)
    out = np.empty(len(y), int)
    for i in range(len(y)):
        cand = np.flatnonzero((y != y[i]) & (cl != cl[i]))
        if len(cand) == 0:
            raise ValueError(f"no partner for item {i}")
        out[i] = rng.choice(cand)
    return out


def decompose(h: torch.Tensor, r_ol: torch.Tensor, r_nla: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"res": h.float() - r_ol.float(), "resn": r_nla.float() - r_ol.float()}


def construct(h: torch.Tensor, r_ol: torch.Tensor, r_nla: torch.Tensor, j: np.ndarray) -> dict[str, torch.Tensor]:
    """{variant: [N, d]} for every variant in spec16.VARIANTS."""
    d = decompose(h, r_ol, r_nla)
    jt = torch.as_tensor(j)
    out = {"h": h.float(), "r_nla": r_nla.float(), "r_ol": r_ol.float(),
           "swap_res": r_ol.float() + d["res"][jt], "swap_ol": r_ol.float()[jt] + d["res"],
           "swap_resn": r_ol.float() + d["resn"][jt], "ol_only": r_ol.float()}
    assert set(out) == set(VARIANTS)
    return out


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None]).mean()
    eq = (pos[:, None] == neg[None]).mean()
    return float(gt + 0.5 * eq)


def follow(scores: np.ndarray, y: np.ndarray, j: np.ndarray) -> dict[str, float]:
    """AUROC of a readout against item i's label and against partner j's label.
    With j of the opposite class, follow_j = 1 - follow_i exactly; both reported for reading ease."""
    y = np.asarray(y)
    return {"follows_i": _auroc(scores, y), "follows_j": _auroc(scores, y[np.asarray(j)])}


def fve_rows(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """Fraction of variance explained of ``pred`` for ``gold`` (rows), plain raw-space version."""
    g = gold.float()
    base = ((g - g.mean(0, keepdim=True)) ** 2).mean()
    return float(1.0 - ((pred.float() - g) ** 2).mean() / base)
