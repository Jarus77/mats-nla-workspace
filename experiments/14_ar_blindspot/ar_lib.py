"""Pure helpers for exp 14 (torch only; no model code). Tested in test_ar_lib.py.

- span crops and LoRA key renaming for the Oracle Lens reconstructor (AR) loader;
- reconstruction quality (FVE, retrieval@1) and concept retention (frozen-probe AUROC on the
  reconstruction and on the error term, gap retention along the probe direction).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from mats.metrics import auroc
from mats.probes import fit_logreg, group_folds


# ----------------------------------------------------------------------------- inputs
def span_crop(ids: Sequence[int], n: int) -> list[int]:
    """The last ``n`` token ids of a prefix (the AR reads a short span ending at the read token)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return list(ids[-n:])


def rename_lora_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the torch.compile ``_orig_mod.`` wrapper segment so PEFT matches the plain backbone."""
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def right_pad(rows: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ``[B, T]`` ids and mask; ``mask.sum(1) - 1`` is each row's last real token."""
    n = max(len(r) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(list(r), dtype=torch.long)
        mask[i, : len(r)] = 1
    return ids, mask


# ----------------------------------------------------------------------------- quality
def fve_raw(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """1 - MSE(pred, gold) / predict-the-mean MSE of ``gold`` (raw space, no whitening)."""
    pred, gold = pred.float(), gold.float()
    base = ((gold - gold.mean(0)) ** 2).mean()
    return float(1 - ((pred - gold) ** 2).mean() / base)


def retrieval_at_1(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """Fraction of rows whose nearest gold vector (cosine, after centring on gold's mean) is its own."""
    pred, gold = pred.float(), gold.float()
    mu = gold.mean(0)
    p = torch.nn.functional.normalize(pred - mu, dim=-1)
    g = torch.nn.functional.normalize(gold - mu, dim=-1)
    nearest = (p @ g.T).argmax(1)
    return float((nearest == torch.arange(len(gold))).float().mean())


# ----------------------------------------------------------------------------- retention
def frozen_probe_retention(
    true: torch.Tensor, recon: torch.Tensor, y: np.ndarray, groups: np.ndarray, k: int = 5,
    lam: float = 1e-3, seed: int = 0,
) -> dict[str, float]:
    """Grouped k-fold: probe fit on TRUE[train]; AUROC of that probe on TRUE[test], RECON[test] and
    the error term (TRUE - RECON)[test]. 'recon' is what the ruler keeps, 'err' what it drops."""
    y = np.asarray(y)
    s_true, s_rec, s_err = (np.empty(len(y)) for _ in range(3))
    for te in group_folds(groups, k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = fit_logreg(true[tr], torch.as_tensor(y[tr]), lam)
        s_true[te] = m.decision(true[te]).cpu().numpy()
        s_rec[te] = m.decision(recon[te]).cpu().numpy()
        s_err[te] = m.decision(true[te] - recon[te]).cpu().numpy()
    return {"true": auroc(s_true[y == 1], s_true[y == 0]),
            "recon": auroc(s_rec[y == 1], s_rec[y == 0]),
            "err": auroc(s_err[y == 1], s_err[y == 0]), "n": int(len(y))}


def retrained_probe_auroc(
    X: torch.Tensor, y: np.ndarray, groups: np.ndarray, k: int = 5, lam: float = 1e-3, seed: int = 0,
) -> float:
    """Grouped k-fold AUROC of a probe fit on X itself (is the concept linearly present at all)."""
    y = np.asarray(y)
    s = np.empty(len(y))
    for te in group_folds(groups, k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        s[te] = fit_logreg(X[tr], torch.as_tensor(y[tr]), lam).decision(X[te]).cpu().numpy()
    return auroc(s[y == 1], s[y == 0])


def gap_retention(true: torch.Tensor, recon: torch.Tensor, y: np.ndarray) -> dict[str, float]:
    """Class-mean gap of the reconstruction relative to the true gap: the projection ratio along the
    true mean-difference direction and the cosine between the two gap vectors."""
    y = np.asarray(y).astype(bool)
    d_true = true[y].float().mean(0) - true[~y].float().mean(0)
    d_rec = recon[y].float().mean(0) - recon[~y].float().mean(0)
    u = d_true / d_true.norm().clamp_min(1e-9)
    return {"ratio": float((d_rec @ u) / d_true.norm().clamp_min(1e-9)),
            "cosine": float(torch.nn.functional.cosine_similarity(d_rec, d_true, dim=0))}


def statement_shift(true: torch.Tensor, y: np.ndarray, rec_a: torch.Tensor, rec_b: torch.Tensor) -> dict[str, float]:
    """How far the reconstruction of statement A sits from statement B along the true A-B direction,
    as a fraction of the true class-mean gap (1.0 = the ruler moves the full real distance)."""
    y = np.asarray(y).astype(bool)
    d_true = true[y].float().mean(0) - true[~y].float().mean(0)
    u = d_true / d_true.norm().clamp_min(1e-9)
    delta = rec_a.float() - rec_b.float()
    return {"ratio": float((delta @ u) / d_true.norm().clamp_min(1e-9)),
            "cosine": float(torch.nn.functional.cosine_similarity(delta, d_true, dim=0))}
