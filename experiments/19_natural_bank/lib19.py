"""Pure helpers for the exp 18 instruments (tested locally, no model): phrasing fill, set log-odds, the
centred push, isotropic nulls, null bands, and the prereg's decision and prediction rules."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from feature_spec19 import THRESH


def fill(template: dict, value: str) -> str:
    return template["template"].replace("{v}", template["fillers"][value])


def set_logodds(lp: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    """[N] log-odds of the token set: log p(set) - log(1 - p(set)) from log-probs [N, V]."""
    a = torch.logsumexp(lp[:, list(ids)], -1)
    return a - torch.log1p(-a.exp().clamp(max=1 - 1e-6))


def unit(v: torch.Tensor) -> torch.Tensor:
    n = v.float().norm()
    if float(n) < 1e-12:
        raise ValueError("zero vector")
    return v.float() / n


def class_mean_direction(X: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    y = np.asarray(y)
    return unit(X[torch.as_tensor(y == 1)].float().mean(0) - X[torch.as_tensor(y == 0)].float().mean(0))


def centred_push(h: torch.Tensor, v: torch.Tensor, rel: float, sign: int) -> torch.Tensor:
    """h - (v.h) v + sign * rel * ||h|| * v  (exp 08/15 centred construction, unit v)."""
    hf, vf = h.float(), unit(v)
    dose = rel * hf.norm(dim=1, keepdim=True) * sign
    return hf - (hf @ vf)[:, None] * vf[None] + dose * vf[None]


def iso_nulls(d: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(k, d, generator=g)
    return v / v.norm(dim=1, keepdim=True)


def band(stat: float, nulls: Sequence[float]) -> dict[str, float]:
    n = np.asarray([x for x in nulls if np.isfinite(x)], float)
    sd = float(n.std(ddof=1)) if len(n) > 1 else float("nan")
    return {"stat": float(stat), "null_mean": float(n.mean()) if len(n) else float("nan"), "null_sd": sd,
            "z": float((stat - n.mean()) / sd) if sd and sd > 0 else float("nan"), "k": int(len(n))}


def decide(instr: dict) -> dict[str, bool]:
    """Instrument values -> booleans at the prereg thresholds (NaN never passes)."""
    def ge(x, t):  # noqa: ANN001, ANN202
        return bool(x is not None and not (isinstance(x, float) and math.isnan(x)) and x >= t)
    return {"readable": ge(instr["readable"], THRESH["readable"]), "span": ge(instr["span"], THRESH["span"]),
            "stmt": ge(instr["stmt"], THRESH["stmt"]),
            "used": ge(abs(instr["used_z"]) if isinstance(instr["used_z"], (int, float)) else instr["used_z"], THRESH["used_z"]),  # two-sided
            "jenr": ge(instr["jenr"], THRESH["jenr"])}


def predict(d: dict[str, bool], flags: dict) -> dict[str, bool]:
    """Prereg §3 prediction rules; True = predicted high."""
    return {"nla_prompted": d["readable"],
            "nla_states": d["used"] and bool(flags["constrains_next"]),
            "nla_names_complement": False,
            "ol_prompted": d["span"],
            "ol_exhibits": d["span"] and bool(flags["short_value"]),
            "ol_states": False,
            "jlens_prompted": d["jenr"] or d["stmt"]}


def frozen_probe_auroc(X_true: torch.Tensor, R: torch.Tensor, y: np.ndarray, groups: np.ndarray, k: int = 5,
                       lam: float = 1e-3, seed: int = 0) -> float:
    """Exp 14's 'sayable' measure: a probe fit on the true activations (grouped folds), applied frozen to the
    reconstructions of the held-out items; AUROC against the labels."""
    from mats.metrics import auroc_labels
    from mats.probes import fit_logreg, group_folds

    y = np.asarray(y)
    sc = np.empty(len(y))
    for te in group_folds(np.asarray(groups), k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = fit_logreg(X_true[tr].float(), torch.as_tensor(y[tr]), lam)
        sc[te] = m.decision(R[te]).numpy()   # standardised with the TRUE-activation moments
    return float(auroc_labels(sc, y))


# ------------------------------------------------------------------ lexical cue check (exp 19 bank admission)
from collections import Counter
LEX_WINDOW = 48      # last tokens of each item used for the bag-of-tokens cue check
MIN_DF = 3


def bag_of_tokens(token_ids: list[list[int]], window: int = LEX_WINDOW, min_df: int = MIN_DF, keep=None) -> torch.Tensor:  # noqa: ANN001
    """keep(token_id) -> bool restricts the bag (exp 19 uses alphabetic word tokens only, so that the structural marker
    tokens a row is defined by, such as #, { or an open quote, do not count as a vocabulary cue)."""
    token_ids = [[t for t in ids[-window:] if keep is None or keep(t)] for ids in token_ids]
    df = Counter(t for ids in token_ids for t in set(ids))
    vocab = {t: i for i, t in enumerate(sorted(t for t, c in df.items() if c >= min_df))}
    X = torch.zeros((len(token_ids), max(1, len(vocab))), dtype=torch.float32)
    for r, ids in enumerate(token_ids):
        for t in set(ids):
            if t in vocab:
                X[r, vocab[t]] = 1.0
    return X


def lexical_cue_auroc(token_ids: list[list[int]], y: np.ndarray, groups: np.ndarray, window: int = LEX_WINDOW, keep=None) -> float:  # noqa: ANN001
    """Grouped-CV AUROC of a bag-of-tokens probe on the last `window` tokens. 0.5 means the two values share
    vocabulary; a high value means a text model could tell them apart with no activation at all, which is what
    the synthetic banks of exp 18 leaked."""
    from mats.probes import cv_auroc

    X = bag_of_tokens(token_ids, window, keep=keep)
    k = min(5, len(np.unique(groups)))
    return cv_auroc(X, np.asarray(y), np.asarray(groups), k=k, lam=1.0)["auroc"]
