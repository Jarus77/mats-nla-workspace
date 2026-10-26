"""Pure helpers for the exp 18 instruments (tested locally, no model): phrasing fill, set log-odds, the
centred push, isotropic nulls, null bands, and the prereg's decision and prediction rules."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from feature_spec18 import THRESH


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
            "stmt": ge(instr["stmt"], THRESH["stmt"]), "used": ge(instr["used_z"], THRESH["used_z"]),
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
