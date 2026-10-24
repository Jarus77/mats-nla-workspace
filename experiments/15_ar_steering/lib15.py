"""Pure helpers for exp 15 (no GPU, no Modal): conditions, centred construction, direction
building, continuation metrics, and the paired-contrast statistics against null bands."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import torch

from spec15 import DIR_NAMES, K_NULL, NULL_CLASSES, REL_DOSE

ZERO = ("zero", None, "r11", 0)


def conditions() -> list[tuple]:
    """(kind, name, dose, sign): zero; real x {r11, q90} x {+, -}; null x {iso_k, nonj4_k} x r11 x {+, -}."""
    out: list[tuple] = [ZERO]
    for d in DIR_NAMES:
        for q in ("r11", "q90"):
            for s in (1, -1):
                out.append(("real", d, q, s))
    for cls in NULL_CLASSES:
        for k in range(K_NULL):
            for s in (1, -1):
                out.append(("null", f"{cls}_{k}", "r11", s))
    return out


def unit(v: torch.Tensor) -> torch.Tensor:
    n = v.float().norm()
    if float(n) < 1e-12:
        raise ValueError("zero vector")
    return v.float() / n


def class_mean_direction(X: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    """unit(mean(X[y == 1]) - mean(X[y == 0])); y in {0, 1}, 1 = values[1]."""
    y = np.asarray(y)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("both classes needed")
    return unit(X[torch.as_tensor(y == 1)].float().mean(0) - X[torch.as_tensor(y == 0)].float().mean(0))


def build(h: torch.Tensor, v: torch.Tensor, cond: tuple, q90: float | None = None) -> torch.Tensor:
    """[N, d] centred construction for one condition (``v`` unit; ``q90`` only for real q90 cells)."""
    kind, _name, q, s = cond
    hf = h.float()
    if kind == "zero":
        return hf.clone()
    vf = unit(v)
    if q == "r11":
        dose = REL_DOSE * hf.norm(dim=1, keepdim=True) * s
    else:
        if q90 is None:
            raise ValueError("q90 dose needed")
        dose = float(q90) * s
    return hf - (hf @ vf)[:, None] * vf[None] + dose * vf[None]


def q90_dose(h: torch.Tensor, v: torch.Tensor) -> float:
    """q90 - median of the natural projection on unit v (prereg dose rule, no 2x step)."""
    p = (h.float() @ unit(v)).double()
    return float(torch.quantile(p, 0.9) - torch.quantile(p, 0.5))


# ----------------------------------------------------------------------------- continuation metrics


def cells_to_newline(text: str) -> tuple[int, bool]:
    """Markdown-table continuation: number of '|' separators before the first newline.
    Returns (count, saw_newline). Without a newline the count covers the whole text."""
    head, nl, _ = text.partition("\n")
    return head.count("|"), bool(nl)


def tokens_to_close(token_strs: Sequence[str]) -> tuple[int, bool]:
    """Index of the first generated token containing ')' (count of tokens before it), and
    whether one was found. Not found -> (len(tokens), False)."""
    for i, t in enumerate(token_strs):
        if ")" in t:
            return i, True
    return len(token_strs), False


def chars_to_close(text: str) -> tuple[int, bool]:
    m = re.search(r"\)", text)
    return (m.start(), True) if m else (len(text), False)


# ----------------------------------------------------------------------------- statistics


def paired_shift(plus: np.ndarray, minus: np.ndarray) -> dict[str, float]:
    p, m = np.asarray(plus, float), np.asarray(minus, float)
    d = p - m
    return {"mean_shift": float(d.mean()), "paired_win": float(np.mean((d > 0) + 0.5 * (d == 0))),
            "n": int(len(d))}


def band(stat: float, nulls: Sequence[float]) -> dict[str, float]:
    n = np.asarray([x for x in nulls if np.isfinite(x)], float)
    sd = float(n.std(ddof=1)) if len(n) > 1 else float("nan")
    lo, hi = float(np.quantile(n, 0.025)), float(np.quantile(n, 0.975))
    return {"stat": float(stat), "null_mean": float(n.mean()), "null_sd": sd,
            "z": float((stat - n.mean()) / sd) if sd and sd > 0 else float("nan"),
            "band_lo": lo, "band_hi": hi, "outside": bool(stat < lo or stat > hi), "k": int(len(n))}


def contrast_vs_nulls(
    real_plus: np.ndarray, real_minus: np.ndarray,
    null_plus: Sequence[np.ndarray], null_minus: Sequence[np.ndarray],
) -> dict[str, dict]:
    """Real +/- paired mean shift placed against the K null +/- mean shifts (same items, same norm)."""
    r = paired_shift(real_plus, real_minus)
    ns = [paired_shift(p, m)["mean_shift"] for p, m in zip(null_plus, null_minus)]
    return {"real": r, "band": band(r["mean_shift"], ns)}


def cosines(dirs: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    names = list(dirs)
    return {a: {b: float(unit(dirs[a]) @ unit(dirs[b])) for b in names} for a in names}
