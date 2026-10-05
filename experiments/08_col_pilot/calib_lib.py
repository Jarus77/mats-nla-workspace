"""Conditions for the matched-dose in-J calibration (K3). Same centred construction as the col pilot.

spec[F] for feature F in {lang, tense, col}, per layer L in {40, 42, 44}:
  h[L] [N, d]; dirs[L] {"raw", "jrem4"}; doses[L][dir] {"q90", "q99"} (bank percentiles, scalar);
  nulls[L] {"iso_k", "nonj4_k"}.
Relative-norm doses "r11"/"r14": per-item Delta_i = 0.11 / 0.14 * ||h_i|| (dose-matched in
relative-norm units across features).

Condition = (kind, name, q, sign, ref); kind zero|real|null; centred construction everywhere.
"""
from __future__ import annotations

import torch

K_NULL = 20
REL = {"r11": 0.11, "r14": 0.14}
ZERO = ("zero", None, None, 0, None)


def dose_keys(feature: str, d: str) -> tuple[str, ...]:
    if feature == "col":  # q90/q99 already scored in the col pilot; only the dose-matched arm here
        return ("q90", "r11", "r14") if d == "raw" else ()
    return ("q90", "q99", "r11", "r14") if d == "raw" else ("q90", "q99")


def conditions(feature: str, layer: int) -> list[tuple]:
    out = [ZERO]
    for d in ("raw", "jrem4"):
        for q in dose_keys(feature, d):
            for s in (1, -1):
                out.append(("real", d, q, s, d))
    if layer != 42:  # OL sensitivity layers: real directions only
        return out
    for d in ("raw", "jrem4"):
        for q in dose_keys(feature, d):
            if feature == "col" and q == "q90":
                continue
            for cls in ("iso", "nonj4"):
                for k in range(K_NULL):
                    for s in (1, -1):
                        out.append(("null", f"{cls}_{k}", q, s, d))
    return out


def jlens_conditions(feature: str) -> list[tuple]:
    return conditions(feature, 42)


def build(sp: dict, layer: int, cond: tuple) -> torch.Tensor:
    kind, name, q, s, ref = cond
    h = sp["h"][layer].float()
    if kind == "zero":
        return h.clone()
    v = (sp["dirs"][layer][name] if kind == "real" else sp["nulls"][layer][name]).float()
    if q in REL:
        dose = REL[q] * h.norm(dim=1, keepdim=True) * s
    else:
        dose = float(sp["doses"][layer][ref][q]) * s
    return h - (h @ v)[:, None] * v[None] + dose * v[None]
