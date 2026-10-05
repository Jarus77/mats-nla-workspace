"""Round-5 conditions (centred construction throughout; L42 read layer).

A set (col / digit / tense) has h [N, d], dirs {name: unit}, doses {name: {"q90", "q99"}} (scalar, bank-based),
nulls {"iso_k", "nonj4_k"}. Dose keys: q90/q99 (scalar from the ref direction) or r11/r14 (per item,
0.11/0.14 * ||h_i||). Condition = (kind, name, q, sign, ref).
"""
from __future__ import annotations

import torch

K_NULL = 20
REL = {"r11": 0.11, "r14": 0.14}
ZERO = ("zero", None, None, 0, None)

PLAN = {
    "col": {"dirs": ("reg_raw", "reg_jrem4", "cm_raw", "cm_jrem4"), "doses": ("q90", "r11"),
            "nulls": [("r11", None), ("q90", "reg_raw")]},
    "digit": {"dirs": ("raw", "jrem4"), "doses": ("q90", "r11", "r14"), "nulls": [("r11", None)]},
    "tense": {"dirs": ("raw", "jrem4"), "doses": ("q90", "r11"),
              "nulls": [("r11", None), ("q90", "raw"), ("q90", "jrem4")]},
}


def conditions(s: str) -> list[tuple]:
    P = PLAN[s]
    out = [ZERO]
    for d in P["dirs"]:
        for q in P["doses"]:
            for sg in (1, -1):
                out.append(("real", d, q, sg, d))
    for q, ref in P["nulls"]:
        for cls in ("iso", "nonj4"):
            for k in range(K_NULL):
                for sg in (1, -1):
                    out.append(("null", f"{cls}_{k}", q, sg, ref))
    return out


def null_keys(s: str, d: str, q: str) -> tuple[str | None, bool]:
    """Which null block matches the real cell (d, q): returns ref (None for relative doses)."""
    for qq, ref in PLAN[s]["nulls"]:
        if qq == q and (ref is None or ref == d):
            return ref, True
    return None, False


def build(sp: dict, cond: tuple) -> torch.Tensor:
    kind, name, q, sg, ref = cond
    h = sp["h"].float()
    if kind == "zero":
        return h.clone()
    v = (sp["dirs"][name] if kind == "real" else sp["nulls"][name]).float()
    dose = REL[q] * h.norm(dim=1, keepdim=True) * sg if q in REL else float(sp["doses"][ref][q]) * sg
    return h - (h @ v)[:, None] * v[None] + dose * v[None]
