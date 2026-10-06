"""09 conditions (centred construction h - (v.h)v + s*D*v at the L42 read token; shared by every stage).

A set spec holds h [N, d], dirs {name: unit}, doses {name: {"q90": float}} (bank-based, col/tense only),
nulls {"iso_k", "nonj4_k"} (the 08 col-pilot null vectors, K = 20). Dose keys: q90 (scalar of the ref
direction) or rNNN (per item, NNN/1000 * ||h_i||). Condition = (kind, name, q, sign, ref).
"""
from __future__ import annotations

import torch

K_NULL = 20
REL = {"r054": 0.054, "r108": 0.108, "r216": 0.216}
ZERO = ("zero", None, None, 0, None)

PLAN = {
    "codeNL": {"dirs": {"indent_raw": ("r054", "r108", "r216"), "indent_jrem4": ("r054", "r108", "r216"),
                        "nextind_raw": ("r054", "r108"), "nextind_jrem4": ("r054", "r108")},
               "nulls": [("r108", None)]},
    "col": {"dirs": {"reg_raw": ("q90", "r054", "r108"), "reg_jrem4": ("q90", "r054", "r108"),
                     **{f"rand{x}": ("r108", "r216") for x in "ABCD"}},
            "nulls": [("r108", None), ("q90", "reg_raw"), ("q90", "reg_jrem4")]},
    "tense": {"dirs": {"raw": ("q90", "r054", "r108"), "jrem4": ("q90", "r054", "r108"),
                       **{f"rand{x}": ("r108", "r216") for x in "ABCD"}},
              "nulls": [("r108", None), ("q90", "raw"), ("q90", "jrem4")]},
}
PLAN["codeIN"] = PLAN["codeNL"]
SETS = ("codeNL", "codeIN", "col", "tense")


def conditions(s: str) -> list[tuple]:
    P = PLAN[s]
    out = [ZERO]
    for d, qs in P["dirs"].items():
        for q in qs:
            for sg in (1, -1):
                out.append(("real", d, q, sg, d))
    for q, ref in P["nulls"]:
        for cls in ("iso", "nonj4"):
            for k in range(K_NULL):
                for sg in (1, -1):
                    out.append(("null", f"{cls}_{k}", q, sg, ref))
    return out


def vec(sp: dict, cond: tuple) -> torch.Tensor:
    kind, name = cond[0], cond[1]
    return (sp["dirs"][name] if kind == "real" else sp["nulls"][name]).float()


def dose_of(sp: dict, cond: tuple, hnorm: torch.Tensor) -> torch.Tensor:
    """[N, 1] signed dose."""
    _kind, _name, q, sg, ref = cond
    if q in REL:
        return REL[q] * hnorm * sg
    return torch.full_like(hnorm, float(sp["doses"][ref][q]) * sg)


def build(sp: dict, cond: tuple) -> torch.Tensor:
    h = sp["h"].float()
    if cond[0] == "zero":
        return h.clone()
    v = vec(sp, cond)
    return h - (h @ v)[:, None] * v[None] + dose_of(sp, cond, h.norm(dim=1, keepdim=True)) * v[None]


def null_key(s: str, d: str, q: str) -> tuple[str, str | None] | None:
    for qq, ref in PLAN[s]["nulls"]:
        if qq == q and (ref is None or ref == d):
            return qq, ref
    return None
