"""Deterministic condition construction for the col pilot (shared by prep, scorers, analysis).

spec (saved by prep) holds, per read layer L in {40, 42, 44}:
  h[L]      [N, d] natural activations at the read token (block-L output)
  dirs[L]   {"raw", "jrem1", "jrem4"}: unit col directions (02's covariance direction; J-removed
            at r*(L) and 4r*(L))
  doses[L]  {dir: {"q90", "q99"}} = q - median of the natural projection on 02's wrap bank
  nulls[L]  {"iso{k}", "nonj1_{k}", "nonj4_{k}"}: unit random directions (isotropic; J-complement
            at r*(L); J-complement at 4r*(L)), k < K_NULL

Constructions:
  centred  h - (v.h) v + s*D*v   PRIMARY (D11: equal norm on both sides)
  add      h + s*D*v             reported (plain addition, no renormalisation)

A condition is (kind, name, variant, q, sign, ref): kind zero|real|null; nulls use the dose of
real direction ``ref`` (same norm, same +/- construction).
"""

from __future__ import annotations

import torch

REAL_DIRS = ("raw", "jrem1", "jrem4")
VARIANTS = ("centred", "add")
DOSES = ("q90", "q99")
K_NULL = 20
# null classes matched to each real direction (same norm as that direction's dose)
NULL_SETS = {"raw": ("iso",), "jrem1": ("iso", "nonj1"), "jrem4": ("iso", "nonj4")}
ZERO = ("zero", None, "add", None, 0, None)


def conditions(layer: int, with_nulls: bool = True) -> list[tuple]:
    """Conditions scored at ``layer``. L42 (primary): everything. L40/L44 (OL sensitivity):
    real directions plus the jrem1 null sets."""
    out = [ZERO]
    for d in REAL_DIRS:
        for v in VARIANTS:
            for q in DOSES:
                for s in (1, -1):
                    out.append(("real", d, v, q, s, d))
    if not with_nulls:
        return out
    refs = REAL_DIRS if layer == 42 else ("jrem1",)
    for ref in refs:
        for cls in NULL_SETS[ref]:
            for q in DOSES:
                for k in range(K_NULL):
                    for s in (1, -1):
                        out.append(("null", f"{cls}_{k}", "centred", q, s, ref))
    return out


JLENS_CONDS = conditions(42)  # J-Lens is cheap: the full L42 list at every layer (40/42/44)


def direction(spec: dict, layer: int, cond: tuple) -> torch.Tensor:
    kind, name = cond[0], cond[1]
    return spec["dirs"][layer][name] if kind == "real" else spec["nulls"][layer][name]


def build(spec: dict, layer: int, cond: tuple) -> torch.Tensor:
    """[N, d] reader inputs for one condition."""
    kind, _name, variant, q, s, ref = cond
    h = spec["h"][layer].float()
    if kind == "zero":
        return h.clone()
    v = direction(spec, layer, cond).float()
    dose = float(spec["doses"][layer][ref][q]) * s
    if variant == "centred":
        return h - (h @ v)[:, None] * v[None] + dose * v[None]
    return h + dose * v[None]


def build_all(spec: dict, layer: int, conds: list[tuple]) -> torch.Tensor:
    """[len(conds) * N, d], condition-major."""
    return torch.cat([build(spec, layer, c) for c in conds])


def eol_token_ids(tok) -> dict[str, list[int]]:  # noqa: ANN001
    """Newline / end-of-line leak sets (lead, Fable round 4).

    narrow: tokens whose decoded string is whitespace containing a newline ("\\n", "\\n\\n", ...).
    broad:  every token whose decoded string contains a newline (".\\n", ",\\n", "\\n\\t", ...)."""
    narrow, broad = [], []
    for i in range(len(tok)):
        s = tok.decode([i])
        if "\n" in s:
            broad.append(i)
            if not s.strip():
                narrow.append(i)
    return {"narrow": narrow, "broad": broad}
