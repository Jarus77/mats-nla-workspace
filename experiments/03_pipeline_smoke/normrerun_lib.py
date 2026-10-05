"""Shared, deterministic construction of the norm-preserving re-run conditions (prep + scorers).

A "spec" (saved by the prep stage) holds, per layer L: natural last-token activations h_L [N, d],
unit directions dirs[L][(name, fold)], doses[L][(name, fold)] = {"q90", "q99"}, and the null
directions nulls[L][(kind, k)]. :func:`conditions` lists every condition; :func:`build` returns
the [N, d] reader inputs for one condition. Folds: items with pair % 2 == f use fold-f
directions, which were fitted on the other fold (2-fold cross-fitting, as in the smoke).

Variants:
  add      h + s*D*v                              (original smoke construction)
  preserve (h + s*D*v) * ||h|| / ||h + s*D*v||    (Fable's norm-preserving steer)
  centred  h - (v.h) v + s*D*v                    (equal norms on both sides of each item)
"""

from __future__ import annotations

import numpy as np
import torch

from mats.capture import centred_steer_vectors, steer_vectors

REAL_DIRS = ("raw", "jrem", "inlp1_jrem", "inlp8_jrem")
VARIANTS = ("add", "preserve", "centred")
NULL_KINDS = ("nonj", "iso")
K_NULL = 20


def conditions(include_nulls: bool = True, null_variants: tuple[str, ...] = ("add", "centred"),
               null_norms: tuple[str, ...] = ("raw", "jrem")) -> list[tuple]:
    """(kind, dir_name, variant, dose_key, sign, norm_ref). kind = zero | real | null.

    Null conditions borrow the q99 dose of the real direction ``norm_ref`` (same norm, same +/-
    construction). For centred nulls only the jrem norm is used (the anti-tracking question)."""
    out = [("zero", None, "add", None, 0, None)]
    for d in REAL_DIRS:
        for v in VARIANTS:
            for q in ("q90", "q99"):
                for s in (1, -1):
                    out.append(("real", d, v, q, s, d))
    if include_nulls:
        for v in null_variants:
            norms = null_norms if v != "centred" else ("jrem",)
            for ref in norms:
                for kind in NULL_KINDS:
                    for k in range(K_NULL):
                        for s in (1, -1):
                            out.append(("null", f"{kind}{k}", v, "q99", s, ref))
    return out


def build(spec: dict, layer: int, cond: tuple, fold: np.ndarray) -> torch.Tensor:
    kind, name, variant, q, s, ref = cond
    h = spec["h"][layer].float()
    if kind == "zero":
        return h.clone()
    out = torch.empty_like(h)
    for f in (0, 1):
        te = np.flatnonzero(fold == f)
        if kind == "real":
            v = spec["dirs"][layer][(name, f)]
        else:
            v = spec["nulls"][layer][name]
        dose = float(spec["doses"][layer][(ref, f)][q]) * s
        if variant == "centred":
            out[te] = centred_steer_vectors(h[te], v, [dose])[0]
        else:
            out[te] = steer_vectors(h[te], v, [dose], preserve_norm=(variant == "preserve"))[0]
    return out


def build_all(spec: dict, layer: int, conds: list[tuple], fold: np.ndarray) -> torch.Tensor:
    """[len(conds) * N, d] in condition-major order."""
    return torch.cat([build(spec, layer, c, fold) for c in conds])
