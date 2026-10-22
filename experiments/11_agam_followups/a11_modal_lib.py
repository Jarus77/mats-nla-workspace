"""Pure helpers shared by the Phase B Modal job (a11_modal_rfm.py) and local analysis.

Frozen choices for the layer x position sweep (PILOT / EXPLORATORY):
  LAYERS      block outputs captured (L = hidden_states[L+1]); OL is scored on its trained grid + 42
  POSITIONS   read token and the three preceding tokens (negative indices from the end of the item)
  directions  class-mean (long vs short line by median split of col; Past vs Pres for tense), fit on the
              pilot items at each (layer, position); J-removed at 4r* = 1536 of that layer's basis
  dose        q90 - median of the items' own natural projection (the 08 bank is not captured off 42)
  nulls       K isotropic + K J-complement unit directions at the J-removed dose
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

LAYERS = (24, 28, 32, 36, 40, 42, 44, 48, 52, 56)
OL_LAYERS = (36, 40, 42, 44, 48, 52)
POSITIONS = (-1, -2, -3, -4)
R_REMOVE = 1536
K_NULL = 10
PUNCT_CHARS = ".,;:!?)(\"'’“”—–-"

# (family, a, b): a = the "+" side (longer line / past), b = the "-" side.
TEMPLATES: dict[str, dict[str, list[tuple[str, str, str]]]] = {
    "col": {
        "ol": [("nl_cont", "- \nthe", "- the"), ("nl_cont", "- \nand", "- and"), ("nl_cont", "- \nof", "- of"),
               ("fixed", "- near the end of a long line", "- at the start of a line")],
        "nla": [("m2", "Mid-sentence continuation located near the end of a hard-wrapped line.",
                 "Mid-sentence continuation located at the beginning of a hard-wrapped line."),
                ("m3", "Readout taken late in the span of the wrapped line.", "Readout taken early in the span of the wrapped line."),
                ("fixed", "The current position is near the end of a long line", "The current position is at the start of a line")],
    },
    "tense": {
        "ol": [("f07", "- They were", "- They are"), ("f07", "- The village had", "- The village has"), ("f07", "- It was", "- It is")],
        "nla": [("f07", "Historical register", "Descriptive register"), ("f07", "Historical factual", "Descriptive factual"),
                ("f07", "Historical tone", "Descriptive tone")],
    },
}
# J-Lens word pairs for tense (token variants are resolved in the job); col uses the EOL (newline) set.
JLENS_TENSE_WORDS = ([" was", " were", " had"], [" is", " are", " has"])


def median_split(values: np.ndarray) -> np.ndarray:
    """1 for values strictly above the median, else 0 (balanced for distinct values)."""
    v = np.asarray(values, float)
    return (v > np.median(v)).astype(int)


def class_mean_direction(X: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    yb = torch.as_tensor(np.asarray(y).astype(bool), device=X.device)
    v = X[yb].float().mean(0) - X[~yb].float().mean(0)
    return v / v.norm()


def q90_dose(X: torch.Tensor, v: torch.Tensor) -> float:
    proj = (X.float() @ v.float()).cpu()
    return float(torch.quantile(proj, 0.9) - proj.median())


def centred(H: torch.Tensor, v: torch.Tensor, dose: float) -> torch.Tensor:
    """h - (v.h)v + dose*v (equal norms at +dose and -dose)."""
    Hf, vf = H.float(), v.float()
    return Hf - (Hf @ vf)[:, None] * vf[None] + dose * vf[None]


def build_conditions(H: torch.Tensor, dirs: dict[str, torch.Tensor], doses: dict[str, float],
                     nulls: dict[str, list[torch.Tensor]], null_ref: str) -> list[tuple[str, torch.Tensor]]:
    """[(name, [N, d])]: zero, then +/- for each real direction, then +/- for each null at the ref dose."""
    out: list[tuple[str, torch.Tensor]] = [("zero", H.float().clone())]
    for name, v in dirs.items():
        for s in (1, -1):
            out.append((f"{name}{'+' if s > 0 else '-'}", centred(H, v, s * doses[name])))
    for fam, vs in nulls.items():
        for k, v in enumerate(vs):
            for s in (1, -1):
                out.append((f"{fam}{k}{'+' if s > 0 else '-'}", centred(H, v, s * doses[null_ref])))
    return out


def punct_token_ids(decode: Callable[[int], str], vocab_size: int) -> dict[str, list[int]]:
    """Token ids whose decoded string starts with punctuation (after stripping spaces) / contains a newline."""
    punct, newline = [], []
    for i in range(vocab_size):
        s = decode(i)
        if "\n" in s:
            newline.append(i)
        t = s.lstrip(" ")
        if t and t[0] in PUNCT_CHARS:
            punct.append(i)
    return {"punct": punct, "newline": newline}


def softmax_mass(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Probability mass on ``ids`` per row of [N, V] logits."""
    p = torch.softmax(logits.float(), dim=-1)
    return p[:, ids].sum(-1)
