"""Pure helpers for exp 24 (a second, independent decoder for what the readers' outputs carry: the Oracle Lens
reconstructor instead of the NLA critic, plus a shuffled-readout control).

Everything here is scored inside the OL-AR's own output space: the reference for a row is the OL-AR's rebuild of the
true last-32-token span, never a raw activation (the AR's whitening moments are gated; see exp 14's loader note).
"""
from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch

CONDITIONS = ("span_32", "span_128", "bag_32", "ol_readout", "ol_readout_shuffled", "nla_readout", "nla_readout_shuffled")
LAYERS = (44, 40)             # the OL-AR has tags for 20..60 step 4 and 63 (no 42); exp 14 used 44 primary, 40 check
SPAN_TOKENS = 32              # the AR's training crop (exp 14)
LONG_SPAN_TOKENS = 128
MAX_TEXT_TOKENS = 256         # exp 14's cap for reader texts fed to the AR
SEED = 24
EMPTY = "(empty)"
_LAM = 1e-3                   # exp 14 / exp 20's L2 penalty; C scales it as lam = _LAM / C
SCORES = (("frozen_auroc", "Frozen-probe AUROC (probe fit on the span_32 rebuilds, grouped folds, applied frozen)"),
          ("cv_auroc", "Grouped cross-validated probe AUROC on the condition's rebuild alone"),
          ("class_direction_cos", "Cosine of the class-mean direction with span_32's class-mean direction"))


def shuffle_tokens(ids: list[int], rng: random.Random) -> list[int]:
    """The same tokens in a random order (structure broken, vocabulary kept)."""
    out = list(ids)
    for _ in range(10):
        rng.shuffle(out)
        if out != list(ids):
            break
    return out


def frozen_probe_auroc(X_ref: torch.Tensor, R: torch.Tensor, y: np.ndarray, groups: np.ndarray, k: int = 5,
                       C: float = 1.0, seed: int = 0) -> float:
    """A probe fit on the reference rebuilds (grouped folds; standardised with the training reference's mean/std,
    which ``fit_logreg`` stores and ``decision`` reapplies), scored frozen on the condition's rebuild of the held-out
    items; AUROC against the labels."""
    from mats.metrics import auroc_labels
    from mats.probes import fit_logreg, group_folds

    y = np.asarray(y)
    sc = np.empty(len(y))
    for te in group_folds(np.asarray(groups), k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = fit_logreg(X_ref[tr].float(), torch.as_tensor(y[tr]), _LAM / C)
        sc[te] = m.decision(R[te].float()).numpy()
    return float(auroc_labels(sc, y))


def class_direction_cos(X_ref: torch.Tensor, R: torch.Tensor, y: np.ndarray) -> float:
    """Cosine between the condition's class-mean difference and the reference's."""
    yb = torch.as_tensor(np.asarray(y).astype(bool))
    d_ref = X_ref[yb].float().mean(0) - X_ref[~yb].float().mean(0)
    d_r = R[yb].float().mean(0) - R[~yb].float().mean(0)
    return float(torch.nn.functional.cosine_similarity(d_r, d_ref, dim=0))


def _readout_ids(text: str | None, tok_encode: Callable[[str], list[int]], cap: int) -> list[int]:
    return tok_encode((text or "").strip() or EMPTY)[:cap]


def conditions(span_ids: list[int], long_ids: list[int], ol_text: str | None, nla_text: str | None,
               tok_encode: Callable[[str], list[int]], rng: random.Random, cap: int = MAX_TEXT_TOKENS) -> dict[str, list[int]]:
    """One item's token-id rows for every condition, in ``CONDITIONS`` order. ``span_ids`` / ``long_ids`` are the
    true last 32 / 128 tokens; readouts are encoded, capped at ``cap`` tokens, empty ones become ``(empty)``; the
    shuffled variants are the same tokens in ``rng``'s order."""
    ol = _readout_ids(ol_text, tok_encode, cap)
    nla = _readout_ids(nla_text, tok_encode, cap)
    return {"span_32": list(span_ids), "span_128": list(long_ids), "bag_32": shuffle_tokens(list(span_ids), rng),
            "ol_readout": ol, "ol_readout_shuffled": shuffle_tokens(ol, rng),
            "nla_readout": nla, "nla_readout_shuffled": shuffle_tokens(nla, rng)}


def _sub_table(rows: list[dict], layer: str, score: str) -> list[str]:
    out = ["| row | " + " | ".join(CONDITIONS) + " |", "|---|" + "---:|" * len(CONDITIONS)]
    for r in rows:
        v = r["layers"].get(layer, {}).get(score, {})
        out.append(f"| {r['feature']} | " + " | ".join(f"{v[c]:.2f}" if c in v else "" for c in CONDITIONS) + " |")
    return out


def table(rows: list[dict], layer: str = "44") -> str:
    """Three markdown sub-tables (frozen-probe AUROC, CV AUROC, class-direction cosine) for one AR layer."""
    out = []
    for score, title in SCORES:
        out += [f"### {title}, OL-AR layer {layer}", ""] + _sub_table(rows, str(layer), score) + [""]
    return "\n".join(out).rstrip() + "\n"
