"""Pure helpers for exp 20 (what the NLA reconstructor keeps from ordered spans, bags of the same tokens, and the
readers' own words)."""
from __future__ import annotations

import random
import re

import numpy as np
import torch

WINDOWS = (4, 8, 16, 32, 64, 128)
CONDITIONS = tuple(f"model_span_{w}" for w in WINDOWS) + tuple(f"span_{w}" for w in WINDOWS) + ("bag_32", "ol_readout", "nla_readout")
_EXPL = re.compile(r"<explanation>(.*?)</explanation>", re.S)


def windows(ids: list[int], sizes=WINDOWS) -> dict[int, list[int]]:  # noqa: ANN001
    return {w: ids[-w:] for w in sizes}


def shuffle_tokens(ids: list[int], rng: random.Random) -> list[int]:
    """The same tokens in a random order (structure broken, vocabulary kept)."""
    out = list(ids)
    for _ in range(10):
        rng.shuffle(out)
        if out != list(ids):
            break
    return out


def explanation_text(raw: str) -> str:
    m = _EXPL.search(raw)
    t = (m.group(1) if m else raw.split("</explanation>")[0]).strip()
    return t if t else "(empty)"


def frozen_probe_auroc(X_true: torch.Tensor, R: torch.Tensor, y: np.ndarray, groups: np.ndarray, k: int = 5,
                       lam: float = 1e-3, seed: int = 0) -> float:
    """Exp 14's measure: a probe fit on the true activations (grouped folds), applied frozen to the reconstructions
    of the held-out items; AUROC against the labels."""
    from mats.metrics import auroc_labels
    from mats.probes import fit_logreg, group_folds

    y = np.asarray(y)
    sc = np.empty(len(y))
    for te in group_folds(np.asarray(groups), k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = fit_logreg(X_true[tr].float(), torch.as_tensor(y[tr]), lam)
        sc[te] = m.decision(R[te]).numpy()
    return float(auroc_labels(sc, y))


def table(rows: list[dict]) -> str:
    conds = [c for c in CONDITIONS if any(c in r["auroc"] for r in rows)]
    out = ["| row | " + " | ".join(conds) + " |", "|---|" + "---:|" * len(conds)]
    for r in rows:
        out.append(f"| {r['feature']} | " + " | ".join(f"{r['auroc'][c]:.2f}" if c in r["auroc"] else "" for c in conds) + " |")
    return "\n".join(out)
