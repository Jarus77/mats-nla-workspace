"""10 operand gate: labels, token sets and the per-slot gate computations (pilot, exploratory).

Feature = operand identity at the final "=" token of "a op b =" (Michael Kenney's 600 prompts, layer 42).
Qwen tokenises digits one per token, so each two-digit operand is two 10-class slots (tens, units).
Gates follow the prereg / 04 conventions through bf_gate_lib: G1 = grouped-CV logistic probes;
G2 = diff-in-means enrichment in the J-space at r* = 384 and 4r* = 1536 (random = 1.0) against a
shuffled-label reference and an isotropic random band; G3 = standard J-Lens readout
softmax(W_U norm(J h)), class label-token ranks vs a 200-permutation null (04 functional_stat).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

SLOTS = ("a_tens", "a_units", "b_tens", "b_units", "ans_tens", "ans_units")
RANKS = (384, 1536)
N_CLASSES = 10


def load_items(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def labels(items: list[dict]) -> dict[str, np.ndarray]:
    a = np.array([it["a"] for it in items])
    b = np.array([it["b"] for it in items])
    ans = np.array([it["answer"] for it in items])
    return {
        "a": a, "b": b, "answer": ans,
        "a_tens": a // 10, "a_units": a % 10, "b_tens": b // 10, "b_units": b % 10,
        "ans_tens": (ans // 10) % 10, "ans_units": ans % 10,
        "op": np.array([1 if it["op"] == "+" else 0 for it in items]),
    }


def digit_label_ids(tok: Any) -> list[list[int]]:
    from mats.readers.jlens import token_variants

    return [token_variants(tok, [str(d)]) for d in range(N_CLASSES)]


def op_label_ids(tok: Any) -> list[list[int]]:
    from mats.readers.jlens import token_variants

    return [token_variants(tok, ["-"]), token_variants(tok, ["+"])]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


# ----------------------------------------------------------------------------- G1


def ovr_cv_scores(X: torch.Tensor, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Held-out one-vs-rest logistic decision scores [N, C] (bf_gate_lib.cv_scores per class)."""
    import bf_gate_lib as G

    return np.stack([G.cv_scores(X, (y == c).astype(int), groups) for c in range(N_CLASSES)], 1)


def macro_auroc(S: np.ndarray, y: np.ndarray) -> float:
    from mats.metrics import auroc

    return float(np.mean([auroc(S[y == c, c], S[y != c, c]) for c in range(S.shape[1])]))


def g1_multiclass(S: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Accuracy (argmax) with a Wilson CI, macro one-vs-rest AUROC with an item-bootstrap CI."""
    pred = S.argmax(1)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == N_CLASSES:
            boots.append(macro_auroc(S[i], y[i]))
    k = int((pred == y).sum())
    return {"acc": k / len(y), "acc_ci95": wilson(k, len(y)), "macro_auroc": macro_auroc(S, y),
            "macro_auroc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "chance_acc": 1 / N_CLASSES, "n": int(len(y)), "pred": pred.tolist()}


# ----------------------------------------------------------------------------- G2


def enrich_rows(Q: torch.Tensor, V: torch.Tensor) -> np.ndarray:
    """Enrichment of each row of V [k, d] in the basis Q [d, r]."""
    from mats.jspace import enrichment_in

    return enrichment_in(Q, V.to(Q.device)).cpu().numpy()


def class_mean_dirs(X: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    import bf_gate_lib as G

    return torch.stack([G.mean_diff(X, (y == c).astype(int)) for c in range(N_CLASSES)])


def probe_weight_dirs(X: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    from mats.probes import fit_logreg

    return torch.stack([fit_logreg(X, torch.as_tensor((y == c).astype(int))).direction() for c in range(N_CLASSES)])


def shuffled_class_mean_dirs(X: torch.Tensor, y: np.ndarray, n_perm: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.cat([class_mean_dirs(X, rng.permutation(y)) for _ in range(n_perm)])


def summarise(e: np.ndarray) -> dict:
    return {"median": float(np.median(e)), "min": float(e.min()), "max": float(e.max()), "mean": float(e.mean())}


def g2_slot(X: torch.Tensor, y: np.ndarray, Qs: dict[int, torch.Tensor], n_perm: int = 10, seed: int = 0) -> dict:
    cm, pw = class_mean_dirs(X, y), probe_weight_dirs(X, y)
    sh = shuffled_class_mean_dirs(X, y, n_perm, seed)
    out: dict = {}
    for r, Q in Qs.items():
        out[str(r)] = {"class_mean": summarise(enrich_rows(Q, cm)), "class_mean_per_class": enrich_rows(Q, cm).tolist(),
                       "probe_weight": summarise(enrich_rows(Q, pw)),
                       "shuffled_label_class_mean": summarise(enrich_rows(Q, sh))}
    return {"ranks": out, "dirs": {"class_mean": cm.cpu(), "probe_weight": pw.cpu()}}


def random_band(Qs: dict[int, torch.Tensor], d: int, k: int = 100, seed: int = 0) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    V = torch.randn(k, d, generator=g)
    V = V / V.norm(dim=1, keepdim=True)
    out = {}
    for r, Q in Qs.items():
        e = enrich_rows(Q, V)
        out[str(r)] = {**summarise(e), "p2.5": float(np.percentile(e, 2.5)), "p97.5": float(np.percentile(e, 97.5))}
    return out


def ridge_direction(X: torch.Tensor, t: np.ndarray, lam: float = 1e-2) -> tuple[torch.Tensor, dict]:
    """Unit ridge-regression direction (standardised features) for a scalar target, plus 5-fold CV Pearson r."""
    from mats.probes import group_folds

    Xf = X.float()
    mu, sd = Xf.mean(0), Xf.std(0).clamp_min(1e-4)
    Z = (Xf - mu) / sd
    tt = torch.as_tensor(t, dtype=torch.float32, device=X.device)
    d = Z.shape[1]

    def fit(rows: np.ndarray) -> torch.Tensor:
        Zr, tr = Z[rows], tt[rows] - tt[rows].mean()
        reg = lam * (Zr ** 2).sum() / len(rows)
        return torch.linalg.solve(Zr.T @ Zr + reg * torch.eye(d, device=X.device), Zr.T @ tr)

    pred = np.empty(len(t))
    for te in group_folds(np.arange(len(t)), 5):
        tr = np.setdiff1d(np.arange(len(t)), te)
        w = fit(tr)
        pred[te] = ((Z[te] - Z[tr].mean(0)) @ w + tt[tr].mean()).cpu().numpy()
    w = fit(np.arange(len(t)))
    v = w / sd
    v = v / v.norm()
    r = float(np.corrcoef(pred, t)[0, 1])
    return v, {"cv_pearson_r": r, "cv_mae": float(np.abs(pred - t).mean())}


# ----------------------------------------------------------------------------- G3


def g3_slot(H: torch.Tensor, J: torch.Tensor, W_U: torch.Tensor, gain: torch.Tensor, label_ids: list[list[int]],
            y: np.ndarray, tok: Any) -> dict:
    import bf_gate_lib as G

    ranks, top = G.jlens_ranks_topk(H, J, W_U, gain, label_ids, k=10)
    f = G.functional_stat(ranks, y)
    true_rank = ranks[np.arange(len(y)), y]
    f["true_label_rank_median"] = float(np.median(true_rank))
    f["true_label_in_top10"] = float((true_rank < 10).mean())
    vals, cnt = np.unique(top[:, :10], return_counts=True)
    f["lens_top10_common"] = [(tok.decode([int(v)]), int(c)) for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1])[:20]]
    return f


# ----------------------------------------------------------------------------- labels


def label_slot(g1: dict, g2: dict, g3: dict, enrich_max: float = 1.5) -> str:
    """Prereg §4.2 labels: NON-J (G2 <= 1.5 at both ranks AND lens not readable), DIFFUSE-BUT-J-READABLE
    (G2 passes, lens reads), IN-J (enriched). G4 is not run at this stage, so 'decodable-unused' is not assigned."""
    if g1["macro_auroc"] < 0.6:
        return "fails G1"
    enriched = any(g2["ranks"][str(r)]["class_mean"]["median"] > enrich_max for r in RANKS)
    readable = bool(g3["j_readable"])
    if enriched:
        return "IN-J"
    return "DIFFUSE-BUT-J-READABLE" if readable else "NON-J (G1-G3; G4 pending)"
