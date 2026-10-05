"""Gate computations for binary features (torch + numpy; runs on GPU inside Modal).

Conventions copied from experiments/02_nonj_features (read-only there):
- J-space at rank r = top-r right singular vectors of W_U diag(1+w_norm) J_L (neuronpedia n1000);
  bases read from /results/03_pipeline_smoke/jspace_basis_L{38,40,42,44}.pt (overlap 1.0000 with 02).
- enrichment = ||Q_r^T d||^2 / (r/d) of the unit diff-in-means direction d (random = 1.0).
- functional = 02 s3d_functional for classes: score_c = -log10(best rank of class-c label tokens + 1)
  in the standard J-Lens readout softmax(W_U norm(J h)); stat = mean one-vs-rest AUROC; null = 200
  label permutations; J-readable iff stat > 97.5th percentile of the null.
- INLP = 8 iterated ridge-probe nullspace projections (s3c_jonly).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mats.metrics import auroc, auroc_ci
from mats.probes import fit_logreg, group_folds


def cv_scores(X: torch.Tensor, y: np.ndarray, groups: np.ndarray, k: int = 5, lam: float = 1e-3,
              seed: int = 0) -> np.ndarray:
    """Grouped k-fold held-out logistic decision scores (every row held out once)."""
    s = np.empty(len(y))
    for te in group_folds(groups, k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = fit_logreg(X[tr], torch.as_tensor(y[tr]), lam)
        s[te] = m.decision(X[te]).cpu().numpy()
    return s


def probe_report(X: torch.Tensor, y: np.ndarray, groups: np.ndarray, n_boot: int = 1000) -> dict:
    s = cv_scores(X, y, groups)
    ci = auroc_ci(s, y, groups, n_boot)
    return {"auroc": ci.auroc, "ci95": [ci.lo, ci.hi], "acc": float(((s > 0) == (y == 1)).mean()),
            "n_rows": int(len(y)), "n_groups": int(len(np.unique(groups)))}


def logloss_dprime(s: np.ndarray, y: np.ndarray) -> dict:
    p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
    ll = float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
    a, b = s[y == 1], s[y == 0]
    dp = float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12))
    return {"logloss": ll, "dprime": dp, "auroc": auroc(a, b)}


def mean_diff(X: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    yb = torch.as_tensor(y.astype(bool), device=X.device)
    v = X[yb].float().mean(0) - X[~yb].float().mean(0)
    return v / v.norm()


def enrich(Q: torch.Tensor, v: torch.Tensor) -> float:
    d, r = Q.shape
    v = v.float().to(Q.device)
    return float(((v @ Q) ** 2).sum() / (v ** 2).sum() / (r / d))


def inlp_dirs(X: torch.Tensor, y: np.ndarray, k: int = 8) -> list[torch.Tensor]:
    Xc = X.float() - X.float().mean(0)
    t = torch.as_tensor(np.where(y == 1, 1.0, -1.0), device=X.device, dtype=torch.float32)
    t = t - t.mean()
    D = X.shape[1]
    lam = 1e-2 * (Xc ** 2).sum() / Xc.shape[0]
    P = torch.eye(D, device=X.device)
    out = []
    for _ in range(k):
        Z = Xc @ P
        w = torch.linalg.solve(Z.T @ Z + lam * torch.eye(D, device=X.device), Z.T @ t)
        w = P @ w
        w = w / w.norm()
        out.append(w)
        P = P - torch.outer(w, w)
    return out


def geometric(X: torch.Tensor, y: np.ndarray, Q_by_r: dict[int, torch.Tensor], n_perm: int = 10,
              with_inlp: bool = False, seed: int = 0) -> dict:
    d = mean_diff(X, y)
    rng = np.random.default_rng(seed)
    Xc = X.float() - X.float().mean(0)
    out: dict[str, Any] = {}
    perms = [mean_diff(X, rng.permutation(y)) for _ in range(n_perm)]
    inl = inlp_dirs(X, y) if with_inlp else []
    for r, Q in Q_by_r.items():
        out[str(r)] = {
            "diffmeans": enrich(Q, d),
            "shuffled_label_diffmeans_mean": float(np.mean([enrich(Q, p) for p in perms])),
            "acts": float(((Xc @ Q) ** 2).sum() / (Xc ** 2).sum() / (r / X.shape[1])),
            "inlp": [enrich(Q, w) for w in inl],
        }
    return out


# ----------------------------------------------------------------------------- J-Lens


def rms_norm(z: torch.Tensor, gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * gain


@torch.no_grad()
def jlens_ranks_topk(H: torch.Tensor, J: torch.Tensor, W_U: torch.Tensor, gain: torch.Tensor,
                     label_ids: list[list[int]], k: int = 25, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Standard J-Lens readout logits W_U norm(J h). Returns (best rank per class [N, C], top-k ids [N, k])."""
    ranks, tops = [], []
    lab = [torch.as_tensor(ids, device=W_U.device) for ids in label_ids]
    for s in range(0, H.shape[0], chunk):
        z = rms_norm(H[s:s + chunk].to(W_U.device).float() @ J.T, gain)
        lg = z @ W_U.T
        best = torch.stack([lg[:, ix].max(1).values for ix in lab], 1)  # [n, C]
        ranks.append(torch.stack([(lg > best[:, c:c + 1]).sum(1) for c in range(len(lab))], 1).cpu())
        tops.append(lg.topk(k, dim=1).indices.cpu())
    return torch.cat(ranks).numpy(), torch.cat(tops).numpy()


@torch.no_grad()
def jlens_readout(H: torch.Tensor, J: torch.Tensor, W_U: torch.Tensor, gain: torch.Tensor,
                  label_ids: list[list[int]], sets: tuple[list[int], list[int]], k: int = 25,
                  chunk: int = 256) -> dict[str, np.ndarray]:
    """Standard J-Lens readout softmax(W_U norm(J h)) per row: label-token best ranks [N, C], top-k ids,
    and the emitted-form scores: soft = logsumexp(logp[set1]) - logsumexp(logp[set0]) (set0 empty ->
    just set1), top10 = #set1 - #set0 among the top-10 tokens."""
    lab = [torch.as_tensor(ids, device=W_U.device) for ids in label_ids]
    s0 = torch.as_tensor(sets[0], device=W_U.device, dtype=torch.long)
    s1 = torch.as_tensor(sets[1], device=W_U.device, dtype=torch.long)
    is0 = torch.zeros(W_U.shape[0], dtype=torch.bool, device=W_U.device)
    is1 = torch.zeros_like(is0)
    is0[s0], is1[s1] = True, True
    out: dict[str, list] = {"ranks": [], "top": [], "soft": [], "top10": []}
    for s in range(0, H.shape[0], chunk):
        z = rms_norm(H[s:s + chunk].to(W_U.device).float() @ J.T, gain)
        lg = z @ W_U.T
        best = torch.stack([lg[:, ix].max(1).values for ix in lab], 1)
        out["ranks"].append(torch.stack([(lg > best[:, c:c + 1]).sum(1) for c in range(len(lab))], 1).cpu())
        tk = lg.topk(max(k, 10), dim=1).indices
        out["top"].append(tk[:, :k].cpu())
        lp = torch.log_softmax(lg, -1)
        soft = torch.logsumexp(lp[:, s1], 1) - (torch.logsumexp(lp[:, s0], 1) if len(s0) else 0)
        out["soft"].append(soft.cpu())
        t10 = tk[:, :10]
        out["top10"].append((is1[t10].sum(1) - is0[t10].sum(1)).float().cpu())
    return {k_: torch.cat(v).numpy() for k_, v in out.items()}


def emitted_stat(score: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int = 4) -> dict:
    """AUROC of an emitted-form lens score vs the label, two-sided against a permutation band."""
    rng = np.random.default_rng(seed)
    stat = auroc(score[y == 1], score[y == 0])
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        null.append(auroc(score[yp == 1], score[yp == 0]))
    band = [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]
    return {"auroc": stat, "null_band95": band, "readable_two_sided": bool(stat > band[1] or stat < band[0]),
            "readable_positive": bool(stat > band[1])}


def functional_stat(ranks: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int = 3) -> dict:
    """02 s3d (classes): mean one-vs-rest AUROC of -log10(rank+1) class scores vs a permutation null."""
    S = -np.log10(ranks.astype(np.float64) + 1)

    def ovr(yy: np.ndarray) -> float:
        return float(np.mean([auroc(S[yy == c, c], S[yy != c, c]) for c in range(S.shape[1])]))

    rng = np.random.default_rng(seed)
    null = [ovr(rng.permutation(y)) for _ in range(n_perm)]
    band = [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]
    stat = ovr(y)
    return {"stat": stat, "null_band95": band, "null_mean": float(np.mean(null)),
            "j_readable": bool(stat > band[1]), "below_band": bool(stat < band[0]),
            "per_class_auroc": [float(auroc(S[y == c, c], S[y != c, c])) for c in range(S.shape[1])],
            "top10_rate": [float((ranks[:, c] < 10).mean()) for c in range(S.shape[1])]}


def bag_matrix(cols: np.ndarray, min_count: int = 3) -> torch.Tensor:
    """Dense binary bag over token ids appearing >= min_count times (cols [N, k], -1 = none)."""
    vals, cnt = np.unique(cols[cols >= 0], return_counts=True)
    keep = vals[cnt >= min_count]
    lut = {int(v): i for i, v in enumerate(keep)}
    M = torch.zeros(cols.shape[0], max(len(keep), 1))
    for i, row in enumerate(cols):
        for t in row:
            j = lut.get(int(t))
            if j is not None:
                M[i, j] = 1.0
    return M


def bag_auroc(cols: np.ndarray, y: np.ndarray, groups: np.ndarray, n_perm: int = 5) -> dict:
    M = bag_matrix(cols).cuda()
    s = cv_scores(M, y, groups, lam=1e-2)
    rng = np.random.default_rng(5)
    null = [auroc(*_split(cv_scores(M, yp, groups, lam=1e-2), yp)) for yp in
            (rng.permutation(y) for _ in range(n_perm))]
    return {"auroc": auroc(*_split(s, y)), "shuffled_max": float(np.max(null)), "dim": int(M.shape[1])}


def _split(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return s[y == 1], s[y == 0]


# ----------------------------------------------------------------------------- J-only probe (secondary)


def jonly(X: torch.Tensor, y: np.ndarray, groups: np.ndarray, Q: torch.Tensor, n_rand: int = 10,
          seed: int = 17) -> dict:
    real = logloss_dprime(cv_scores(X @ Q, y, groups), y)
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = []
    for _ in range(n_rand):
        Qr, _ = torch.linalg.qr(torch.randn(Q.shape[0], Q.shape[1], device="cuda", generator=g))
        rnd.append(logloss_dprime(cv_scores(X @ Qr, y, groups), y))
    out = {"jonly": real}
    for key in ("logloss", "dprime", "auroc"):
        v = np.array([r[key] for r in rnd])
        out[f"rand_{key}_mean"] = float(v.mean())
        out[f"rand_{key}_band95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out[f"z_{key}"] = float((real[key] - v.mean()) / max(v.std(ddof=1), 1e-9))
    return out
