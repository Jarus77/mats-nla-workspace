"""Linear probes for binary features (G1 decodability, G2 J-removal retention, steering direction).

Small, dependency-free (torch only) versions of the probe code in experiments/02 s3_probe.py:
L2-regularised logistic regression by LBFGS on standardised features, grouped K-fold (groups =
passages/items, never split across folds), and the class-mean-difference steering direction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .metrics import auroc


@dataclass(frozen=True)
class LogReg:
    w: torch.Tensor  # [d] in standardised space
    b: float
    mu: torch.Tensor
    sd: torch.Tensor

    def decision(self, X: torch.Tensor) -> torch.Tensor:
        return ((X.float() - self.mu) / self.sd) @ self.w + self.b

    def direction(self) -> torch.Tensor:
        """Unit probe direction in raw activation space (w / sd)."""
        v = self.w / self.sd
        return v / v.norm()


def fit_logreg(X: torch.Tensor, y: torch.Tensor, lam: float = 1e-3, iters: int = 200) -> LogReg:
    X = X.float()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-4)
    Z = (X - mu) / sd
    yt = torch.as_tensor(y, dtype=torch.float32, device=X.device)
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, yt) + lam * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return LogReg(w.detach(), float(b.detach()), mu, sd)


def group_folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """k test-index arrays; each group lands in exactly one fold."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    return [np.flatnonzero(np.isin(groups, part)) for part in np.array_split(ug, k)]


def cv_auroc(
    X: torch.Tensor, y: np.ndarray, groups: np.ndarray, k: int = 5, lam: float = 1e-3,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None, retrain: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Grouped k-fold held-out AUROC. ``transform`` (e.g. J-removal) is applied to the test set
    (frozen probe: retrain=False trains on raw X) or to both sets (retrain=True)."""
    y = np.asarray(y)
    scores = np.empty(len(y))
    for te in group_folds(groups, k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        Xtr = X[tr] if (transform is None or not retrain) else transform(X[tr])
        m = fit_logreg(Xtr, torch.as_tensor(y[tr]), lam)
        Xte = X[te] if transform is None else transform(X[te])
        scores[te] = m.decision(Xte).cpu().numpy()
    return {"auroc": auroc(scores[y == 1], scores[y == 0]), "n": len(y)}


def mean_diff_direction(X: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    """Unit (mean[y=1] - mean[y=0]) in raw space (the 02 s9 steering direction for classes)."""
    yb = torch.as_tensor(np.asarray(y).astype(bool), device=X.device)
    v = X[yb].float().mean(0) - X[~yb].float().mean(0)
    return v / v.norm()
