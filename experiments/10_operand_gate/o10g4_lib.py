"""10 operand gate, stage 2a-G4 helpers (pure; CPU-testable): the one-digit few-shot bank, class-mean subspace
swaps (Michael Kenney's construction), random subspaces for the swap null, and the per-condition summaries.

Conventions: directions are unit rows; the centred edit sets the projection on v to ±Δ_i with Δ_i = rel·‖h_i‖
(r11 = 0.11, r22 = 0.22); a subspace swap replaces the item's profile U^T h with a target profile in span(U).
"""

from __future__ import annotations

import numpy as np
import torch

from o10s2_lib import exemplars

REL = {"r11": 0.11, "r22": 0.22}
DIGITS = tuple(range(1, 10))


# ----------------------------------------------------------------------------- bank


def onedigit_items() -> list[dict]:
    """All a, b in 1..9 for "+", and a > b for "-": 81 + 36 = 117 items in Kenney's "a op b =" format."""
    out = []
    for op in ("+", "-"):
        for a in DIGITS:
            for b in DIGITS:
                if op == "-" and a <= b:
                    continue
                out.append({"id": f"od{len(out):03d}", "a": a, "b": b, "op": op, "answer": a + b if op == "+" else a - b,
                            "text": f"{a} {op} {b} ="})
    return out


def fewshot_text_onedigit(item: dict, seed: int = 0) -> str:
    ex = exemplars(item["a"], item["b"], seed=seed)  # two-digit exemplars allowed, operands disjoint from {a, b}
    return "\n".join(f"{x} {op} {y} = {z}" for x, op, y, z in ex) + "\n" + item["text"]


def aprime(a: int) -> int:
    """The swap target digit: a + 3 wrapped inside 1..9 (so a' is never 0 or equal to a)."""
    return ((a + 3 - 1) % 9) + 1


def predicted_answer(a2: int, b: int, op: str) -> int:
    return a2 + b if op == "+" else a2 - b


# ----------------------------------------------------------------------------- subspaces


def orthonormal_span(V: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis U [d, k] of the row space of V [k, d] (QR, sign-fixed)."""
    Q, R = torch.linalg.qr(V.float().T)
    return Q * torch.sign(torch.diagonal(R))[None, :]


def project_out_rows(Q: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return V - (V @ Q) @ Q.T


def swap_vectors(H: torch.Tensor, U: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """h' = h - U U^T h + U target, rows of H [N, d]; target [N, k] (per-item target profile)."""
    Hf, Uf = H.float(), U.float().to(H.device)
    return Hf - (Hf @ Uf) @ Uf.T + target.float().to(H.device) @ Uf.T


def class_profiles(H: torch.Tensor, labels: np.ndarray, U: torch.Tensor, classes: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """Mean profile U^T h per class over the rows of H."""
    P = H.float() @ U.float().to(H.device)
    return {c: P[torch.as_tensor(labels == c, device=H.device)].mean(0) for c in classes}


def random_subspaces(d: int, k: int, n: int, Q: torch.Tensor | None, seed: int) -> torch.Tensor:
    """n random orthonormal k-dim subspaces [n, d, k]; span(Q) projected out first when Q is given (J-complement)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n):
        A = torch.randn(d, k, generator=g)
        if Q is not None:
            Qc = Q.cpu().float()
            A = A - Qc @ (Qc.T @ A)
        out.append(orthonormal_span(A.T))
    return torch.stack(out)


def random_displacements(U: torch.Tensor, norms: torch.Tensor, seed: int) -> torch.Tensor:
    """[N, d] random vectors inside span(U) [d, k] with the given per-row norms (the swap's null edit)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(len(norms), U.shape[1], generator=g)
    r = r / r.norm(dim=1, keepdim=True)
    return (r.to(U.device) @ U.float().T) * norms.to(U.device)[:, None]


# ----------------------------------------------------------------------------- summaries


def _arr(x: list) -> np.ndarray:
    return np.array([v if v is not None else np.nan for v in x], dtype=float)


def flip_rate(ans: dict[str, list], key: str) -> float:
    base, a = _arr(ans["none"]), _arr(ans[key])
    return float(np.mean(np.nan_to_num(a, nan=-1e9) != np.nan_to_num(base, nan=-1e9)))


def banded(stat: float, null_stats: dict[str, list[float]]) -> dict:
    from mats.metrics import null_band

    return {fam: null_band(stat, v) for fam, v in null_stats.items()}


def swap_stats(ans: dict[str, list], key: str, truth: np.ndarray, pred: np.ndarray) -> dict:
    """Did the greedy answer follow the swap? exact-follow rate, moved-toward rate, median normalised shift
    (value - truth) / (pred - truth) (1 = fully followed), plus the flip rate."""
    a = _arr(ans[key])
    ok = np.isfinite(a) & (pred != truth)
    rel = (a[ok] - truth[ok]) / (pred[ok] - truth[ok])
    return {"follow_exact": float(np.mean(a[ok] == pred[ok])) if ok.any() else float("nan"),
            "moved_toward": float(np.mean(rel > 0)) if ok.any() else float("nan"),
            "norm_shift_median": float(np.median(rel)) if ok.any() else float("nan"),
            "norm_shift_mean": float(np.mean(rel)) if ok.any() else float("nan"),
            "flip_rate": flip_rate(ans, key), "n_valid": int(ok.sum())}
