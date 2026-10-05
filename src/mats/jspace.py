"""J-space (prereg §4.1): basis, projection, enrichment, random equal-rank controls.

Ported from experiments/02_nonj_features (READ ONLY there):
- ``jspace_gram`` / ``jspace_basis``: s8_rstar.py ``rstar()`` -- J-space at rank r = span of the
  top-r right singular vectors V_r of the lens map M_l = W_U diag(g) J_l (neuronpedia n1000
  J-lens; g = effective gain of the final RMSNorm), via eigh of M^T M accumulated over W_U
  chunks. The 02 agent saved the result for L38/L42 as
  /results/02_nonj_features/jspace_basis_L{38,42}.pt (``evecs_top4096``), which
  :func:`load_basis_file` reads.
- ``load_jlens_layers``, ``norm_gain``: nonj_lib.py (mmap, only the requested layers).
- ``jlens_atoms``, ``nn_pursuit``, ``project_out_span``: nonj_lib.py (Anthropic-style per-
  activation sparse nonnegative J-lens decomposition, k=25; the prereg's robustness definition).

Layer convention: J-lens layer L = output of decoder block L = HF hidden_states[L+1].
r* (02 s8_rstar, 90% top-10 readout agreement on basic_readout): 384 at L40/41/42, 256 at L44,
192 at L38.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

D_MODEL = 5120


# ----------------------------------------------------------------------------- basis


@dataclass(frozen=True)
class JBasis:
    """Orthonormal J-space basis: ``vecs[:, :r]`` spans S_r (columns sorted by singular value)."""

    vecs: torch.Tensor  # [d, r_max] fp32
    evals: torch.Tensor  # [r_max] eigenvalues of M^T M (= squared singular values of M)
    layer: int | None = None
    source: str = ""

    @property
    def d(self) -> int:
        return int(self.vecs.shape[0])

    @property
    def r_max(self) -> int:
        return int(self.vecs.shape[1])

    def Q(self, r: int) -> torch.Tensor:  # noqa: N802 (math notation)
        if r > self.r_max or r < 0:
            raise ValueError(f"rank {r} outside the stored basis (r_max {self.r_max})")
        return self.vecs[:, :r]

    def to(self, device: Any) -> JBasis:
        return JBasis(self.vecs.to(device), self.evals.to(device), self.layer, self.source)


@torch.no_grad()
def jspace_gram(
    J: torch.Tensor, W_U: torch.Tensor, gain: torch.Tensor | None = None, chunk: int = 32768
) -> torch.Tensor:
    """M^T M for M = W_U diag(g) J, accumulated over vocab chunks (fp32). [d, d]."""
    Jg = (J.float() if gain is None else gain.float()[:, None].to(J.device) * J.float())
    G = torch.zeros(J.shape[1], J.shape[1], device=J.device, dtype=torch.float32)
    for s in range(0, W_U.shape[0], chunk):
        a = W_U[s : s + chunk].to(J.device).float() @ Jg
        G += a.T @ a
    return G


@torch.no_grad()
def jspace_basis(
    J: torch.Tensor,
    W_U: torch.Tensor,
    gain: torch.Tensor | None = None,
    r_max: int = 4096,
    chunk: int = 32768,
    layer: int | None = None,
) -> JBasis:
    """Top-``r_max`` right singular vectors of M = W_U diag(g) J (02 s8_rstar definition)."""
    G = jspace_gram(J, W_U, gain, chunk)
    evals, evecs = torch.linalg.eigh(G.double() if G.device.type == "cpu" else G)
    evals, evecs = evals.flip(0).float(), evecs.flip(1).float()
    return JBasis(evecs[:, :r_max].contiguous(), evals[:r_max].contiguous(), layer,
                  "top right singular vectors of W_U diag(g) J_l")


def load_basis_file(path: str, device: str = "cpu") -> JBasis:
    """Read a basis saved by 02 s8_rstar.py or by :func:`save_basis`."""
    obj = torch.load(path, map_location=device, weights_only=False)
    vecs = obj.get("evecs_top4096", obj.get("vecs"))
    evals = obj.get("evals_top4096", obj.get("evals"))
    return JBasis(vecs.float(), evals.float(), obj.get("layer_jlens", obj.get("layer")),
                  obj.get("def", obj.get("source", path)))


def save_basis(basis: JBasis, path: str) -> None:
    torch.save({"vecs": basis.vecs.half().cpu(), "evals": basis.evals.cpu(), "layer": basis.layer,
                "source": basis.source}, path)


# ----------------------------------------------------------------------------- projection


def _proj_coeffs(Q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return v.float() @ Q.to(v.device).float()


def project_onto_of(Q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """P_S v for an orthonormal basis Q [d, r]; v is [d] or [N, d]."""
    return _proj_coeffs(Q, v) @ Q.to(v.device).float().T


def project_out_of(Q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(I - P_S) v for an orthonormal basis Q [d, r]."""
    return v.float() - project_onto_of(Q, v)


def enrichment_in(Q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(||P_S v||^2 / ||v||^2) / (r / d). A random vector scores 1.0 in expectation."""
    d, r = Q.shape
    vf = v.float()
    frac = (_proj_coeffs(Q, vf) ** 2).sum(-1) / (vf**2).sum(-1).clamp_min(1e-30)
    return frac / (r / d)


def project_onto(basis: JBasis, v: torch.Tensor, r: int) -> torch.Tensor:
    return project_onto_of(basis.Q(r), v)


def project_out(basis: JBasis, v: torch.Tensor, r: int) -> torch.Tensor:
    """v - P_{S_r} v (the "non-J component" at rank r)."""
    return project_out_of(basis.Q(r), v)


def enrichment(basis: JBasis, v: torch.Tensor, r: int) -> torch.Tensor | float:
    e = enrichment_in(basis.Q(r), v)
    return float(e) if e.ndim == 0 else e


def j_removed_direction(basis: JBasis, w: torch.Tensor, r: int) -> torch.Tensor:
    """u = normalise((I - P_{S_r}) w): the prereg §4.2 steering direction for non-J features."""
    u = project_out(basis, w, r)
    n = u.norm()
    if float(n) < 1e-8:
        raise ValueError("direction lies entirely inside S_r")
    return u / n


# ----------------------------------------------------------------------------- controls


def random_subspace(d: int, r: int, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """Haar-random orthonormal basis [d, r] (the equal-rank random-subspace control)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, r, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]
    return Q.float().to(device)


def random_direction(
    basis: JBasis, r: int, kind: Literal["nonj", "j", "iso"] = "nonj", seed: int = 0
) -> torch.Tensor:
    """Unit random direction for equal-norm steering controls (prereg §4.4):
    "nonj" = random vector with S_r projected out, "j" = random vector inside S_r,
    "iso" = isotropic."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(basis.d, generator=g).to(basis.vecs.device)
    if kind == "nonj":
        v = project_out(basis, v, r)
    elif kind == "j":
        v = project_onto(basis, v, r)
    elif kind != "iso":
        raise ValueError(f"unknown kind {kind!r}")
    return v / v.norm()


# ----------------------------------------------------------------------------- J-lens I/O


def load_jlens_layers(path: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """{J-lens layer: J fp32 [d, d]} for only the requested layers (mmap; nonj_lib port)."""
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    jac = obj["J"] if isinstance(obj, dict) and "J" in obj else obj["jacobians"]
    out = {}
    for L in layers:
        m = (jac[L] if L in jac else jac[str(L)]) if isinstance(jac, dict) else jac[L]
        out[L] = m.float().clone()
    return out


@torch.no_grad()
def norm_gain(final_norm: Any, d: int = D_MODEL) -> torch.Tensor:
    """Effective elementwise gain of the final RMSNorm: norm(ones) (rms(ones) = 1)."""
    p = next(final_norm.parameters())
    return final_norm(torch.ones(1, d, device=p.device, dtype=p.dtype)).float()[0]


# ----------------------------------------------------------------------------- NNLS (robustness)


@torch.no_grad()
def jlens_atoms(
    J: torch.Tensor, W_U: torch.Tensor, gain: torch.Tensor, chunk: int = 32768
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-normalised J-lens vectors (rows of W_U diag(g) J) in bf16, plus their norms."""
    V, d = W_U.shape
    Jg = (gain[:, None] * J).to(torch.float32)
    atoms = torch.empty(V, d, dtype=torch.bfloat16, device=J.device)
    norms = torch.empty(V, dtype=torch.float32, device=J.device)
    for s in range(0, V, chunk):
        a = W_U[s : s + chunk].float() @ Jg
        n = a.norm(dim=1).clamp_min(1e-9)
        atoms[s : s + chunk] = (a / n[:, None]).to(torch.bfloat16)
        norms[s : s + chunk] = n
    return atoms, norms


@torch.no_grad()
def nn_pursuit(
    H: torch.Tensor, atoms: torch.Tensor, k: int = 25, refine_steps: int = 6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched nonnegative gradient pursuit (greedy atom selection + projected-gradient refits).

    H [B, d] fp32, atoms [V, d] unit-norm bf16. Returns (idx [B, k], coef [B, k],
    recon [B, d] = the J-space component). Faithful port of nonj_lib.nn_pursuit.
    """
    B, d = H.shape
    idx = torch.zeros(B, k, dtype=torch.long, device=H.device)
    coef = torch.zeros(B, k, dtype=torch.float32, device=H.device)
    R = H.clone()
    mask = torch.zeros(B, atoms.shape[0], dtype=torch.bool, device=H.device)
    for j in range(k):
        corr = (R.to(torch.bfloat16) @ atoms.T).float()
        corr.masked_fill_(mask, -1e30)
        best = corr.argmax(dim=1)
        idx[:, j] = best
        mask[torch.arange(B, device=H.device), best] = True
        A = atoms[idx[:, : j + 1]].float()
        c = coef[:, : j + 1].clone()
        for _ in range(refine_steps):
            r = H - torch.einsum("bk,bkd->bd", c, A)
            g = torch.einsum("bkd,bd->bk", A, r)
            g = torch.where((c <= 0) & (g < 0), torch.zeros_like(g), g)
            Ag = torch.einsum("bk,bkd->bd", g, A)
            step = (g * g).sum(1) / (Ag * Ag).sum(1).clamp_min(1e-9)
            c = (c + step[:, None] * g).clamp_min(0)
        coef[:, : j + 1] = c
        R = H - torch.einsum("bk,bkd->bd", c, A)
    return idx, coef, H - R


@torch.no_grad()
def project_out_span(H: torch.Tensor, atoms: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """h - Pi_S h, Pi_S = orthogonal projection onto span(atoms[idx[b]]) per row."""
    A = atoms[idx].float()
    Q, _ = torch.linalg.qr(A.transpose(1, 2))
    coeff = torch.einsum("bdk,bd->bk", Q, H.float())
    return H.float() - torch.einsum("bdk,bk->bd", Q, coeff)
