"""Library code that runs inside Modal containers (model loading, hooks, J-lens, pursuit)."""

from __future__ import annotations

import json
import os
from typing import Any

OUT = "/results/02_nonj_features"
HS_LAYERS = [40, 41, 42, 43, 44, 45]
MAIN_HS = 43
ALT_HS = 42


# ----------------------------------------------------------------------------- model


def decoder_blocks(model: Any) -> Any:
    for path in ("model.layers", "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("decoder blocks not found")


def final_norm_module(model: Any) -> Any:
    for path in ("model.norm", "model.language_model.norm"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("final norm not found")


def load_model(max_layers: int | None = None) -> tuple[Any, Any]:
    """Load Qwen3.6-27B bf16. ``max_layers`` truncates the decoder stack (capture-only runs)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    if max_layers is not None:
        blocks = decoder_blocks(model)
        del blocks[max_layers:]
    return tok, model


def norm_gain(model: Any) -> Any:
    """Effective elementwise gain of the final RMSNorm: norm(ones) (rms(ones)=1)."""
    import torch

    nm = final_norm_module(model)
    d = model.config.hidden_size if hasattr(model.config, "hidden_size") else 5120
    p = next(nm.parameters())
    with torch.no_grad():
        g = nm(torch.ones(1, d, device=p.device, dtype=torch.float32).to(p.dtype)).float()[0]
    return g


# ----------------------------------------------------------------------------- J-lens


def load_jlens_layers(path: str, layers: list[int]) -> dict[int, Any]:
    """Return {jlens_layer: J fp32 [d,d]} for the requested J-lens (block-output) layers."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    jac = obj["J"] if isinstance(obj, dict) and "J" in obj else obj["jacobians"]
    out = {}
    for L in layers:
        if isinstance(jac, dict):
            m = jac[L] if L in jac else jac[str(L)]
        else:
            m = jac[L]
        out[L] = m.float().clone()
    return out


def jlens_atoms(J: Any, W_U: Any, gain: Any, chunk: int = 32768) -> tuple[Any, Any]:
    """J-lens vectors (rows of W_U diag(g) J), returned unit-normalised in bf16 plus norms.

    Paper: "We refer to the rows of W_U J_l as the J-lens vectors". The final-norm gain is folded
    in so that <v_t, h> is (up to the RMS factor) exactly the lens logit of token t.
    """
    import torch

    V, d = W_U.shape
    Jg = (gain[:, None] * J).to(torch.float32)  # [d, d]; (W_U * g) @ J == W_U @ diag(g) @ J
    atoms = torch.empty(V, d, dtype=torch.bfloat16, device=J.device)
    norms = torch.empty(V, dtype=torch.float32, device=J.device)
    for s in range(0, V, chunk):
        a = W_U[s : s + chunk].float() @ Jg
        n = a.norm(dim=1).clamp_min(1e-9)
        atoms[s : s + chunk] = (a / n[:, None]).to(torch.bfloat16)
        norms[s : s + chunk] = n
    return atoms, norms


def nn_pursuit(H: Any, atoms: Any, k: int = 25, refine_steps: int = 6) -> tuple[Any, Any, Any]:
    """Batched nonnegative gradient pursuit (greedy atom selection + projected-gradient refits).

    H [B,d] fp32, atoms [V,d] unit-norm bf16. Returns (idx [B,k] long, coef [B,k] fp32,
    recon [B,d] fp32 = J-space component). Mirrors the paper's "sparse nonnegative combination
    of k J-lens vectors ... using gradient pursuit".
    """
    import torch

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
        A = atoms[idx[:, : j + 1]].float()  # [B,j+1,d]
        c = coef[:, : j + 1].clone()
        for _ in range(refine_steps):
            r = H - torch.einsum("bk,bkd->bd", c, A)
            g = torch.einsum("bkd,bd->bk", A, r)  # negative gradient
            g = torch.where((c <= 0) & (g < 0), torch.zeros_like(g), g)
            Ag = torch.einsum("bk,bkd->bd", g, A)
            step = (g * g).sum(1) / (Ag * Ag).sum(1).clamp_min(1e-9)
            c = (c + step[:, None] * g).clamp_min(0)
        coef[:, : j + 1] = c
        R = H - torch.einsum("bk,bkd->bd", c, A)
    recon = H - R
    return idx, coef, recon


def project_out_span(H: Any, atoms: Any, idx: Any) -> Any:
    """h - Pi_S h with Pi_S orthogonal projection onto span of atoms[idx[b]] (paper formalization)."""
    import torch

    A = atoms[idx].float()  # [B,k,d]
    Q, _ = torch.linalg.qr(A.transpose(1, 2))  # [B,d,k]
    coeff = torch.einsum("bdk,bd->bk", Q, H)
    return H - torch.einsum("bdk,bk->bd", Q, coeff)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
