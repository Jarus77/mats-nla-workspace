"""J-Lens reader (no decoder): token scores of a Jacobian lens applied to one activation.

Artifact (primary, D3): ``neuronpedia/jacobian-lens``
``qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt``.

Readout of record (WorkspaceBench ``JLens.read``, the pilot's default): the magnitude-free
cosine score  s_t(h) = (W_U J_L h)_t / ||J_L^T W_U[t]||  (no final norm). A "logit" readout
(W_U norm(J_L h), as in 02's leak scripts) is available via ``readout="logit"``.

Paired log-odds analogue (prereg §6.1): LO = max_{t in variants(a)} s_t - max_{t in variants(b)} s_t.
It is a difference of scores, not of log-probs, so it is only comparable to the decoders after
AUROC normalisation. Deterministic and linear, so batch composition is irrelevant.

Memory: only the requested layers are loaded (mmap), J in fp32 (~105 MB/layer) + W_U fp32
(~5 GB). That fits an H100 next to the 27B subject model; the 01 H200 requirement came from
loading all 64 layers plus a full ``W_U @ J`` transient.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from ..jspace import load_jlens_layers

REPO = "neuronpedia/jacobian-lens"
FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
DEFAULT_LAYER = 42


def token_variants(tok: Any, words: Sequence[str]) -> list[int]:
    """Single-token ids for each word as-is, with a leading space, and capitalised."""
    out: set[int] = set()
    for w in words:
        for s in {w, " " + w, w.capitalize(), " " + w.capitalize(), w.lower(), " " + w.lower()}:
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                out.add(e[0])
    return sorted(out)


@dataclass
class JLensReader:
    """J-Lens over a few layers. Build with :func:`load_jlens`."""

    J: dict[int, torch.Tensor]  # {layer: [d, d] fp32 on device}
    W_U: torch.Tensor  # [V, d] fp32 on device
    tok: Any
    final_norm: Any | None = None
    device: str = "cuda"
    _denom: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def _check(self, layer: int) -> torch.Tensor:
        if layer not in self.J:
            raise KeyError(f"J-Lens layer {layer} not loaded (have {sorted(self.J)})")
        return self.J[layer]

    @torch.no_grad()
    def denom(self, layer: int, chunk: int = 32768) -> torch.Tensor:
        """||J^T W_U[t]|| for every token (cached)."""
        if layer not in self._denom:
            J = self._check(layer)
            parts = [(self.W_U[s : s + chunk] @ J).norm(dim=1)
                     for s in range(0, self.W_U.shape[0], chunk)]
            self._denom[layer] = torch.cat(parts).clamp_min(1e-9)
        return self._denom[layer]

    @torch.no_grad()
    def scores(
        self, acts: torch.Tensor, layer: int = DEFAULT_LAYER,
        ids: Sequence[int] | None = None, readout: Literal["cosine", "logit"] = "cosine",
    ) -> torch.Tensor:
        """[N, V] (or [N, len(ids)]) lens scores for activations [N, d] (fp32, CPU out)."""
        J = self._check(layer)
        z = acts.reshape(-1, acts.shape[-1]).to(self.device).float() @ J.T  # J h
        W = self.W_U if ids is None else self.W_U[torch.as_tensor(list(ids), device=self.device)]
        if readout == "cosine":
            d = self.denom(layer)
            d = d if ids is None else d[torch.as_tensor(list(ids), device=self.device)]
            return ((z @ W.T) / d).cpu()
        if self.final_norm is None:
            raise ValueError("logit readout needs final_norm")
        p = next(self.final_norm.parameters())
        return (self.final_norm(z.to(p.dtype)).float() @ W.T).cpu()

    def readout(self, acts: torch.Tensor, layer: int = DEFAULT_LAYER, k: int = 10,
                readout: Literal["cosine", "logit"] = "cosine") -> list[list[tuple[str, float]]]:
        """Top-k (token, score) per activation (the WorkspaceBench jlens readout)."""
        s = self.scores(acts, layer, readout=readout)
        vals, idx = torch.topk(s, k, dim=1)
        return [[(self.tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(vr, ir)]
                for vr, ir in zip(vals, idx)]

    def paired_logodds(
        self, acts: torch.Tensor, layer: int = DEFAULT_LAYER,
        cand_a: Sequence[str] | Sequence[int] = (), cand_b: Sequence[str] | Sequence[int] = (),
        readout: Literal["cosine", "logit"] = "cosine",
    ) -> torch.Tensor:
        """[N] max score over a's token variants minus max over b's (words or token ids)."""
        ia = list(cand_a) if cand_a and isinstance(cand_a[0], int) else token_variants(self.tok, cand_a)
        ib = list(cand_b) if cand_b and isinstance(cand_b[0], int) else token_variants(self.tok, cand_b)
        if not ia or not ib:
            raise ValueError("a candidate has no single-token variant")
        s = self.scores(acts, layer, ids=ia + ib, readout=readout)
        return s[:, : len(ia)].max(1).values - s[:, len(ia):].max(1).values

    def rank_of(self, acts: torch.Tensor, layer: int, ids: Sequence[int]) -> torch.Tensor:
        """[N] best (0-based) rank among ``ids`` in the full readout (presence / G3 checks)."""
        s = self.scores(acts, layer)
        sub = s[:, list(ids)].max(1).values
        return (s > sub[:, None]).sum(1)


def load_jlens(
    model: Any, tok: Any, layers: Sequence[int] = (40, 42, 44), path: str | None = None,
    device: str = "cuda",
) -> JLensReader:
    """Load J for ``layers`` and the subject model's W_U (fp32) onto ``device``."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, FILE)
    J = {L: m.to(device) for L, m in load_jlens_layers(path, list(layers)).items()}
    lm_head = model.get_output_embeddings()
    W_U = lm_head.weight.detach().to(device).float()
    norm = None
    for name, mod in model.named_modules():
        if name.endswith("model.norm") or name.endswith("language_model.norm"):
            norm = mod
            break
    return JLensReader(J=J, W_U=W_U, tok=tok, final_norm=norm, device=device)
