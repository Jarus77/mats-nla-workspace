"""Oracle Lens (OL) reader: a LoRA on the subject model that verbalizes one activation.

Artifact: ``agu18dec/olens_and_ar:olens_s3d_rl600`` (byte-identical to the WorkspaceBench
default; verified in experiments/01). Contract (experiments/00_nla_gate RESULTS, run.json):
- trained on block-output activations at layers 20..60 step 4; alpha 16000, transform "unit";
- carrier = one chat-templated user turn (``add_generation_prompt=True``,
  ``enable_thinking=False``) with the layer as a plain integer in the prompt;
- injection REPLACES the input embedding of the slot character (an enclosed ideograph that
  survives as one token; 00 found U+321C, id 158983) with ``alpha * h / ||h||``;
- native output = '- ' bullet lines, one concept per line;
- the same model serves as subject: capture with the adapter DISABLED (:meth:`base`).

Layer policy (D7, prereg §5): **L42 is the PRIMARY read layer for OL too** (prompt says
"layer 42"; off-grid but D6 shows it is no worse than 40/44). 40/44 are sensitivity only. The
natural-item layer sweep uses ``SWEEP_LAYERS`` (20..60 step 4 plus 42); every metric is
computed per layer -- never select the best layer per item (any-layer selection is biased).

Usage (H100 is enough: one 27B copy + LoRA):
    ol = load_olens()                         # loads Qwen3.6-27B + LoRA
    with ol.base():                           # adapter off -> subject model
        acts = capture(ol.model, seqs, [42], positions)
    texts = ol.verbalize(acts[42][:, 0], layer=42, k=1)
    lo = ol.paired_logodds(acts[42][:, 0], 42, "- French language", "- German language")
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .nla import Sampling

REPO, SUBDIR = "agu18dec/olens_and_ar", "olens_s3d_rl600"
BASE_MODEL = "Qwen/Qwen3.6-27B"
ALPHA = 16000.0
TRAINED_LAYERS = tuple(range(20, 61, 4))
DEFAULT_LAYER = 42
SENSITIVITY_LAYERS = (40, 44)
SWEEP_LAYERS = tuple(sorted({*TRAINED_LAYERS, 42}))
PROMPT = (
    "An activation vector from layer {layer} of a language model is enclosed in activation "
    "tags: <activation>{char}</activation>. Produce distinct concepts that encode this "
    "activation, each as a '- ' bullet on its own line."
)
WSBENCH_SAMPLING = Sampling(temperature=1.0, top_p=0.95, top_k=64, max_new_tokens=256)


@dataclass
class OracleLens:
    """Loaded OL (subject model + LoRA). Build with :func:`load_olens`."""

    model: Any  # PeftModel
    tok: Any
    alpha: float = ALPHA
    prompt: str = PROMPT
    device: str = "cuda"
    _slots: dict[int, tuple[list[int], int]] = field(default_factory=dict, repr=False)

    # -- roles --------------------------------------------------------------------------
    def base(self) -> Any:
        """Context manager: adapter off, i.e. the plain subject model (for capture)."""
        return self.model.disable_adapter()

    # -- carrier ------------------------------------------------------------------------
    def carrier(self, layer: int) -> tuple[list[int], int]:
        """Carrier ids and slot index (WorkspaceBench ``OLens._carrier`` algorithm)."""
        if layer in self._slots:
            return self._slots[layer]
        if layer not in TRAINED_LAYERS and layer != DEFAULT_LAYER:
            warnings.warn(f"OL layer {layer} is off-grid (trained {TRAINED_LAYERS})",
                          stacklevel=2)
        for code in range(0x3200, 0x3400):
            char = chr(code)
            ids = self.tok(char, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                continue
            out = self.tok.apply_chat_template(
                [{"role": "user", "content": self.prompt.format(layer=layer, char=char)}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            rendered = list(out["input_ids"] if hasattr(out, "keys") else out)
            slots = [i for i, t in enumerate(rendered) if t == ids[0]]
            if len(slots) == 1:
                self._slots[layer] = (rendered, slots[0])
                return self._slots[layer]
        raise ValueError("no enclosed ideograph survives as a single token in the carrier")

    def _embed(self, ids: torch.Tensor, slot: int, acts: torch.Tensor) -> torch.Tensor:
        """Input embeddings with row ``slot`` replaced by alpha * unit(h) (one act per row)."""
        e = self.model.get_input_embeddings()(ids).clone()
        v = acts.to(self.device).float()
        e[:, slot, :] = (self.alpha * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)).to(e.dtype)
        return e

    # -- verbalize ----------------------------------------------------------------------
    @torch.no_grad()
    def verbalize(
        self, acts: torch.Tensor, layer: int = DEFAULT_LAYER, k: int = 1, sampling: Sampling = WSBENCH_SAMPLING,
        batch: int = 64,
    ) -> list[list[str]]:
        """k free-form readouts per activation (batched across rows; no padding needed)."""
        acts = acts.reshape(-1, acts.shape[-1])
        ids0, slot = self.carrier(layer)
        rows = acts.repeat_interleave(k, dim=0)
        texts: list[str] = []
        for s in range(0, rows.shape[0], batch):
            v = rows[s : s + batch]
            ids = torch.tensor([ids0] * v.shape[0], device=self.device)
            e = self._embed(ids, slot, v)
            greedy = sampling.temperature <= 0
            kw = {} if greedy else {"temperature": sampling.temperature, "top_p": sampling.top_p,
                                    "top_k": sampling.top_k}
            out = self.model.generate(
                inputs_embeds=e, attention_mask=torch.ones_like(ids), do_sample=not greedy,
                max_new_tokens=sampling.max_new_tokens,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id, **kw,
            )
            texts += [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]
        return [texts[i * k : (i + 1) * k] for i in range(acts.shape[0])]

    # -- teacher-forced scoring ---------------------------------------------------------
    @torch.no_grad()
    def logprob_of_strings(
        self, acts: torch.Tensor, layer: int = DEFAULT_LAYER, candidates: Sequence[str] = (), prefix: str = "",
        batch: int = 32,
    ) -> torch.Tensor:
        """[N, C] summed log p(candidate tokens | carrier with act injected, prefix).

        All C candidates of one activation always share a forward batch (batch composition
        shifts bf16 scores by up to 0.26 nats; 00 RESULTS)."""
        acts = acts.reshape(-1, acts.shape[-1])
        ids0, slot = self.carrier(layer)
        head = ids0 + (self.tok.encode(prefix, add_special_tokens=False) if prefix else [])
        cands = [self.tok.encode(c, add_special_tokens=False) for c in candidates]
        C = len(cands)
        per = max(1, batch // C)  # activations per forward
        pad = self.tok.pad_token_id or 0
        out = torch.empty(acts.shape[0], C)
        for s in range(0, acts.shape[0], per):
            a = acts[s : s + per]
            rows = [(i, c) for i in range(a.shape[0]) for c in range(C)]
            n = len(head) + max(len(c) for c in cands)
            ids = torch.full((len(rows), n), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, (_, c) in enumerate(rows):
                seq = head + cands[c]
                ids[r, : len(seq)] = torch.tensor(seq)
                mask[r, : len(seq)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            e = self._embed(ids, slot, a.repeat_interleave(C, dim=0))
            keep = n - len(head) + 1
            logits = self.model(inputs_embeds=e, attention_mask=mask, logits_to_keep=keep).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tok_lp = lp.gather(-1, ids[:, len(head):].unsqueeze(-1)).squeeze(-1)
            for r, (i, c) in enumerate(rows):
                out[s + i, c] = float(tok_lp[r, : len(cands[c])].sum())
        return out

    def paired_logodds(
        self, acts: torch.Tensor, layer: int = DEFAULT_LAYER, cand_a: str = "", cand_b: str = "",
        prefix: str = "",
    ) -> torch.Tensor:
        """[N] log p(a | act) - log p(b | act), native bullet format (e.g. "- French")."""
        lp = self.logprob_of_strings(acts, layer, [cand_a, cand_b], prefix=prefix)
        return lp[:, 0] - lp[:, 1]

    def paired_logodds_templates(
        self, acts: torch.Tensor, layer: int, pairs: Sequence[tuple[str, str]], prefix: str = ""
    ) -> torch.Tensor:
        """[T, N] LO for each (a, b) template pair (average over T for the prereg's 3-template
        mean; report the spread)."""
        return torch.stack([self.paired_logodds(acts, layer, a, b, prefix) for a, b in pairs])

    def layer_sweep_logodds(
        self, acts_by_layer: dict[int, torch.Tensor], pairs: Sequence[tuple[str, str]],
        prefix: str = "",
    ) -> dict[int, torch.Tensor]:
        """{layer: [T, N] LO} for natural items over a layer sweep (``acts_by_layer[L]`` = the
        block-L captures, e.g. for L in SWEEP_LAYERS). Score each layer separately; do NOT
        take a per-item max over layers."""
        return {L: self.paired_logodds_templates(a, L, pairs, prefix)
                for L, a in sorted(acts_by_layer.items())}


def load_olens(
    model: Any | None = None, tok: Any | None = None, lora_dir: str | None = None,
    device: str = "cuda",
) -> OracleLens:
    """Attach the OL LoRA to ``model`` (loads Qwen3.6-27B bf16 if None)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if lora_dir is None:
        from huggingface_hub import snapshot_download

        lora_dir = f"{snapshot_download(REPO, allow_patterns=[f'{SUBDIR}/*'])}/{SUBDIR}"
    tok = tok or AutoTokenizer.from_pretrained(BASE_MODEL)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                     device_map=device)
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    return OracleLens(model=model, tok=tok, device=device)
