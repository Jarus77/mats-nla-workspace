"""Natural Language Autoencoder (NLA) reader for Qwen3.6-27B, layer 42.

Repo: ``ceselder/qwen3.6-27b-nla-rl`` (EasyNLA, GRPO). Contract, verified in
experiments/00_nla_gate:

- Activation = residual stream at the OUTPUT of decoder block 42 (``model.layers[42]``,
  HF ``output_hidden_states[43]``), raw (no normalisation), d_model 5120.
- AV (verbalizer) = ``av_base`` (full 27B, warm-start LoRA merged) + an RL LoRA adapter.
  Prompt = ``nla_meta.yaml`` actor template through av_base's chat template
  (``enable_thinking`` configurable; see RESULTS.md for which one reproduces FVE).
  Injection = norm-matched ADD at the output of AV block 1, at the marker token
  (char U+321C, id 158983, neighbours 29/510): ``h <- h + ||h|| * v/||v||``.
- AR (reconstructor) = 43-block truncated Qwen3.5 backbone (final norm and lm_head stripped)
  + fp32 ``value_head``. Input = ``"Summary of the following text: <text>{e}</text> <summary>"``
  (no chat template, no special tokens); pred = value_head(sqrt(d) * unit(h_last)).
- FVE (EasyNLA "paper" definition): vectors are rescaled to norm sqrt(d) ("mse_scale");
  mse_i = mean_j (unit_pred_ij - unit_gold_ij)^2; baseline = mean_ij (unit_gold_ij - mu_j)^2
  with mu = mean of the rescaled gold set; FVE = 1 - mean(mse) / baseline.

Usage (GPU, ~90 GB for AV + AR, so H200 or reader-only):
    nla = load_nla(adapter="av_rl_adapters/iter_000400")
    texts = nla.verbalize(acts, k=4)                 # list[list[str | None]]
    pred = nla.reconstruct(["..."])                  # [N, 5120] fp32
    out = nla.fve_on(acts, k=4)                      # per-example FVE, explanations
    lp = nla.logprob_of_strings(act, ["cand a", "cand b"])
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO = "ceselder/qwen3.6-27b-nla-rl"
DEFAULT_ADAPTER = "av_rl_adapters/iter_000400"
D_MODEL = 5120
MSE_SCALE = math.sqrt(D_MODEL)
EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)
EXPLANATION_PREFIX = "<explanation>\n"


@dataclass(frozen=True)
class Sampling:
    """AV sampling. Defaults = the NLA training eval (T=1, no top-p/top-k, 256 tokens)."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 256


def snapshot_dir(repo: str = REPO, hf_home: str = "/weights/hf") -> Path:
    """Local snapshot of a (partially downloaded) HF repo, without touching the network."""
    dirs = list(Path(hf_home, "hub").glob(f"models--{repo.replace('/', '--')}/snapshots/*"))
    if len(dirs) != 1:
        raise FileNotFoundError(f"expected one snapshot of {repo} under {hf_home}, got {dirs}")
    return dirs[0]


# ----------------------------------------------------------------------------- pure math


def rescale(v: torch.Tensor, scale: float = MSE_SCALE) -> torch.Tensor:
    """Rows rescaled to L2 norm ``scale`` (EasyNLA ``normalize_activation``)."""
    v = v.float()
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-12) / scale)


def predict_mean_baseline(gold: torch.Tensor, scale: float = MSE_SCALE) -> float:
    """Per-element variance of the rescaled set around its mean (FVE denominator)."""
    g = rescale(gold, scale)
    return float(((g - g.mean(dim=0, keepdim=True)) ** 2).mean())


def recon_mse(pred: torch.Tensor, gold: torch.Tensor, scale: float = MSE_SCALE) -> torch.Tensor:
    """Per-row MSE between rescaled prediction and rescaled gold (= -reward in training)."""
    return ((rescale(pred, scale) - rescale(gold, scale).to(pred.device)) ** 2).mean(dim=-1)


def fve(pred: torch.Tensor, gold: torch.Tensor, baseline: float) -> torch.Tensor:
    """Per-row FVE = 1 - mse_i / baseline. Average rows for the headline number."""
    return 1.0 - recon_mse(pred, gold) / baseline


def extract_explanation(text: str) -> str | None:
    """Payload between <explanation> tags, stripped; None if the tags are missing."""
    m = EXPLANATION_RE.search(text)
    return m.group(1).strip() if m else None


# ----------------------------------------------------------------------------- model


@dataclass
class NLA:
    """Loaded AV (+ optional AR). Build with :func:`load_nla`."""

    av: Any
    tok: Any
    ar: Any | None
    value_head: Any | None
    meta: dict
    prompt_ids: list[int]
    marker: int
    device: str
    _vec: dict = field(default_factory=dict, repr=False)

    # -- injection ------------------------------------------------------------------
    def _hook(self, _mod: Any, _inp: Any, out: Any) -> Any:
        h = out[0] if isinstance(out, tuple) else out
        v = self._vec.get("v")
        if v is None or h.shape[1] <= self.marker:  # decode steps with KV cache
            return out
        if v.shape[0] != h.shape[0]:
            raise RuntimeError(f"{v.shape[0]} vectors for a batch of {h.shape[0]}")
        hp = h[:, self.marker, :]
        vu = v.to(h.device, torch.float32)
        vu = (vu / (vu.norm(dim=-1, keepdim=True) + 1e-9)).to(h.dtype)
        h[:, self.marker, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
        return out

    @contextmanager
    def _inject(self, vecs: torch.Tensor) -> Iterator[None]:
        self._vec["v"] = vecs
        try:
            yield
        finally:
            self._vec["v"] = None

    def set_prompt(self, enable_thinking: bool) -> None:
        """Re-render the AV prompt with/without the empty think block (marker index unchanged)."""
        ids = _render_prompt(self.tok, self.meta, enable_thinking)[0]
        if ids[self.marker] != self.meta["tokens"]["injection_token_id"]:
            raise RuntimeError("marker moved when re-rendering the prompt")
        self.prompt_ids = ids

    def set_adapter(self, name: str) -> None:
        """Switch the active AV adapter (name = last path component, e.g. "iter_000300")."""
        self.av.set_adapter(name)

    def no_adapter(self) -> Any:
        """Context manager: run av_base alone (pre-RL warm-start AV)."""
        return self.av.disable_adapter()

    # -- verbalize --------------------------------------------------------------------
    @torch.no_grad()
    def verbalize_raw(
        self, acts: torch.Tensor, k: int = 1, sampling: Sampling = Sampling(), batch: int = 64
    ) -> list[list[str]]:
        """Raw AV completions (special tokens stripped), k per activation."""
        acts = acts.reshape(-1, acts.shape[-1]).float()
        rows = acts.repeat_interleave(k, dim=0)
        texts: list[str] = []
        for s in range(0, rows.shape[0], batch):
            v = rows[s : s + batch].to(self.device)
            ids = torch.tensor([self.prompt_ids] * v.shape[0], device=self.device)
            greedy = sampling.temperature <= 0
            kw = (
                {}
                if greedy
                else {
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                    "top_k": sampling.top_k,
                }
            )
            with self._inject(v):
                out = self.av.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=not greedy,
                    max_new_tokens=sampling.max_new_tokens,
                    pad_token_id=self.tok.pad_token_id,
                    **kw,
                )
            texts += self.tok.batch_decode(out[:, len(self.prompt_ids) :], skip_special_tokens=True)
        return [texts[i * k : (i + 1) * k] for i in range(acts.shape[0])]

    def verbalize(
        self, acts: torch.Tensor, k: int = 1, sampling: Sampling = Sampling(), batch: int = 64
    ) -> list[list[str | None]]:
        """Explanations (payload of <explanation> tags; None when the tag never closed)."""
        raw = self.verbalize_raw(acts, k, sampling, batch)
        return [[extract_explanation(t) for t in row] for row in raw]

    # -- reconstruct ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, texts: Sequence[str], batch: int = 16) -> torch.Tensor:
        """AR prediction [N, d] (fp32, raw value-head output) for explanation strings."""
        if self.ar is None:
            raise RuntimeError("load_nla(load_ar=False): no reconstructor loaded")
        tmpl = self.meta["prompt_templates"]["critic"]
        enc = [self.tok.encode(tmpl.format(explanation=t), add_special_tokens=False) for t in texts]
        preds = []
        pad = self.tok.pad_token_id
        for s in range(0, len(enc), batch):
            chunk = enc[s : s + batch]
            n = max(map(len, chunk))
            ids = torch.full((len(chunk), n), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), n), dtype=torch.long)
            for r, e in enumerate(chunk):
                ids[r, : len(e)] = torch.tensor(e)
                mask[r, : len(e)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            h = self.ar(input_ids=ids, attention_mask=mask).last_hidden_state
            last = h[torch.arange(len(chunk)), mask.sum(1) - 1].float()
            preds.append(self.value_head(rescale(last)).float())
        return torch.cat(preds).cpu()

    def fve_on(
        self,
        acts: torch.Tensor,
        k: int = 4,
        sampling: Sampling = Sampling(),
        baseline: float | None = None,
        explanations: list[list[str | None]] | None = None,
    ) -> dict:
        """Verbalize -> reconstruct -> FVE for arbitrary (natural or steered) activations.

        ``baseline`` defaults to the predict-the-mean variance of ``acts`` itself; pass the
        natural-set baseline when scoring steered vectors so the numbers are comparable.
        Failed extractions are excluded (as in NLA training eval) and counted.
        Returns per-example mean FVE over valid samples (NaN if none) plus all samples.
        """
        acts = acts.float()
        base = predict_mean_baseline(acts) if baseline is None else baseline
        expl = explanations if explanations is not None else self.verbalize(acts, k, sampling)
        flat = [(i, j, e) for i, row in enumerate(expl) for j, e in enumerate(row) if e]
        sample_fve = torch.full((acts.shape[0], len(expl[0])), float("nan"))
        if flat:
            pred = self.reconstruct([e for _, _, e in flat])
            gold = acts[[i for i, _, _ in flat]]
            for (i, j, _), f in zip(flat, fve(pred, gold, base).tolist()):
                sample_fve[i, j] = f
        return {
            "baseline": base,
            "per_example": torch.nanmean(sample_fve, dim=1),
            "per_sample": sample_fve,
            "explanations": expl,
            "n_failed": sum(e is None for row in expl for e in row),
        }

    # -- teacher-forced scoring -------------------------------------------------------
    @torch.no_grad()
    def logprob_of_strings(
        self,
        act: torch.Tensor,
        candidates: Sequence[str],
        prefix: str = EXPLANATION_PREFIX,
        batch: int = 16,
    ) -> torch.Tensor:
        """Sum log p(candidate tokens | AV prompt with ``act`` injected, ``prefix``).

        The candidate is scored as the start of the AV's answer, right after ``prefix``
        (default ``"<explanation>\\n"``, the AV's native opening). Returns [len(candidates)].
        """
        head = self.prompt_ids + self.tok.encode(prefix, add_special_tokens=False)
        cands = [self.tok.encode(c, add_special_tokens=False) for c in candidates]
        out = []
        for s in range(0, len(cands), batch):
            chunk = cands[s : s + batch]
            n = len(head) + max(map(len, chunk))
            ids = torch.full((len(chunk), n), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, c in enumerate(chunk):
                ids[r, : len(head) + len(c)] = torch.tensor(head + c)
                mask[r, : len(head) + len(c)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            v = act.reshape(1, -1).float().expand(len(chunk), -1).to(self.device)
            keep = n - len(head) + 1  # logits from the last head token on (predict candidates)
            with self._inject(v):
                logits = self.av(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)  # kept u predicts cand token u
            tgt = ids[:, len(head) :]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for r, c in enumerate(chunk):
                out.append(float(tok_lp[r, : len(c)].sum()))
        return torch.tensor(out)

    def paired_logodds(
        self, act: torch.Tensor, cand_a: str, cand_b: str, prefix: str = EXPLANATION_PREFIX
    ) -> float:
        """log p(a | act) - log p(b | act). Use a/b that differ ONLY in the feature value."""
        lp = self.logprob_of_strings(act, [cand_a, cand_b], prefix=prefix)
        return float(lp[0] - lp[1])


# ----------------------------------------------------------------------------- loading


def _load_ar(root: Path, device: str) -> tuple[Any, Any]:
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    ar_dir = root / "ar_reconstructor"
    full, info = AutoModelForCausalLM.from_pretrained(
        str(ar_dir), dtype=torch.bfloat16, device_map=device, output_loading_info=True
    )
    # final norm and lm_head are absent from the checkpoint by design (both stripped below)
    missing = sorted(set(info["missing_keys"]) - {"lm_head.weight", "model.norm.weight"})
    unexpected = sorted(info["unexpected_keys"])
    if missing or unexpected:
        raise RuntimeError(f"AR weights mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    inner = full.model
    inner.norm = torch.nn.Identity()  # value head reads the raw block-42 residual
    full.lm_head = torch.nn.Identity()
    inner.eval()
    sd = load_file(str(ar_dir / "value_head.safetensors"))
    (w,) = [t for name, t in sd.items() if name.endswith("weight")]
    head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.float32)
    head.weight.data.copy_(w.float())
    return inner, head.to(device).eval()


def _render_prompt(tok: Any, meta: dict, enable_thinking: bool) -> tuple[list[int], int]:
    """AV prompt token ids and the marker index (checked against nla_meta neighbours)."""
    tm = meta["tokens"]
    msg = meta["prompt_templates"]["actor"].format(injection_char=tm["injection_char"])
    text = tok.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    ids = tok.encode(text, add_special_tokens=False)
    pos = [i for i, t in enumerate(ids) if t == tm["injection_token_id"]]
    if len(pos) != 1:
        raise RuntimeError(f"marker must appear once in the AV prompt, found {len(pos)}")
    m = pos[0]
    if (ids[m - 1], ids[m + 1]) != (tm["injection_left_neighbor_id"], tm["injection_right_neighbor_id"]):
        raise RuntimeError(f"marker neighbours {ids[m - 1]}, {ids[m + 1]} differ from nla_meta")
    return ids, m


def load_nla(
    adapter: str | Sequence[str] | None = DEFAULT_ADAPTER,
    load_ar: bool = True,
    device: str = "cuda",
    enable_thinking: bool = False,
    root: str | Path | None = None,
    trainable: bool = False,
) -> NLA:
    """Load av_base (+ RL adapter(s)) and optionally the AR.

    ``adapter``: one path under the repo (e.g. ``"av_rl_adapters/iter_000300"``), a list
    (all loaded, the first active; switch with ``nla.set_adapter("iter_000300")``, i.e. the
    last path component), or None for av_base alone (the pre-RL warm-start verbalizer).
    ``trainable`` (added 28 Oct 2026 for exp 30): see :func:`make_trainable`; the default keeps
    the inference loader exactly as before.
    """
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rootp = Path(root) if root else snapshot_dir()
    base_dir = rootp / "av_base"
    meta = yaml.safe_load((rootp / "nla_meta.yaml").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    ids, m = _render_prompt(tok, meta, enable_thinking)

    av = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map=device)
    adapters = [adapter] if isinstance(adapter, str) else list(adapter or [])
    if adapters:
        from peft import PeftModel

        av = PeftModel.from_pretrained(
            av, str(rootp / adapters[0]), adapter_name=Path(adapters[0]).name
        )
        for a in adapters[1:]:
            av.load_adapter(str(rootp / a), adapter_name=Path(a).name)
        av.set_adapter(Path(adapters[0]).name)
    av.eval()
    (block1,) = [mod for n, mod in av.named_modules() if n.endswith("model.layers.1")]

    ar, head = _load_ar(rootp, device) if load_ar else (None, None)
    nla = NLA(av=av, tok=tok, ar=ar, value_head=head, meta=meta, prompt_ids=ids, marker=m,
              device=device)
    if trainable:  # added 28 Oct 2026 for exp 30: the cloning hook and LoRA grads instead of _hook
        return make_trainable(nla, block1)
    block1.register_forward_hook(nla._hook)
    return nla


# ----------------------------------------------------------------------------- batched scoring
# Added for the sprint pipeline (src/mats README): module-level helpers only; the NLA class above
# is unchanged. They reuse NLA._inject, whose hook already accepts one vector per batch row.


@torch.no_grad()
def logprob_of_strings_batch(
    nla: NLA,
    acts: torch.Tensor,
    candidates: Sequence[str],
    prefix: str = EXPLANATION_PREFIX,
    batch: int = 32,
) -> torch.Tensor:
    """[N, C] summed log p(candidate | AV prompt with act_i injected, prefix), batched over
    activations. All C candidates of one activation always share a forward batch."""
    acts = acts.reshape(-1, acts.shape[-1]).float()
    head = nla.prompt_ids + nla.tok.encode(prefix, add_special_tokens=False)
    cands = [nla.tok.encode(c, add_special_tokens=False) for c in candidates]
    C = len(cands)
    per = max(1, batch // C)
    out = torch.empty(acts.shape[0], C)
    n = len(head) + max(map(len, cands))
    for s in range(0, acts.shape[0], per):
        a = acts[s : s + per]
        rows = [(i, c) for i in range(a.shape[0]) for c in range(C)]
        ids = torch.full((len(rows), n), nla.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, (_, c) in enumerate(rows):
            seq = head + cands[c]
            ids[r, : len(seq)] = torch.tensor(seq)
            mask[r, : len(seq)] = 1
        ids, mask = ids.to(nla.device), mask.to(nla.device)
        v = a.repeat_interleave(C, dim=0).to(nla.device)
        keep = n - len(head) + 1
        with nla._inject(v):
            logits = nla.av(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tok_lp = lp.gather(-1, ids[:, len(head):].unsqueeze(-1)).squeeze(-1)
        for r, (i, c) in enumerate(rows):
            out[s + i, c] = float(tok_lp[r, : len(cands[c])].sum())
    return out


def paired_logodds_batch(
    nla: NLA, acts: torch.Tensor, cand_a: str, cand_b: str, prefix: str = EXPLANATION_PREFIX,
    batch: int = 32,
) -> torch.Tensor:
    """[N] log p(a | act_i) - log p(b | act_i) (native explanation register)."""
    lp = logprob_of_strings_batch(nla, acts, [cand_a, cand_b], prefix=prefix, batch=batch)
    return lp[:, 0] - lp[:, 1]


def paired_logodds_templates(
    nla: NLA, acts: torch.Tensor, pairs: Sequence[tuple[str, str]],
    prefix: str = EXPLANATION_PREFIX, batch: int = 32,
) -> torch.Tensor:
    """[T, N] LO per (a, b) template pair."""
    return torch.stack([paired_logodds_batch(nla, acts, a, b, prefix, batch) for a, b in pairs])


# ----------------------------------------------------------------------------- training
# added 28 Oct 2026 for exp 30 (supervised naming term on the NLA's LoRA). Nothing above changes:
# ``load_nla(trainable=True)`` registers the cloning hook below in place of ``NLA._hook`` and
# ``nll_of_targets`` is the gradient twin of ``logprob_of_strings_batch`` (same prompt, same
# injection, teacher-forced, loss on the target tokens after the ``<explanation>\n`` prefix).


def _grad_hook(nla: NLA):  # noqa: ANN202
    """Forward hook with the arithmetic of ``NLA._hook`` that clones the hidden state before writing
    the injected row, so autograd flows through the injection (the in-place write is safe only under
    ``no_grad``). Returns a new tuple / tensor in place of the block output."""

    def hook(_mod: Any, _inp: Any, out: Any) -> Any:
        h = out[0] if isinstance(out, tuple) else out
        v = nla._vec.get("v")
        if v is None or h.shape[1] <= nla.marker:  # decode steps with KV cache
            return out
        if v.shape[0] != h.shape[0]:
            raise RuntimeError(f"{v.shape[0]} vectors for a batch of {h.shape[0]}")
        hp = h[:, nla.marker, :]
        vu = v.to(h.device, torch.float32)
        vu = (vu / (vu.norm(dim=-1, keepdim=True) + 1e-9)).to(h.dtype)
        h2 = h.clone()
        h2[:, nla.marker, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    return hook


def _active_adapter_name(av: Any) -> str:
    name = getattr(av, "active_adapter", None)
    if not isinstance(name, str):
        raise RuntimeError("no peft adapter is active: pass param_filter to make_trainable")
    return name


def make_trainable(
    nla: NLA,
    block1: Any | None = None,
    param_filter: Any | None = None,
    checkpointing: bool = True,
) -> NLA:
    """Put the AV in training mode for exp 30: the cloning injection hook on block 1, gradients on the
    active LoRA adapter only (``param_filter(name) -> bool`` overrides the choice of parameters),
    gradient checkpointing (non-reentrant) with input grads enabled, KV cache off, ``train()`` mode.
    Returns ``nla``; the trainable parameter count is in ``nla._vec["n_trainable"]``."""
    av = nla.av
    if param_filter is None:
        name = _active_adapter_name(av)
        param_filter = lambda n, _a=name: "lora_" in n and f".{_a}." in n  # noqa: E731
    n_train = 0
    for n, p in av.named_parameters():
        on = bool(param_filter(n))
        p.requires_grad_(on)
        n_train += p.numel() if on else 0
    if n_train == 0:
        raise RuntimeError("make_trainable: no parameter matched; nothing to train")
    if checkpointing:
        av.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        av.enable_input_require_grads()
    av.config.use_cache = False
    av.train()
    if block1 is None:
        (block1,) = [mod for n, mod in av.named_modules() if n.endswith("model.layers.1")]
    block1.register_forward_hook(_grad_hook(nla))
    nla._vec["n_trainable"] = n_train
    return nla


def eos_id(nla: NLA) -> int:
    """The token the AV emits after ``</explanation>`` (first eos of its generation config)."""
    gc = getattr(nla.av, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(e, (list, tuple)):
        e = e[0] if e else None
    return int(e if e is not None else nla.tok.eos_token_id)


def nll_of_targets(
    nla: NLA,
    acts: torch.Tensor,
    targets: Sequence[str],
    prefix: str = EXPLANATION_PREFIX,
    add_eos: bool = True,
    backward: bool = False,
    scale: float = 1.0,
) -> torch.Tensor:
    """Mean token NLL of ``targets[i]`` (+ eos) given the AV prompt with ``acts[i]`` injected and
    ``prefix``: the gradient twin of :func:`logprob_of_strings_batch` (one target per activation,
    right padding, loss masked to the target tokens). With ``backward=True`` the backward of
    ``loss * scale`` runs INSIDE the injection context and the detached loss is returned; this is
    required under gradient checkpointing, whose recompute re-runs the hook and must still see
    the injected vectors (refused otherwise)."""
    import torch.nn.functional as F

    acts = acts.reshape(-1, acts.shape[-1]).float()
    if len(targets) != acts.shape[0]:
        raise ValueError(f"{len(targets)} targets for {acts.shape[0]} activations")
    if not backward and torch.is_grad_enabled() and getattr(nla.av, "is_gradient_checkpointing", False):
        raise RuntimeError("gradient checkpointing recomputes the hook: pass backward=True")
    head = nla.prompt_ids + nla.tok.encode(prefix, add_special_tokens=False)
    tail = [eos_id(nla)] if add_eos else []
    tg = [nla.tok.encode(t, add_special_tokens=False) + tail for t in targets]
    n = len(head) + max(map(len, tg))
    ids = torch.full((len(tg), n), nla.tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full((len(tg), n - len(head)), -100, dtype=torch.long)
    for r, t in enumerate(tg):
        ids[r, : len(head) + len(t)] = torch.tensor(head + t)
        mask[r, : len(head) + len(t)] = 1
        labels[r, : len(t)] = torch.tensor(t)
    ids, mask, labels = ids.to(nla.device), mask.to(nla.device), labels.to(nla.device)
    keep = n - len(head) + 1  # logits from the last head token on (each predicts the next target token)
    with nla._inject(acts.to(nla.device)):
        logits = nla.av(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
        lp = logits[:, :-1].float()
        loss = F.cross_entropy(lp.reshape(-1, lp.shape[-1]), labels.reshape(-1), ignore_index=-100)
        if backward:
            (loss * scale).backward()
            return loss.detach()
    return loss


def save_adapter(nla: NLA, out_dir: str | Path) -> str:
    """``av.save_pretrained`` in peft's layout; returns the directory ``load_nla(adapter=...)`` takes
    (peft writes a non-"default" adapter under ``out_dir/<adapter name>``)."""
    nla.av.save_pretrained(str(out_dir))
    name = getattr(nla.av, "active_adapter", None)
    path = Path(out_dir) / name if isinstance(name, str) and name != "default" else Path(out_dir)
    if not path.is_dir():
        raise RuntimeError(f"adapter was not written under {path}")
    return str(path)
