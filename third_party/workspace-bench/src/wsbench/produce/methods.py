"""Readout methods. A method turns one residual vector at (layer, position) into a readout: a
ranked token list for the vector lenses, free text for the verbalizers. All of them share the
``Method`` protocol so the producer treats them alike."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .backend import Backend
from .render import display_tokens

TOP_K = 10


@dataclass(frozen=True)
class Readout:
    """One cell's readout: ``tokens`` + ``scores`` for a token lens, ``samples`` for a
    verbalizer."""

    tokens: list[str] | None = None
    scores: list[float] | None = None
    samples: list[str] | None = None

    def row(self) -> dict[str, Any]:
        if self.samples is not None:
            return {"samples": self.samples}
        return {"tokens": self.tokens, "scores": self.scores}


class Method(Protocol):
    name: str
    layers: list[int] | None  # fixed read layers (a lens trained at one layer), else None

    def bind(self, backend: Backend) -> None: ...  # load artifacts onto the backend's device
    def read(self, h: Any, layer: int) -> Readout: ...  # h: [d] fp32 CPU residual vector


# ---------------------------------------------------------------- vector lenses


@dataclass
class LogitLens:
    """``W_U · norm(h)``: the model's own unembedding after its own final norm module. No
    artifact."""

    name: str = "logit_lens"
    layers: list[int] | None = None
    k: int = TOP_K
    _b: Backend | None = field(default=None, repr=False)
    _w_u: Any = field(default=None, repr=False)

    def bind(self, backend: Backend) -> None:
        self._b = backend
        self._w_u = backend.unembed.float()

    def read(self, h: Any, layer: int) -> Readout:
        import torch

        b = self._b
        assert b is not None
        x = b.final_norm(h.to(b.device))  # the model's own norm module, fp32 in, fp32 out
        vals, ids = torch.topk(x @ self._w_u.T, self.k)
        return Readout(
            tokens=display_tokens(b.tokenizer, ids.tolist()),
            scores=[round(v, 4) for v in vals.tolist()],
        )


def _load_jacobians(repo: str, filename: str, device: str) -> Any:
    """``J[layer, d, d]`` fp32 from a J-lens ``.pt`` (the official ``JacobianLens`` layout with
    ``J`` keyed by layer, or a plain ``jacobians`` stack)."""
    import torch
    from huggingface_hub import hf_hub_download

    obj = torch.load(hf_hub_download(repo, filename), map_location="cpu", weights_only=False)
    jac = obj["J"] if "J" in obj else obj["jacobians"]
    if isinstance(jac, dict):
        layers = sorted(int(k) for k in jac)
        stacked = torch.stack([jac[L] if L in jac else jac[str(L)] for L in layers]).float()
    else:
        stacked = jac.float()
        layers = list(range(stacked.shape[0]))
    if layers != list(range(len(layers))):
        raise ValueError(f"{filename}: source_layers not contiguous from 0: {layers}")
    return stacked.to(device)


@dataclass
class JLens:
    """The Jacobian lens with the magnitude-free cosine readout ``(W_U·J·h) / ‖Jᵀ W_U[t]‖``, the
    readout of record for the benchmark's J-lens arm. Default artifact: neuronpedia's n=1000
    wikitext lens for Qwen3.6-27B."""

    name: str = "jlens"
    layers: list[int] | None = None
    repo: str = "neuronpedia/jacobian-lens"
    filename: str = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
    k: int = TOP_K
    _b: Backend | None = field(default=None, repr=False)
    _jac: Any = field(default=None, repr=False)
    _w_u: Any = field(default=None, repr=False)
    _denom: dict[int, Any] = field(default_factory=dict, repr=False)

    def bind(self, backend: Backend) -> None:
        self._b = backend
        self._jac = _load_jacobians(self.repo, self.filename, backend.device)
        self._w_u = backend.unembed.float()

    def read(self, h: Any, layer: int) -> Readout:
        import torch

        b = self._b
        assert b is not None and self._jac is not None
        if layer >= self._jac.shape[0]:
            raise ValueError(f"layer {layer} beyond the lens's {self._jac.shape[0]} source layers")
        w_u = self._w_u
        if layer not in self._denom:
            self._denom[layer] = (w_u @ self._jac[layer]).norm(dim=1).clamp_min(1e-9)
        scores = ((h.to(b.device) @ self._jac[layer].T) @ w_u.T) / self._denom[layer]
        vals, ids = torch.topk(scores, self.k)
        return Readout(
            tokens=display_tokens(b.tokenizer, ids.tolist()),
            scores=[round(v, 4) for v in vals.tolist()],
        )


@dataclass
class RLens:
    """The RelP lens (paper recipe, n=25, penultimate target) with the standard readout
    ``W_U · norm(J·h)``. Default artifact: ``camilablank/workspace-lenses``."""

    name: str = "rlens"
    layers: list[int] | None = None
    repo: str = "camilablank/workspace-lenses"
    filename: str = "qwen3.6-27b/r-lens/lens.pt"
    k: int = TOP_K
    _b: Backend | None = field(default=None, repr=False)
    _jac: Any = field(default=None, repr=False)
    _w_u: Any = field(default=None, repr=False)

    def bind(self, backend: Backend) -> None:
        self._b = backend
        self._jac = _load_jacobians(self.repo, self.filename, backend.device)
        self._w_u = backend.unembed.float()

    def read(self, h: Any, layer: int) -> Readout:
        import torch

        b = self._b
        assert b is not None and self._jac is not None
        x = b.final_norm(h.to(b.device) @ self._jac[layer].T)
        vals, ids = torch.topk(x @ self._w_u.T, self.k)
        return Readout(
            tokens=display_tokens(b.tokenizer, ids.tolist()),
            scores=[round(v, 4) for v in vals.tolist()],
        )


# ---------------------------------------------------------------- verbalizers


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 256
    k: int = 1  # samples per cell


@dataclass
class OLens:
    """The oracle lens: a LoRA on the probed model that verbalizes an activation placed in an
    ``<activation>`` slot of a carrier prompt. The residual vector replaces the slot token's
    input embedding as ``alpha · unit(h)``; the adapter is disabled while capturing and enabled
    while generating, so one model serves both roles."""

    name: str = "olens"
    layers: list[int] | None = None
    lora: str = "agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600"
    alpha: float = 16000.0
    prompt: str = (
        "An activation vector from layer {layer} of a language model is enclosed in activation "
        "tags: <activation>{char}</activation>. Produce distinct concepts that encode this "
        "activation, each as a '- ' bullet on its own line."
    )
    sampling: Sampling = field(default_factory=Sampling)
    _b: Backend | None = field(default=None, repr=False)
    _slots: dict[int, tuple[list[int], int]] = field(default_factory=dict, repr=False)

    def bind(self, backend: Backend) -> None:
        from huggingface_hub import snapshot_download
        from peft import PeftModel

        repo, _, sub = self.lora.partition(":")
        local = (
            snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{sub}/*"])
            if sub
            else snapshot_download(repo)
        )
        adir = f"{local}/{sub}" if sub else local
        if isinstance(backend.model, PeftModel):  # a previous verbalizer's adapter
            backend.model = backend.model.unload()
        backend.model = PeftModel.from_pretrained(backend.model, adir)
        backend.model.eval()
        self._b = backend

    def _carrier(self, layer: int) -> tuple[list[int], int]:
        """The carrier prompt's ids and its slot index, choosing an enclosed-ideograph char that
        survives as one token inside the full chat render (as the in-house renderer does)."""
        if layer in self._slots:
            return self._slots[layer]
        assert self._b is not None
        tok = self._b.tokenizer
        for code in range(0x3200, 0x3400):
            char = chr(code)
            ids = tok(char, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                continue
            out = tok.apply_chat_template(
                [{"role": "user", "content": self.prompt.format(layer=layer, char=char)}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rendered = list(out["input_ids"] if hasattr(out, "keys") else out)
            slots = [i for i, t in enumerate(rendered) if t == ids[0]]
            if len(slots) == 1:
                self._slots[layer] = (rendered, slots[0])
                return self._slots[layer]
        raise ValueError("no enclosed ideograph survives as a single token in the carrier prompt")

    def read(self, h: Any, layer: int) -> Readout:
        import torch

        b = self._b
        assert b is not None
        ids, slot = self._carrier(layer)
        embed = b.model.get_input_embeddings()
        with torch.no_grad():
            e = embed(torch.tensor([ids], device=b.device)).clone()
            v = h.to(b.device).float()
            e[0, slot, :] = (self.alpha * v / v.norm().clamp_min(1e-9)).to(e.dtype)
            s = self.sampling
            out = b.model.generate(
                inputs_embeds=e,
                attention_mask=torch.ones(1, len(ids), device=b.device, dtype=torch.long),
                do_sample=s.temperature > 0,
                temperature=max(s.temperature, 1e-5),
                top_p=s.top_p,
                top_k=s.top_k,
                max_new_tokens=s.max_new_tokens,
                num_return_sequences=s.k,
                pad_token_id=b.tokenizer.pad_token_id or b.tokenizer.eos_token_id,
            )
        texts = b.tokenizer.batch_decode(out, skip_special_tokens=True)
        return Readout(samples=[t.strip() for t in texts])


@dataclass
class NLA:
    """Karvonen's natural-language autoencoder: a separate reader (``av_base`` + a PEFT adapter)
    that receives the vector as a norm-matched ADD at the output of its decoder block 1, at the
    marker token of its own prompt. Trained on layer-42 activations of Qwen3.6-27B, so it reads
    at 42 unless the caller says otherwise; the vector's layer is not part of its prompt."""

    name: str = "nla"
    layers: list[int] | None = field(default_factory=lambda: [42])
    repo: str = "ceselder/qwen3.6-27b-nla-rl"
    adapter: str = "av_rl_adapters/iter_000400"
    sampling: Sampling = field(default_factory=Sampling)
    _b: Backend | None = field(default=None, repr=False)
    _reader: Any = field(default=None, repr=False)
    _tok: Any = field(default=None, repr=False)
    _ids: list[int] = field(default_factory=list, repr=False)
    _marker: int = field(default=-1, repr=False)
    _vec: dict[str, Any] = field(default_factory=dict, repr=False)

    def bind(self, backend: Backend) -> None:
        import torch
        import yaml
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._b = backend
        base = f"{snapshot_download(self.repo, allow_patterns=['av_base/*'])}/av_base"
        meta = yaml.safe_load(Path(f"{base}/nla_meta.yaml").read_text(encoding="utf-8"))
        tmpl = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
        marker_id = int(meta["tokens"]["injection_token_id"])
        self._tok = AutoTokenizer.from_pretrained(base)
        out = self._tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": tmpl.format(injection_char=meta["tokens"]["injection_char"]),
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self._ids = list(out["input_ids"] if hasattr(out, "keys") else out)
        (self._marker,) = [i for i, t in enumerate(self._ids) if t == marker_id]
        reader = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map=backend.device
        )
        if self.adapter and self.adapter != "none":
            local = snapshot_download(self.repo, allow_patterns=[f"{self.adapter}/*"])
            adir = f"{local}/{self.adapter}"
            reader = PeftModel.from_pretrained(reader, adir, torch_dtype=torch.bfloat16)
        reader.eval()
        (block1,) = [m for n, m in reader.named_modules() if n.endswith("model.layers.1")]
        mp = self._marker

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = self._vec.get("v")
            if v is None or h.shape[1] <= mp:
                return out
            hp = h[:, mp, :]
            vu = v.to(h.dtype).to(h.device) / (v.norm() + 1e-9)
            h[:, mp, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
            return out

        block1.register_forward_hook(hook)
        self._reader = reader

    def read(self, h: Any, layer: int) -> Readout:
        import torch

        b = self._b
        assert b is not None and self._reader is not None
        self._vec["v"] = h.float()
        s = self.sampling
        try:
            with torch.no_grad():
                out = self._reader.generate(
                    torch.tensor([self._ids], device=b.device),
                    do_sample=s.temperature > 0,
                    temperature=max(s.temperature, 1e-5),
                    top_p=s.top_p,
                    top_k=s.top_k,
                    max_new_tokens=s.max_new_tokens,
                    num_return_sequences=s.k,
                    pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
                )
        finally:
            self._vec["v"] = None
        texts = self._tok.batch_decode(out[:, len(self._ids) :], skip_special_tokens=True)
        texts = [t.split("</explanation>")[0].removeprefix("<explanation>").strip() for t in texts]
        return Readout(samples=texts)


METHODS: dict[str, type] = {
    "logit_lens": LogitLens,
    "jlens": JLens,
    "rlens": RLens,
    "olens": OLens,
    "nla": NLA,
}


def method(spec: str | Method, **kw: Any) -> Any:
    """A method instance from its name (``"jlens"``) or an instance passed through."""
    if isinstance(spec, str):
        if spec not in METHODS:
            raise KeyError(f"unknown method {spec!r}; known {sorted(METHODS)}")
        return METHODS[spec](**kw)
    return spec
