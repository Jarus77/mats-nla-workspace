"""The Oracle Lens activation reconstructor (OL-AR): ``[Layer NN] <span>`` -> the AR's prediction of the block-NN
activation at the span's last token, in the AR's own whitened output space.

Lifted verbatim from exp 14's ``modal14.py`` (class ``OLensAR``) on 27 Oct 2026 so that later experiments can import
it without touching exp 14; the two loader helpers (``rename_lora_keys``, ``right_pad``) and the two calibration
scores (``fve_raw``, ``retrieval_at_1``) come from exp 14's ``ar_lib.py``, the constants from ``spec14.py``.

What exp 14 established about this loader (its RESULTS.md, loader note):
- the shipped LoRA is live; the tags decode to ``[Layer NN]``; tags exist for layers 20..60 step 4 and 63 (no 42);
- the head reads target layer li at backbone block ``layer_indices[li]`` (blocks 5..16), not the final block;
- the head output is in the AR's whitened space, whose moments are gated (``agu18dec/local-workspace``), so rebuilds
  are compared with each other inside that space, never with raw activations;
- ``calibrate`` (retrieval@1 of true spans against gold activations) chose the ``("li", "ln")`` variant at both 44
  and 40 (exp 14 ``results/calib14.json``); it is also the constructor default, so without gold set ``variant``
  explicitly and record it.

torch / peft / transformers are imported inside methods, as in the original, so the module imports without them.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

AR_REPO, AR_SUBDIR = "agu18dec/olens_and_ar", "ar_ptag_pooled"
BASE = "Qwen/Qwen3.6-27B"
DEFAULT_VARIANT = ("li", "ln")


# ----------------------------------------------------------------------------- loader helpers (exp 14 ar_lib)
def rename_lora_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the torch.compile ``_orig_mod.`` wrapper segment so PEFT matches the plain backbone."""
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def right_pad(rows: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ``[B, T]`` ids and mask; ``mask.sum(1) - 1`` is each row's last real token."""
    import torch

    n = max(len(r) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(list(r), dtype=torch.long)
        mask[i, : len(r)] = 1
    return ids, mask


def fve_raw(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """1 - MSE(pred, gold) / predict-the-mean MSE of ``gold`` (raw space, no whitening)."""
    pred, gold = pred.float(), gold.float()
    base = ((gold - gold.mean(0)) ** 2).mean()
    return float(1 - ((pred - gold) ** 2).mean() / base)


def retrieval_at_1(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """Fraction of rows whose nearest gold vector (cosine, after centring on gold's mean) is its own."""
    import torch

    pred, gold = pred.float(), gold.float()
    mu = gold.mean(0)
    p = torch.nn.functional.normalize(pred - mu, dim=-1)
    g = torch.nn.functional.normalize(gold - mu, dim=-1)
    nearest = (p @ g.T).argmax(1)
    return float((nearest == torch.arange(len(gold))).float().mean())


# ----------------------------------------------------------------------------- the reconstructor (exp 14 modal14)
class OLensAR:
    """The ptag reconstructor: ``[Layer NN] <span>`` -> predicted block-NN activation at the span's
    last token. Backbone = Qwen3.6-27B text model + the shipped LoRA (compile-wrapper keys renamed),
    head = norm (no affine) + linear. Which norm the head uses and whether the backbone's final norm
    is applied are not documented; both are calibrated on true spans (retrieval@1) before use."""

    def __init__(self, device: str = "cuda") -> None:
        import shutil
        import tempfile

        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from mats.capture import decoder_blocks

        root = Path(snapshot_download(AR_REPO, allow_patterns=[f"{AR_SUBDIR}/*"])) / AR_SUBDIR
        self.tok = AutoTokenizer.from_pretrained(BASE)
        full = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map=device).eval()
        text = full.model
        tmp = Path(tempfile.mkdtemp())
        sd = rename_lora_keys(load_file(str(root / "lora" / "adapter_model.safetensors")))
        save_file(sd, str(tmp / "adapter_model.safetensors"))
        shutil.copy(root / "lora" / "adapter_config.json", tmp / "adapter_config.json")
        self.model = PeftModel.from_pretrained(text, str(tmp)).eval()
        # the adapter must be live: one loaded B matrix equals the checkpoint's
        key = next(k for k in sd if k.endswith("lora_B.weight"))
        live = dict(self.model.named_parameters())[key.replace("lora_B.weight", "lora_B.default.weight")]
        assert torch.allclose(live.float().cpu(), sd[key].float()), "AR LoRA did not load"
        assert sd[key].abs().sum() > 0, "AR LoRA B matrix is zero"
        heads = torch.load(root / "heads.pt", map_location="cpu", weights_only=False)
        self.W = heads["head"]["linear.weight"].float().to(device)
        self.b = heads["head"]["linear.bias"].float().to(device)
        self.tag_ids = {int(L): heads["tag_ids"][i].tolist() for i, L in enumerate(heads["layers"])}
        self.final_norm = self.model.base_model.model.norm
        self.blocks = decoder_blocks(self.model)
        meta = json.load(open(root / "meta.json"))["config"]
        # ptag reads target layer li at backbone block layer_indices[li] (12 blocks, 5..16)
        self.read_block = {int(L): int(meta["layer_indices"][i]) for i, L in enumerate(heads["layers"])}
        self.device = device
        self.variant = DEFAULT_VARIANT
        self._cur = {}
        for i, blk in enumerate(self.blocks):
            blk.register_forward_hook(lambda _m, _i, o, i=i: self._cur.__setitem__(i, (o[0] if isinstance(o, tuple) else o)))

    @staticmethod
    def _head_norm(x, kind: str):  # noqa: ANN001
        import torch

        if kind == "ln":
            return torch.nn.functional.layer_norm(x, (x.shape[-1],))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def reconstruct(self, rows: list[list[int]], layer: int, variant=None, batch: int = 16):  # noqa: ANN001
        """[N, d] fp32 CPU predictions for token-id rows under the given (final_norm, head_norm)."""
        import torch

        fn, hn = variant or self.variant
        tag = self.tag_ids[layer]
        out = []
        with torch.no_grad():
            for s in range(0, len(rows), batch):
                chunk = [tag + list(r) for r in rows[s : s + batch]]
                ids, mask = right_pad(chunk, self.tok.pad_token_id or 0)
                ids, mask = ids.to(self.device), mask.to(self.device)
                self.model(input_ids=ids, attention_mask=mask)
                h = self._cur[self.read_block[layer] if fn == "li" else len(self.blocks) - 1]
                last = h[torch.arange(len(chunk)), mask.sum(1) - 1]
                if fn == "last_post":
                    last = self.final_norm(last)
                x = self._head_norm(last.float(), hn)
                out.append((x @ self.W.T + self.b).cpu())
        return torch.cat(out)

    def calibrate(self, rows: list[list[int]], gold, layer: int) -> dict:  # noqa: ANN001
        """Pick (final_norm, head_norm) by retrieval@1 of true-span rebuilds against gold activations."""
        res = {}
        for fn in ("li", "last_pre", "last_post"):
            for hn in ("ln", "rms"):
                p = self.reconstruct(rows, layer, (fn, hn))
                res[f"{fn}/{hn}"] = {"ret1": retrieval_at_1(p, gold), "fve": fve_raw(p, gold)}
        best = max(res, key=lambda k: res[k]["ret1"])
        self.variant = tuple(best.split("/"))
        return {"scores": res, "chosen": best}
