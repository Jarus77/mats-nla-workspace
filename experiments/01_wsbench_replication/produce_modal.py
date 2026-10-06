"""WorkspaceBench replication: produce readouts on Modal with the fork's Python Producer API.

    cd experiments/01_wsbench_replication
    PYTHONPATH=../../infra modal run produce_modal.py::download_jlens        # once
    PYTHONPATH=../../infra modal run produce_modal.py --arm olens --families typo_mt --limit 2 --check
    PYTHONPATH=../../infra modal run --detach produce_modal.py --arm olens --families typo_mt,multilingual_mt

Arms (output: /results/01_wsbench_replication/readouts/<arm>/<family>.jsonl):
  olens  OLens with the LoRA from the open model repo agu18dec/olens_and_ar:olens_s3d_rl600
         (byte-identical to the gated default), layers GRID + 42 (12 layers), H100.
  jlens  JLens (neuronpedia n1000 wikitext Jacobians), layers GRID + 42, H100.
  nla    NLA (ceselder/qwen3.6-27b-nla-rl, iter_000400), wsbench default layer 42, H200.

Everything (render, capture, carrier prompts, injection, sampling settings) is the fork's own
code. The only change is that verbalizer generations are BATCHED across cells (same prompt
length, no padding, independent sampling per row), because the bench's batch-1 loop runs at
~5 tok/s on this image. `--check` compares batched vs the bench's own batch-1 `read` under
greedy decoding on a few cells.
"""

import json
import time
from pathlib import Path

import modal

from modal_common import (
    HF_ENV,
    HF_PINS,
    HF_SECRETS,
    RESULTS_DIR,
    VOLUMES,
    download_image,
    weights_volume,
)

APP_NAME = "mats-01-wsbench-replication"
_HERE = Path(__file__).resolve()
# parents[2] exists only locally (in the container this file sits at /root/produce_modal.py)
FORK_LOCAL = (_HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent) / "third_party" / "workspace-bench"
FORK_REMOTE = "/wsbench"
OUT_ROOT = f"{RESULTS_DIR}/01_wsbench_replication"
MODEL = "Qwen/Qwen3.6-27B"
GRID = list(range(20, 61, 4))
GRID_PLUS_42 = sorted({*GRID, 42})
JLENS_REPO = "neuronpedia/jacobian-lens"
JLENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
OLENS_REPO, OLENS_SUB = "agu18dec/olens_and_ar", "olens_s3d_rl600"
DEFAULT_BATCH = {"olens": 64, "nla": 50, "jlens": 1}

app = modal.App(APP_NAME)

# Same pins as modal_common.gpu_image (which ends in a local mount, so it cannot be extended with
# build steps), plus the fork's core deps; the fork is mounted and put on PYTHONPATH.
wsbench_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "peft==0.21.0",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "numpy==2.5.3",
        "pyyaml",
        "flash-linear-attention==0.5.2",
        *HF_PINS,
        "openai>=1.40",
        "anthropic>=1.0",
        "pydra-config>=0.0.17",
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": f"{FORK_REMOTE}/src"})
    .add_local_python_source("modal_common")
    .add_local_dir(
        str(FORK_LOCAL),
        FORK_REMOTE,
        ignore=[".venv", ".venv/**", "outputs", "outputs/**", "**/__pycache__", ".git", "tests"],
    )
)


# ------------------------------------------------------------------ weights


@app.function(image=download_image, volumes=VOLUMES, secrets=HF_SECRETS, cpu=4, memory=8192,
              timeout=2 * 60 * 60)
def download_jlens() -> str:
    """Add the neuronpedia J-lens Jacobians to rfm-weights (same HF cache as everything else)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(JLENS_REPO, JLENS_FILE)
    weights_volume.commit()
    size = Path(path).stat().st_size / 1e9
    print(f"jlens -> {path} ({size:.2f} GB); volume committed", flush=True)
    return path


# ------------------------------------------------------------------ batched methods (container)


def _make_methods():
    """Subclasses of the fork's methods. Imported lazily: wsbench exists only in the container."""
    import torch
    from wsbench.produce import NLA, OLens
    from wsbench.produce.methods import Readout

    class OLensBatched(OLens):
        """OLens with the open model-repo LoRA and a batched read over cells."""

        def bind(self, backend):  # noqa: ANN001
            from huggingface_hub import snapshot_download
            from peft import PeftModel

            d = snapshot_download(OLENS_REPO, allow_patterns=[f"{OLENS_SUB}/*"])
            backend.model = PeftModel.from_pretrained(backend.model, f"{d}/{OLENS_SUB}")
            backend.model.eval()
            self._b = backend

        def read_batch(self, hs, layers):  # noqa: ANN001
            b = self._b
            carriers = [self._carrier(L) for L in layers]
            if len({len(ids) for ids, _ in carriers}) != 1:  # never padded; split by length
                return [self.read(h, L) for h, L in zip(hs, layers, strict=True)]
            embed = b.model.get_input_embeddings()
            with torch.no_grad():
                e = embed(torch.tensor([ids for ids, _ in carriers], device=b.device)).clone()
                for i, ((_, slot), h) in enumerate(zip(carriers, hs, strict=True)):
                    v = h.to(b.device).float()
                    e[i, slot, :] = (self.alpha * v / v.norm().clamp_min(1e-9)).to(e.dtype)
                s = self.sampling
                out = b.model.generate(
                    inputs_embeds=e,
                    attention_mask=torch.ones(e.shape[:2], device=b.device, dtype=torch.long),
                    do_sample=s.temperature > 0,
                    temperature=max(s.temperature, 1e-5),
                    top_p=s.top_p,
                    top_k=s.top_k,
                    max_new_tokens=s.max_new_tokens,
                    num_return_sequences=s.k,
                    pad_token_id=b.tokenizer.pad_token_id or b.tokenizer.eos_token_id,
                )
            texts = [t.strip() for t in b.tokenizer.batch_decode(out, skip_special_tokens=True)]
            k = s.k
            return [Readout(samples=texts[i * k:(i + 1) * k]) for i in range(len(hs))]

    class NLABatched(NLA):
        """NLA whose block-1 hook adds a per-row vector, so a batch of cells shares a generate."""

        def bind(self, backend):  # noqa: ANN001
            super().bind(backend)
            # replace the parent's single-vector hook with a per-row one (same arithmetic)
            (block1,) = [m for n, m in self._reader.named_modules()
                         if n.endswith("model.layers.1")]
            for hid, fn in list(block1._forward_hooks.items()):
                if "NLA.bind" in getattr(fn, "__qualname__", ""):
                    del block1._forward_hooks[hid]
            mp = self._marker

            def hook(_mod, _inp, out):  # noqa: ANN001, ANN202
                h = out[0] if isinstance(out, tuple) else out
                v = self._vec.get("v")
                if v is None or h.shape[1] <= mp:
                    return out
                v = v.to(h.device)
                hp = h[:, mp, :]
                vu = v.to(h.dtype) / (v.norm(dim=-1, keepdim=True) + 1e-9).to(h.dtype)
                h[:, mp, :] = hp + hp.norm(dim=-1, keepdim=True) * vu
                return out

            block1.register_forward_hook(hook)

        def read(self, h, layer):  # noqa: ANN001
            return self.read_batch([h], [layer])[0]

        def read_batch(self, hs, layers):  # noqa: ANN001, ARG002
            b = self._b
            s = self.sampling
            v = torch.stack([h.float() for h in hs])
            self._vec["v"] = v.repeat_interleave(s.k, dim=0)
            try:
                with torch.no_grad():
                    out = self._reader.generate(
                        torch.tensor([self._ids] * len(hs), device=b.device),
                        attention_mask=torch.ones(len(hs), len(self._ids), device=b.device,
                                                  dtype=torch.long),
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
            texts = self._tok.batch_decode(out[:, len(self._ids):], skip_special_tokens=True)
            texts = [t.split("</explanation>")[0].removeprefix("<explanation>").strip()
                     for t in texts]
            k = s.k
            return [Readout(samples=texts[i * k:(i + 1) * k]) for i in range(len(hs))]

    return OLensBatched, NLABatched


def _build_method(arm: str, nla_layer: int = 42):
    from wsbench.produce import JLens

    olens_cls, nla_cls = _make_methods()
    if arm == "olens":
        return olens_cls(), GRID_PLUS_42
    if arm == "nla":
        return nla_cls(layers=[nla_layer]), None  # wsbench default: NLA.layers = [42]
    if arm == "jlens":
        return JLens(), GRID_PLUS_42
    raise ValueError(f"unknown arm {arm!r}")


def _read_cells(method, cells, batch: int):  # noqa: ANN001
    """cells: [(id, layer, pos, token, h)] -> contract rows."""
    rows = []
    for i in range(0, len(cells), batch):
        chunk = cells[i:i + batch]
        hs, layers = [c[4] for c in chunk], [c[1] for c in chunk]
        if hasattr(method, "read_batch") and batch > 1:
            outs = method.read_batch(hs, layers)
        else:
            outs = [method.read(h, L) for h, L in zip(hs, layers, strict=True)]
        for (iid, L, p, tok, _), r in zip(chunk, outs, strict=True):
            rows.append({"id": iid, "layer": L, "pos": p, "token": tok, **r.row()})
    return rows


def _capture_family(producer, family: str, layers, limit: int, skip: set):  # noqa: ANN001
    """Render + capture every plan row with the fork's own code; yield per-item cell lists."""
    from wsbench import readplan
    from wsbench.produce.producer import _positions
    from wsbench.produce.render import render

    specs = readplan.plan(family)[: limit or None]
    for spec in specs:
        if spec.id in skip:
            continue
        rendered = render(spec, producer.backend.tokenizer)
        lays = producer._layers(layers, spec.layers)
        pos = _positions(spec.positions, rendered.tokens)
        idx = [p % len(rendered) for p in pos]
        acts = producer.backend.capture(rendered.ids, lays, idx)
        yield spec.id, [(spec.id, L, p, rendered.decoded[j], acts[L][i])
                        for L in lays for i, (p, j) in enumerate(zip(pos, idx, strict=True))]


def _greedy_check(producer, family: str, layers, n_cells: int = 4) -> dict:  # noqa: ANN001
    """Greedy decode: the bench's own batch-1 read vs our batched read on the same cells."""
    m = producer.method
    if not hasattr(m, "read_batch"):
        return {"skipped": "no batched read"}
    cells = []
    for _, cs in _capture_family(producer, family, layers, 2, set()):
        cells.extend(cs)
    cells = cells[:n_cells]
    saved = m.sampling.temperature
    m.sampling.temperature = 0.0
    try:
        base_cls = type(m).__mro__[1]
        single = [base_cls.read(m, c[4], c[1]).samples[0] if base_cls.__name__ == "OLens"
                  else m.read_batch([c[4]], [c[1]])[0].samples[0] for c in cells]
        batched = [r.samples[0] for r in m.read_batch([c[4] for c in cells], [c[1] for c in cells])]
    finally:
        m.sampling.temperature = saved
    same = [a == b for a, b in zip(single, batched, strict=True)]
    return {"n": len(cells), "identical": sum(same), "single": single, "batched": batched}


def _produce(arm: str, families: list[str], limit: int, batch: int, check: bool,
             nla_layer: int = 42) -> dict:
    import torch
    from wsbench.produce import Producer

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    method, layers = _build_method(arm, nla_layer)
    tag = f"{arm}_L{nla_layer}" if arm == "nla" and nla_layer != 42 else arm
    producer = Producer.load(MODEL, method)
    load_s = time.time() - t0
    gpu = torch.cuda.get_device_name(0)
    print(f"[{arm}] loaded on {gpu} in {load_s:.0f}s; layers={layers or method.layers}", flush=True)
    summary = {"arm": arm, "gpu": gpu, "load_s": round(load_s, 1), "batch": batch, "families": {}}
    if check:
        summary["greedy_check"] = _greedy_check(producer, families[0], layers)
        print(json.dumps(summary["greedy_check"], ensure_ascii=False)[:3000], flush=True)
    for fam in families:
        out = Path(f"{OUT_ROOT}/readouts/{tag}{'_smoke' if limit else ''}/{fam}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {json.loads(x)["id"] for x in out.read_text().splitlines() if x.strip()}
        tf, n_cells, n_items, pending = time.time(), 0, 0, []
        cap_s, read_s = 0.0, 0.0
        gen = _capture_family(producer, fam, layers, limit, done)
        with out.open("a", encoding="utf-8") as fh:
            while True:
                tc = time.time()
                item = next(gen, None)
                cap_s += time.time() - tc
                if item is not None:
                    pending.extend(item[1])
                    n_items += 1
                flush_now = item is None or len(pending) >= max(batch, 1)
                if flush_now and pending:
                    tr = time.time()
                    rows = _read_cells(producer.method, pending, batch)
                    read_s += time.time() - tr
                    # rows are written grouped by item; partial items only at the very end
                    fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                    fh.flush()
                    n_cells += len(rows)
                    pending = []
                    VOLUMES[RESULTS_DIR].commit()
                    rate = (time.time() - tf) / max(n_cells, 1)
                    print(f"[{arm}/{fam}] {n_items} items, {n_cells} cells, {rate:.2f} s/cell",
                          flush=True)
                if item is None:
                    break
        wall = time.time() - tf
        summary["families"][fam] = {
            "items": n_items, "cells": n_cells, "wall_s": round(wall, 1),
            "capture_s": round(cap_s, 1), "read_s": round(read_s, 1),
            "s_per_cell": round(wall / max(n_cells, 1), 3),
            "s_per_item": round(wall / max(n_items, 1), 3), "skipped_done": len(done),
        }
    summary["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    summary["total_s"] = round(time.time() - t0, 1)
    meta = Path(f"{OUT_ROOT}/meta/{tag}_{'-'.join(families)}{'_smoke' if limit else ''}.json")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    VOLUMES[RESULTS_DIR].commit()
    print(json.dumps({k: v for k, v in summary.items() if k != "greedy_check"}, indent=1), flush=True)
    return summary


_FN_KW = {"image": wsbench_image, "volumes": VOLUMES, "timeout": 8 * 60 * 60, "memory": 98304}


@app.function(gpu="H100", **_FN_KW)
def produce_h100(arm: str, families: list[str], limit: int = 0, batch: int = 1,
                 check: bool = False, nla_layer: int = 42) -> dict:
    return _produce(arm, families, limit, batch, check, nla_layer)


@app.function(gpu="H200", **_FN_KW)
def produce_h200(arm: str, families: list[str], limit: int = 0, batch: int = 1,
                 check: bool = False, nla_layer: int = 42) -> dict:
    return _produce(arm, families, limit, batch, check, nla_layer)


@app.local_entrypoint()
def main(arm: str, families: str = "typo_mt,multilingual_mt", limit: int = 0, batch: int = 0,
         check: bool = False, per_family: bool = False, nla_layer: int = 42) -> None:
    fams = [f.strip() for f in families.split(",") if f.strip()]
    fn = produce_h200 if arm in ("nla", "jlens") else produce_h100  # jlens OOMs an H100 (J+W_U fp32)
    b = batch or DEFAULT_BATCH[arm]
    groups = [[f] for f in fams] if per_family else [fams]
    calls = [fn.spawn(arm, g, limit, b, check, nla_layer) for g in groups]
    for c in calls:
        print(json.dumps(c.get(), indent=1, ensure_ascii=False)[:4000])
