"""Exp 29 (app `m7cx`): naming and carrying along the NLA's training trajectory, on the exp 22 public bank.

Checkpoints: `av_base` (the warm-start verbalizer, no RL adapter: `nla.no_adapter()`), every `av_rl_adapters/iter_*`
on the volume snapshot (iter_000300 and iter_000400 as of 28 Oct; the fetch stage adds iter_000100..000800 from the
HF repo at the same revision), and `textnla` (base Qwen3.6-27B given the prompt TEXT under the NLA warm-start
instruction, exp 06's arm). For each checkpoint and each of exp 22's eight rows:
  gen29_{ckpt}_{row}.json    {"feature", "ckpt", "sel": {id: k=1 readout}, "mine": {id: [8 readouts]}}
                             "sel" holds the explanation payload (naming, carrying, text bag); "mine" holds the raw AV
                             text as exp 22 stored it for the judge (modal22 `_strip_expl`), so the judge sees the same
                             form it saw in exp 22.
  recon29_{ckpt}_{row}.json  the k=1 readouts rebuilt by the critic and scored by exp 14's frozen probe against the TRUE
                             layer-42 activations (lib25 `frozen_probe_auroc`), the class-direction cosine, and the FVE
                             of the rebuilds. The critic is the repo's single final reconstructor (the AR every sprint
                             experiment pairs with iter_000400; the repo saves no per-step AR), for EVERY checkpoint.
Stages (run29.sh; one GPU at a time):
  fetch  (CPU)   download the missing iter_* adapters into the volume snapshot at the pinned revision
  text   s5nt (H100)   the TextNLA arm: base Qwen, chat template, thinking off, NLA sampling (T=1, no top-p/top-k,
                       256 tokens); k=1 on the selection split, k=8 on the mining split -> gen29_textnla_{row}.json
  gen    d3wq (H200)   per checkpoint: AV (av_base + all adapters loaded, switched per checkpoint) k=1 and k=8 readouts,
                       then the critic pass. `--ckpts textnla` is recon-only (reads gen29_textnla_* from stage text).
Resumable per (checkpoint, row); the volume is committed after each.
    cd experiments/29_trajectory && modal run --detach modal29.py --stage gen --ckpts av_base,textnla
"""
import json
import time
from pathlib import Path

import modal

from env import NLA_REVISION, NLA_SNAPSHOT, OUT, OUT22, VOLUMES, download_image, gpu_image
from lib29 import CKPT_BASE, CKPT_TEXT, K_MINE, K_SEL, LAYER, MARKED, OUT_FILE, ROWS, adapter_path, checkpoint_order

app = modal.App("m7cx")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
EXP06_INSTRUCTION = HERE.parent / "06_resolving_decomposition" / "data" / "nla_instruction.txt"
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)
NLA_REPO = "ceselder/qwen3.6-27b-nla-rl"
ALL_ITERS = tuple(f"iter_{s:06d}" for s in range(100, 900, 100))
AV_BATCH = 64                 # modal22 h9vt / modal25 m4ks
TEXT_TOKEN_BUDGET = 20000     # prompt tokens x k per generate() call: half of exp 06 qwen_modal (H200) for the H100
TEXT_MAX_PROMPTS = 32


def _load(row: str):  # noqa: ANN202
    """exp 22's items of `row` and the true layer-42 activations; selection split labelled as modal22's `_sel`."""
    import torch

    from lib29 import select_split

    items = [it for it in json.load(open(f"{OUT22}/items_base.json")) if it["feature"] == row]
    blob = torch.load(f"{OUT22}/acts22_{row}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items], f"{row}: acts22 ids differ from items_base"
    sel, y, groups = select_split(items, MARKED[row])
    mine = [i for i, it in enumerate(items) if it["split"] == "mine"]
    return items, blob["acts"][LAYER].float(), sel, y, groups, mine


def _volume_iters() -> list[str]:
    return sorted(p.name for p in Path(NLA_SNAPSHOT, "av_rl_adapters").glob("iter_*") if (p / "adapter_model.safetensors").exists())


# ============================================================================ fetch (CPU, network on)

@app.function(image=download_image, volumes=VOLUMES, cpu=2, timeout=3600)
def s9kr(iters: list[str]) -> dict:
    """Download `iters` adapters (adapter_config.json + adapter_model.safetensors) at NLA_REVISION into the HF cache
    on the weights volume, which is the snapshot directory load_nla reads. Nothing is deleted or overwritten."""
    from huggingface_hub import hf_hub_download

    VOLUMES["/weights"].reload()
    done, skipped = [], []
    for it in iters:
        target = Path(NLA_SNAPSHOT, "av_rl_adapters", it)
        if (target / "adapter_model.safetensors").exists():
            skipped.append(it)
            continue
        for fn in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download(NLA_REPO, f"av_rl_adapters/{it}/{fn}", revision=NLA_REVISION, cache_dir="/weights/hf/hub")
        if not (target / "adapter_model.safetensors").exists():
            raise RuntimeError(f"{it}: download did not land in {target}")
        done.append(it)
        VOLUMES["/weights"].commit()
    return {"downloaded": done, "already_present": skipped, "now_on_volume": _volume_iters()}


# ============================================================================ text (H100)

@app.function(image=gpu_image, gpu="H100", **GPU)
def s5nt(rows: list[str], instruction: str) -> dict:
    """TextNLA: base Qwen3.6-27B reads the prompt text (the item's token ids decoded) under the NLA warm-start
    instruction (exp 06 `data/nla_instruction.txt`, passed in; checked against the volume's nla_meta.yaml)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lib29 import strip_analysis
    from mats.readers.nla import Sampling
    from mats.textnla import nla_instruction

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    meta_instr = nla_instruction()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    smp = Sampling()

    def chat(text: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": instruction.replace("{text}", text)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def generate(prompts: list[str], k: int) -> list[list[str]]:
        ntok = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: ntok[i])
        batches, cur = [], []
        for i in order:
            if cur and ((len(cur) + 1) * ntok[i] * k > TEXT_TOKEN_BUDGET or len(cur) >= TEXT_MAX_PROMPTS):
                batches.append(cur)
                cur = []
            cur.append(i)
        batches.append(cur)
        out: list[list[str]] = [[] for _ in prompts]
        for idx in batches:
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=smp.max_new_tokens, num_return_sequences=k, do_sample=True,
                                     temperature=smp.temperature, top_p=smp.top_p, top_k=smp.top_k, pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for j, i in enumerate(idx):
                out[i] = dec[j * k:(j + 1) * k]
        return out

    Path(OUT).mkdir(parents=True, exist_ok=True)
    out_all = {}
    for row in rows:
        path = f"{OUT}/{OUT_FILE('gen', CKPT_TEXT, row)}"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items, _, sel, _, _, mine = _load(row)
        sel_raw = generate([chat(tok.decode(items[i]["ids"])) for i in sel], K_SEL)
        mine_raw = generate([chat(tok.decode(items[i]["ids"])) for i in mine], K_MINE)
        res = {"feature": row, "ckpt": CKPT_TEXT, "layer": LAYER, "sampling": smp.__dict__,
               "instruction_matches_meta": instruction.strip() == meta_instr.strip(),
               "sel": {items[i]["id"]: strip_analysis(sel_raw[r][0]) for r, i in enumerate(sel)},
               "sel_raw": {items[i]["id"]: sel_raw[r][0] for r, i in enumerate(sel)},
               "mine": {items[i]["id"]: [t.strip() for t in mine_raw[r]] for r, i in enumerate(mine)},
               "wall_s": round(time.time() - t0, 1)}
        Path(path).write_text(json.dumps(res, ensure_ascii=False))
        VOLUMES["/results"].commit()
        out_all[row] = {"n_sel": len(sel), "n_mine": len(mine)}
    return {"rows": out_all, "instruction_matches_meta": instruction.strip() == meta_instr.strip(), "wall_s": round(time.time() - t0, 1)}


# ============================================================================ gen + critic (H200)

def _critic_only():  # noqa: ANN202
    """The reconstructor alone (no AV): an NLA whose `reconstruct` works and whose `av` is None."""
    import yaml
    from transformers import AutoTokenizer

    from mats.readers import nla as NL

    root = Path(NLA_SNAPSHOT)
    meta = yaml.safe_load((root / "nla_meta.yaml").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(root / "av_base"))
    ar, head = NL._load_ar(root, "cuda")
    return NL.NLA(av=None, tok=tok, ar=ar, value_head=head, meta=meta, prompt_ids=[], marker=0, device="cuda")


def _load_av(ckpts: list[str]):  # noqa: ANN202
    """av_base + every adapter among `ckpts` loaded at once (switch with set_adapter / no_adapter) + the critic."""
    from mats.readers import nla as NL

    adapters = [adapter_path(c) for c in ckpts if adapter_path(c) is not None]
    return NL.load_nla(adapter=adapters or None, load_ar=True)


def _recon(nla, X, y, groups, texts: list[str]) -> dict:  # noqa: ANN001
    import torch

    from lib29 import frozen_probe_auroc
    from mats.readers.nla import fve, predict_mean_baseline

    R = nla.reconstruct(texts, batch=16).float()
    d_true = X[y == 1].mean(0) - X[y == 0].mean(0)
    d_rec = R[y == 1].mean(0) - R[y == 0].mean(0)
    base = predict_mean_baseline(X)
    return {"auroc": frozen_probe_auroc(X, R, y, groups),
            "class_direction_cos": float(torch.nn.functional.cosine_similarity(d_rec, d_true, dim=0)),
            "fve": float(fve(R, X, base).mean()), "fve_baseline": base}


@app.function(image=gpu_image, gpu="H200", **GPU)
def d3wq(ckpts: list[str], rows: list[str]) -> dict:
    import torch

    from lib29 import readout_text, strip_expl_tail

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    ckpts = checkpoint_order(ckpts)
    av_ckpts = [c for c in ckpts if c != CKPT_TEXT]
    missing = [c for c in av_ckpts if adapter_path(c) and not Path(NLA_SNAPSHOT, adapter_path(c), "adapter_model.safetensors").exists()]
    if missing:
        return {"error": f"adapters not on the volume: {missing}; run --stage fetch first", "on_volume": _volume_iters()}
    nla = _load_av(av_ckpts) if av_ckpts else _critic_only()
    t_load = round(time.time() - t0, 1)
    out_all: dict = {}
    for ckpt in ckpts:
        for row in rows:
            gen_path = Path(OUT, OUT_FILE("gen", ckpt, row))
            rec_path = Path(OUT, OUT_FILE("recon", ckpt, row))
            if rec_path.exists():
                out_all[f"{ckpt}/{row}"] = {"skipped": True}
                continue
            items, X, sel, y, groups, mine = _load(row)
            Xs = X[sel]
            if ckpt == CKPT_TEXT:
                if not gen_path.exists():
                    out_all[f"{ckpt}/{row}"] = {"error": f"{gen_path} missing: run --stage text first"}
                    continue
                gen = json.loads(gen_path.read_text())
            elif gen_path.exists():
                gen = json.loads(gen_path.read_text())
            else:
                ctx = _base_ctx(nla) if ckpt == CKPT_BASE else _active(nla, ckpt)
                with ctx:
                    sel_raw = nla.verbalize_raw(Xs, k=K_SEL, batch=AV_BATCH)
                    mine_raw = nla.verbalize_raw(X[mine], k=K_MINE, batch=AV_BATCH)
                gen = {"feature": row, "ckpt": ckpt, "layer": LAYER, "adapter": adapter_path(ckpt),
                       "sel": {items[i]["id"]: readout_text(sel_raw[r][0], ckpt) for r, i in enumerate(sel)},
                       "sel_raw": {items[i]["id"]: sel_raw[r][0] for r, i in enumerate(sel)},
                       "mine": {items[i]["id"]: [strip_expl_tail(t) for t in mine_raw[r]] for r, i in enumerate(mine)},
                       "wall_s": round(time.time() - t0, 1)}
                gen_path.write_text(json.dumps(gen, ensure_ascii=False))
                VOLUMES["/results"].commit()
            texts = [gen["sel"][items[i]["id"]] or "(empty)" for i in sel]
            rec = {"feature": row, "ckpt": ckpt, "n": len(sel), "critic": "ar_reconstructor (final AR, paired with iter_000400 everywhere)",
                   **_recon(nla, Xs, y, groups, texts),
                   "examples": {items[i]["id"]: gen["sel"][items[i]["id"]] for i in sel[:4]}}
            rec_path.write_text(json.dumps(rec, indent=1))
            VOLUMES["/results"].commit()
            out_all[f"{ckpt}/{row}"] = {"auroc": rec["auroc"], "fve": rec["fve"]}
    return {"cells": out_all, "t_load_s": t_load, "wall_s": round(time.time() - t0, 1)}


def _base_ctx(nla):  # noqa: ANN001, ANN202
    """av_base alone: PeftModel.disable_adapter() when adapters are loaded (nla.no_adapter), else nothing to do."""
    from contextlib import nullcontext

    return nla.no_adapter() if hasattr(nla.av, "disable_adapter") else nullcontext()


class _active:
    """Context manager: make `ckpt`'s adapter the active one (restores nothing; the next cell sets its own)."""

    def __init__(self, nla, ckpt: str) -> None:  # noqa: ANN001
        self.nla, self.ckpt = nla, ckpt

    def __enter__(self) -> None:
        self.nla.set_adapter(self.ckpt)

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        return None


@app.local_entrypoint()
def main(stage: str = "gen", ckpts: str = "", features: str = "", iters: str = "") -> None:
    RES.mkdir(exist_ok=True)
    rows = features.split(",") if features else list(ROWS)
    if stage == "fetch":
        r = s9kr.remote(iters.split(",") if iters else list(ALL_ITERS))
    elif stage == "text":
        r = s5nt.remote(rows, EXP06_INSTRUCTION.read_text(encoding="utf-8"))
    elif stage == "gen":
        if not ckpts:
            raise SystemExit("--ckpts is required for stage gen (e.g. av_base,textnla or iter_000300)")
        r = d3wq.remote(ckpts.split(","), rows)
    else:
        raise SystemExit(f"unknown stage {stage!r}: fetch | text | gen")
    (RES / f"{stage}_{(ckpts or 'all').replace(',', '+')}.json").write_text(json.dumps(r, indent=1, default=str))
    print(stage, json.dumps(r, indent=1, default=str)[:3000])
