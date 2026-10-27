"""Exp 20 (app `v7qd`): what does the NLA reconstructor keep? On the exp 19 natural rows (selection split, 160 items,
layer-42 activations already on the volume), rebuild each item's vector from
  (a) its true last w tokens, w in 4/8/16/32/64/128     (how much structure the window carries)
  (b) the same 32 tokens in random order                (vocabulary kept, order broken)
  (c) Oracle Lens's own free readout of the vector      (what the reader says)
  (d) the NLA's own explanation of the vector
and score each rebuild with exp 14's frozen-probe AUROC. Stages:
  r3hx (H100): Oracle Lens readouts, k=1 at layer 42, on the selection split   -> ol_sel_{row}.json
  w9kc (H200): NLA explanations (k=1) and all reconstructions + AUROCs        -> nla_sel_{row}.json, recon20_{row}.json
    cd experiments/20_ar_structure && modal run --detach modal20.py --stage ol
"""
import json
import random
import time
from pathlib import Path

import modal

from env import OUT, OUT19, VOLUMES, gpu_image

app = modal.App("v7qd")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LAYER = 42
ROWS = ("py_string", "js_string", "latex_math", "py_arg", "json_depth", "html_depth", "md_heading", "md_list")
MARKED = {"py_string": "inside", "js_string": "inside", "latex_math": "inside", "py_arg": "third", "json_depth": "deep",
          "html_depth": "deep", "md_heading": "deep", "md_list": "nested"}
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)


def _load(row: str):  # noqa: ANN202
    import numpy as np
    import torch

    items = [it for it in json.load(open(f"{OUT19}/items_base.json")) if it["feature"] == row]
    blob = torch.load(f"{OUT19}/acts19_{row}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    sel = [i for i, it in enumerate(items) if it["split"] == "select"]
    y = np.array([items[i]["value"] == MARKED[row] for i in sel], int)
    groups = np.array([items[i]["cluster"] for i in sel])
    return items, blob["acts"][LAYER][sel].float(), sel, y, groups


@app.function(image=gpu_image, gpu="H100", **GPU)
def r3hx(rows: list[str]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats.readers.olens import WSBENCH_SAMPLING, load_olens

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    out_all = {}
    for row in rows:
        path = f"{OUT}/ol_sel_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items, X, sel, _, _ = _load(row)
        g = ol.verbalize(X, LAYER, k=1, sampling=WSBENCH_SAMPLING, batch=32)
        Path(path).write_text(json.dumps({"feature": row, "ol": {items[i]["id"]: g[r][0] for r, i in enumerate(sel)}}))
        VOLUMES["/results"].commit()
        out_all[row] = {"n": len(sel)}
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def w9kc(rows: list[str]) -> dict:
    import torch
    from transformers import AutoTokenizer

    from lib20 import WINDOWS, explanation_text, frozen_probe_auroc, shuffle_tokens, windows
    from mats.readers import nla as NL

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    out_all = {}
    for row in rows:
        path = f"{OUT}/recon20_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items, X, sel, y, groups = _load(row)
        rng = random.Random(20)
        raw = nla.verbalize_raw(X, k=1, batch=64)
        expl = [explanation_text(r[0]) for r in raw]
        Path(f"{OUT}/nla_sel_{row}.json").write_text(json.dumps({"feature": row, "nla": {items[i]["id"]: expl[r] for r, i in enumerate(sel)}}))
        ol = json.load(open(f"{OUT}/ol_sel_{row}.json"))["ol"]
        texts = {f"span_{w}": [tok.decode(windows(items[i]["ids"])[w]) for i in sel] for w in WINDOWS}
        texts["bag_32"] = [tok.decode(shuffle_tokens(items[i]["ids"][-32:], rng)) for i in sel]
        texts["ol_readout"] = [ol[items[i]["id"]] or "(empty)" for i in sel]
        texts["nla_readout"] = expl
        auroc, cos = {}, {}
        d_true = X[y == 1].mean(0) - X[y == 0].mean(0)
        for cond, tx in texts.items():
            R = nla.reconstruct(tx, batch=16).float()
            auroc[cond] = frozen_probe_auroc(X, R, y, groups)
            d_rec = R[y == 1].mean(0) - R[y == 0].mean(0)
            cos[cond] = float(torch.nn.functional.cosine_similarity(d_rec, d_true, dim=0))
        res = {"feature": row, "n": len(sel), "auroc": auroc, "class_direction_cos": cos,
               "examples": {items[i]["id"]: {"ol": ol[items[i]["id"]], "nla": expl[r], "span_32": texts["span_32"][r]} for r, i in enumerate(sel[:6])}}
        Path(path).write_text(json.dumps(res, indent=1))
        VOLUMES["/results"].commit()
        out_all[row] = auroc
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str = "ol", features: str = "") -> None:
    RES.mkdir(exist_ok=True)
    rows = features.split(",") if features else list(ROWS)
    r = {"ol": r3hx, "recon": w9kc}[stage].remote(rows)
    print(stage, json.dumps(r, indent=1, default=str)[:3000])
