"""Exp 20 (app `v7qd`): what does the NLA reconstructor keep? On the exp 19 natural rows (selection split, 160 items,
layer-42 activations already on the volume), rebuild each item's vector from
  (a) its true last w tokens, w in 4/8/16/32/64/128     (how much structure the window carries)
  (b) the same 32 tokens in random order                (vocabulary kept, order broken)
  (c) Oracle Lens's own free readout of the vector      (what the reader says)
  (d) the NLA's own explanation of the vector
and score each rebuild with exp 14's frozen-probe AUROC; also (e) the model's own activation on the window alone. Stages:
  r3hx (H100): Oracle Lens readouts, k=1 at layer 42, on the selection split, and the BASE model's own layer-42
               activation when it reads the last w tokens alone (the image of the window map, no reconstructor)
                                                                              -> ol_sel_{row}.json, spanacts20_{row}.pt
  w9kc (H200): NLA explanations (k=1) and all reconstructions + AUROCs        -> nla_sel_{row}.json, recon20_{row}.json
    cd experiments/20_ar_structure && modal run --detach modal20.py --stage ol
"""
import json
import random
import time
from pathlib import Path

import modal

from env import OUT, OUT13, OUT19, VOLUMES, gpu_image

app = modal.App("v7qd")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LAYER = 42
ROWS = ("py_string", "js_string", "latex_math", "py_arg", "json_depth", "html_depth", "md_heading", "md_list")
MARKED = {"py_string": "inside", "js_string": "inside", "latex_math": "inside", "py_arg": "third", "json_depth": "deep",
          "html_depth": "deep", "md_heading": "deep", "md_list": "nested",
          "tb_tablecol": "first", "tb_paren": "inside"}            # exp 13's headline rows, same protocol
SRC13 = ("tb_tablecol", "tb_paren")
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)


def _load(row: str):  # noqa: ANN202
    import numpy as np
    import torch

    base, acts_name = (OUT13, f"acts_{row}.pt") if row in SRC13 else (OUT19, f"acts19_{row}.pt")
    items = [it for it in json.load(open(f"{base}/items_base.json")) if it["feature"] == row]
    blob = torch.load(f"{base}/{acts_name}", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    A = blob["acts"][LAYER] if isinstance(blob["acts"], dict) else blob["acts"]
    if A.dim() == 3:
        A = A[:, 0]
    sel = [i for i, it in enumerate(items) if it["split"] == "select"]
    y = np.array([items[i]["value"] == MARKED[row] for i in sel], int)
    groups = np.array([items[i]["cluster"] for i in sel])
    return items, A[sel].float(), sel, y, groups


@app.function(image=gpu_image, gpu="H100", **GPU)
def r3hx(rows: list[str]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lib20 import WINDOWS, windows
    from mats import capture as C
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
        span_acts = {}
        with ol.base():                                   # the model's own activation when it reads the span alone
            for w in WINDOWS:
                seqs = [windows(items[i]["ids"])[w] for i in sel]
                span_acts[w] = C.capture(model, seqs, [LAYER], [[-1]] * len(seqs), batch_size=16)[LAYER][:, 0].contiguous().cpu()
        torch.save({"sel_ids": [items[i]["id"] for i in sel], "span_acts": span_acts}, f"{OUT}/spanacts20_{row}.pt")
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
        sp = torch.load(f"{OUT}/spanacts20_{row}.pt", weights_only=False)
        assert sp["sel_ids"] == [items[i]["id"] for i in sel]
        for w, A in sp["span_acts"].items():               # no reconstructor: the model itself on the window
            auroc[f"model_span_{w}"] = frozen_probe_auroc(X, A.float(), y, groups)
            d_a = A.float()[y == 1].mean(0) - A.float()[y == 0].mean(0)
            cos[f"model_span_{w}"] = float(torch.nn.functional.cosine_similarity(d_a, d_true, dim=0))
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
