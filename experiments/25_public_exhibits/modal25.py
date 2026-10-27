"""Exp 25 (app `n7qe`): exp 20's reconstructor protocol on the exp 22 public bank. For each of exp 22's eight rows
(selection split, 160 items, layer-42 activations already on the volume from exp 22's instr1), rebuild each item's
vector with the NLA reconstructor ("critic") from
  (a) its true last 32 and last 128 tokens          (span_32, span_128)
  (b) the same 32 tokens in random order, seed 25   (bag_32)
  (c) Oracle Lens's own free readout, k = 1         (ol_readout)
  (d) the NLA's own explanation, k = 1              (nla_readout)
and score each rebuild with exp 14's frozen-probe AUROC against the true activations, plus class-direction cosine.
The `ol_readout` number is the structure-aware "exhibits" measure exp 20 used in place of the judge (rescore25.py).
Stages, strictly one after the other (run25.sh):
  b8fn (H100): Oracle Lens readouts, k=1 at layer 42, on the selection split            -> ol_sel_{row}.json
  m4ks (H200): NLA explanations (k=1), the reconstructions and the AUROCs               -> nla_sel_{row}.json, recon25_{row}.json
    cd experiments/25_public_exhibits && modal run --detach modal25.py --stage ol
"""
import json
import random
import time
from pathlib import Path

import modal

from env import OUT, OUT22, VOLUMES, gpu_image
from feature_spec22 import LAYER
from lib25 import BAG_SEED, MARKED, ROWS, WINDOWS

app = modal.App("n7qe")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)


def _load(row: str):  # noqa: ANN202
    """exp 22's items and true layer-42 activations of `row`, restricted to the selection split, labelled as
    modal22's `_sel` (1 = the marked value, FEATURES[row]["values"][0])."""
    import torch

    from lib25 import select_split

    items = [it for it in json.load(open(f"{OUT22}/items_base.json")) if it["feature"] == row]
    blob = torch.load(f"{OUT22}/acts22_{row}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items], f"{row}: acts22 ids differ from items_base"
    idx, y, groups = select_split(items, MARKED[row])
    return items, blob["acts"][LAYER][idx].float(), idx, y, groups


@app.function(image=gpu_image, gpu="H100", **GPU)
def b8fn(rows: list[str]) -> dict:
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
def m4ks(rows: list[str]) -> dict:
    import torch
    from transformers import AutoTokenizer

    from lib25 import explanation_text, frozen_probe_auroc, shuffle_tokens, windows
    from mats.readers import nla as NL

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    out_all = {}
    for row in rows:
        path = f"{OUT}/recon25_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        ol_path = Path(f"{OUT}/ol_sel_{row}.json")
        if not ol_path.exists():
            out_all[row] = {"error": f"{ol_path} missing: run --stage ol first"}
            continue
        items, X, sel, y, groups = _load(row)
        rng = random.Random(BAG_SEED)
        raw = nla.verbalize_raw(X, k=1, batch=64)
        expl = [explanation_text(r[0]) for r in raw]
        Path(f"{OUT}/nla_sel_{row}.json").write_text(json.dumps({"feature": row, "nla": {items[i]["id"]: expl[r] for r, i in enumerate(sel)}}))
        ol = json.loads(ol_path.read_text())["ol"]
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
    r = {"ol": b8fn, "recon": m4ks}[stage].remote(rows)
    print(stage, json.dumps(r, indent=1, default=str)[:3000])
