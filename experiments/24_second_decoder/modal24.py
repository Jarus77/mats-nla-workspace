"""Exp 24 (app `t6bn`): a second, independent decoder for what the readers' outputs carry. Exp 20 showed that both
readers' free readouts, fed through the NLA's own reconstructor, rebuild structural state (string / depth / column /
parenthesis) they never name; that critic was trained on NLA explanations. Here the same readouts (exp 20's
ol_sel_{row}.json / nla_sel_{row}.json, selection split, ten rows) go through the Oracle Lens reconstructor (OL-AR,
`mats.readers.olens_ar`, lifted from exp 14) at tags 44 (primary) and 40 (check), with a shuffled-readout control.
The OL-AR's output space is whitened and its moments are gated, so nothing is compared with a raw activation: the
reference for every row is the OL-AR's own rebuild of the true last-32-token span, and each condition is scored inside
that space three ways (frozen probe fit on the span_32 rebuilds; grouped CV probe on the condition alone; cosine of the
class-mean directions). Conditions: span_32, span_128, bag_32, ol_readout, ol_readout_shuffled, nla_readout,
nla_readout_shuffled. One stage:
  g2wq (H100): OL-AR rebuilds + scores per row                                 -> recon24_{row}.json
    cd experiments/24_second_decoder && modal run --detach modal24.py --stage recon
"""
import json
import random
import time
from pathlib import Path

import modal

from env import OUT, OUT13, OUT19, OUT20, VOLUMES, gpu_image

app = modal.App("t6bn")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ROWS = ("py_string", "js_string", "latex_math", "py_arg", "json_depth", "html_depth", "md_heading", "md_list",
        "tb_tablecol", "tb_paren")
MARKED = {"py_string": "inside", "js_string": "inside", "latex_math": "inside", "py_arg": "third", "json_depth": "deep",
          "html_depth": "deep", "md_heading": "deep", "md_list": "nested",
          "tb_tablecol": "first", "tb_paren": "inside"}            # exp 13's headline rows, same protocol
SRC13 = ("tb_tablecol", "tb_paren")
VARIANT = ("li", "ln")        # exp 14's calibrated (final_norm, head_norm) choice at 44 and 40 (results/calib14.json)
N_EXAMPLES = 4
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)


def _load(row: str):  # noqa: ANN202
    """Selection-split items of one row (exp 19's bank for the natural rows, exp 13's for the two tb rows), their
    indices, labels (marked value = 1) and cluster groups; exp 20's `_load` without the activations."""
    import numpy as np

    base = OUT13 if row in SRC13 else OUT19
    items = [it for it in json.load(open(f"{base}/items_base.json")) if it["feature"] == row]
    sel = [i for i, it in enumerate(items) if it["split"] == "select"]
    y = np.array([items[i]["value"] == MARKED[row] for i in sel], int)
    groups = np.array([items[i]["cluster"] for i in sel])
    return items, sel, y, groups


def _readouts(row: str, ids: list[str]) -> tuple[dict, dict]:
    """Exp 20's k=1 readouts of the selection items, keyed by item id; every selected item must have both."""
    ol = json.load(open(f"{OUT20}/ol_sel_{row}.json"))["ol"]
    nla = json.load(open(f"{OUT20}/nla_sel_{row}.json"))["nla"]
    missing = [i for i in ids if i not in ol or i not in nla]
    assert not missing, f"{row}: {len(missing)} selection items without an exp 20 readout, e.g. {missing[:3]}"
    return ol, nla


def _scores(rec: dict, y, groups) -> dict:  # noqa: ANN001
    from lib24 import CONDITIONS, class_direction_cos, frozen_probe_auroc
    from mats.probes import cv_auroc

    ref = rec["span_32"]
    return {"frozen_auroc": {c: frozen_probe_auroc(ref, rec[c], y, groups) for c in CONDITIONS},
            "cv_auroc": {c: float(cv_auroc(rec[c], y, groups)["auroc"]) for c in CONDITIONS},
            "class_direction_cos": {c: class_direction_cos(ref, rec[c], y) for c in CONDITIONS}}


@app.function(image=gpu_image, gpu="H100", **GPU)
def g2wq(rows: list[str]) -> dict:
    import torch

    from lib24 import CONDITIONS, LAYERS, LONG_SPAN_TOKENS, MAX_TEXT_TOKENS, SEED, SPAN_TOKENS, conditions
    from mats.readers.olens_ar import OLensAR

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    ar = OLensAR()
    ar.variant = VARIANT                                  # no gold activations here: exp 14's calibrated choice
    tok = ar.tok
    encode = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    out_all = {}
    for row in rows:
        path = f"{OUT}/recon24_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items, sel, y, groups = _load(row)
        ol, nla = _readouts(row, [items[i]["id"] for i in sel])
        rng = random.Random(SEED)
        per_item = [conditions(items[i]["ids"][-SPAN_TOKENS:], items[i]["ids"][-LONG_SPAN_TOKENS:],
                               ol[items[i]["id"]], nla[items[i]["id"]], encode, rng, MAX_TEXT_TOKENS) for i in sel]
        by_cond = {c: [p[c] for p in per_item] for c in CONDITIONS}
        layers = {}
        for L in LAYERS:
            rec = {c: ar.reconstruct(by_cond[c], L, batch=16).float() for c in CONDITIONS}
            layers[str(L)] = _scores(rec, y, groups)
            print(f"{row} L{L} {json.dumps(layers[str(L)]['frozen_auroc'])} {time.time() - t0:.0f}s", flush=True)
        res = {"feature": row, "n": len(sel), "variant": "/".join(ar.variant), "layers": layers,
               "examples": {items[i]["id"]: {"ol": ol[items[i]["id"]], "nla": nla[items[i]["id"]],
                                             "span_32": tok.decode(by_cond["span_32"][r])} for r, i in enumerate(sel[:N_EXAMPLES])}}
        Path(path).write_text(json.dumps(res, indent=1))
        VOLUMES["/results"].commit()
        out_all[row] = layers[str(LAYERS[0])]["frozen_auroc"]
    return {"rows": out_all, "variant": "/".join(ar.variant), "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str = "recon", features: str = "") -> None:
    RES.mkdir(exist_ok=True)
    rows = features.split(",") if features else list(ROWS)
    r = {"recon": g2wq}[stage].remote(rows)
    print(stage, json.dumps(r, indent=1, default=str)[:3000])
