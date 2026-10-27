"""Exp 26 (app `k2vd`): a broad Oracle Lens layer sweep on the exp 22 public bank. For each of exp 22's eight rows
(selection split, 160 items) the read-token activation is captured at every layer of olens.SWEEP_LAYERS (the trained
grid 20..60 step 4 plus 42, adapter disabled) and, per layer L, four numbers are taken:
  readable[L]    grouped 5-fold probe AUROC on the true activations at L
  ol_prompted[L] Oracle Lens prompted AUROC at L with exp 22's frozen phrasings (max of the mean-of-three and the fixed label)
  carries[L]     the Oracle Lens k=1 free readout at L, rebuilt by the NLA reconstructor and scored by exp 14's frozen
                 probe against the TRUE layer-42 activations (exp 25's `ol_readout` measure)
  names[L]       share of those readouts in which the row's strict state regex fires (exp 23's phrasing)
Stages, strictly one after the other (run26.sh):
  r6tp (H100): capture + readable + prompted + readouts  -> acts26_{row}.pt, ol26_{row}_L{L}.json, sweep26a_{row}.json
  x9ae (H200): critic rebuilds + naming + merge         -> sweep26_{row}.json
    cd experiments/26_layer_sweep && modal run --detach modal26.py --stage ol
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, OUT22, VOLUMES, gpu_image
from lib26 import LAYER, MARKED, ROWS, sweep_layers

app = modal.App("k2vd")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)
LAYERS = sweep_layers()
CAPTURE_BATCH = 8
PROMPTED_BATCH = 64            # modal22 j6rb: max(64, 2 * n_candidates); 8 candidates -> 8 activations per forward
READOUT_BATCH = 32


def _rows_items(row: str):  # noqa: ANN202
    """exp 22's selection items of `row` in file order, labelled as modal22's `_sel` (1 = the marked value)."""
    from lib26 import select_items

    items = [it for it in json.load(open(f"{OUT22}/items_base.json")) if it["feature"] == row]
    return select_items(items, MARKED[row])


def _acts_path(row: str) -> str:
    return f"{OUT}/acts26_{row}.pt"


def _readout_path(row: str, L: int) -> str:
    return f"{OUT}/ol26_{row}_L{L}.json"


def _load_acts(row: str, sel: list[dict]) -> dict:
    """{"ids", "acts": {L: [N, d] bf16 cpu}} from the volume, checked against the selection items."""
    import torch

    blob = torch.load(_acts_path(row), weights_only=False)
    assert blob["ids"] == [it["id"] for it in sel], f"{row}: acts26 ids differ from items_base's selection split"
    return blob


@app.function(image=gpu_image, gpu="H100", **GPU)
def r6tp(jobs: list[tuple]) -> dict:
    """jobs = [(row, templates, selected_ids, fixed_id)] with exp 22's frozen Oracle Lens phrasings of the row."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from feature_spec22 import FEATURES
    from lib26 import prompted_auroc, prompted_candidates, prompted_logodds
    from mats import capture as C
    from mats import probes as P
    from mats.readers.olens import WSBENCH_SAMPLING, load_olens

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    out_all = {}
    for row, templates, selected_ids, fixed_id in jobs:
        path = f"{OUT}/sweep26a_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        t_row = time.time()
        sel, y, groups = _rows_items(row)
        if Path(_acts_path(row)).exists():
            blob = _load_acts(row, sel)
        else:
            seqs = [it["ids"] for it in sel]
            with ol.base():                                                     # adapter off: the plain subject model
                cap = C.capture(model, seqs, LAYERS, [[-1]] * len(seqs), batch_size=CAPTURE_BATCH)
            blob = {"ids": [it["id"] for it in sel],
                    "acts": {L: cap[L][:, 0].contiguous().to(torch.bfloat16).cpu() for L in LAYERS}}
            torch.save(blob, _acts_path(row))                                   # bf16 is the model's own precision: lossless
            VOLUMES["/results"].commit()
        vals = FEATURES[row]["values"]
        cands = prompted_candidates(templates, vals)
        readable, prompted, prompted_detail = {}, {}, {}
        for L in LAYERS:
            X = blob["acts"][L].float()
            readable[L] = float(P.cv_auroc(X, y, groups)["auroc"])
            lp = ol.logprob_of_strings(X, L, cands, batch=max(PROMPTED_BATCH, 2 * len(cands)))
            lo = prompted_logodds(lp)
            d = prompted_auroc({t["id"]: lo[k].tolist() for k, t in enumerate(templates)}, selected_ids, fixed_id, y)
            prompted[L], prompted_detail[L] = d["auroc"], d
            rp = _readout_path(row, L)
            if not Path(rp).exists():
                g = ol.verbalize(X, L, k=1, sampling=WSBENCH_SAMPLING, batch=READOUT_BATCH)
                Path(rp).write_text(json.dumps({"feature": row, "layer": L, "ol": {it["id"]: g[r][0] for r, it in enumerate(sel)}}))
                VOLUMES["/results"].commit()
            print(f"{row} L{L}: readable {readable[L]:.3f} prompted {prompted[L]:.3f} ({time.time() - t_row:.0f}s)", flush=True)
        res = {"feature": row, "n": len(sel), "layers": LAYERS, "readable": readable, "ol_prompted": prompted,
               "ol_prompted_detail": prompted_detail, "templates": templates, "wall_s": round(time.time() - t_row, 1)}
        Path(path).write_text(json.dumps(res, indent=1))
        VOLUMES["/results"].commit()
        out_all[row] = {"readable": readable, "ol_prompted": prompted}
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def x9ae(rows: list[str]) -> dict:
    import torch

    from lib26 import frozen_probe_auroc, merge, names_rate
    from mats.readers import nla as NL

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    out_all = {}
    for row in rows:
        path = f"{OUT}/sweep26_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        a_path = Path(f"{OUT}/sweep26a_{row}.json")
        missing = [p for p in [a_path, Path(_acts_path(row))] + [Path(_readout_path(row, L)) for L in LAYERS] if not p.exists()]
        if missing:
            out_all[row] = {"error": f"{missing[0]} missing: run --stage ol first"}
            continue
        t_row = time.time()
        sel, y, groups = _rows_items(row)
        blob = _load_acts(row, sel)
        X42 = blob["acts"][LAYER].float()                                      # the TRUE layer-42 activations
        carries, names, examples = {}, {}, {}
        for L in LAYERS:
            ol = json.loads(Path(_readout_path(row, L)).read_text())["ol"]
            texts = [ol[it["id"]] or "(empty)" for it in sel]
            R = nla.reconstruct(texts, batch=16).float()
            carries[L] = frozen_probe_auroc(X42, R, y, groups)
            names[L] = names_rate(texts, row)
            examples[L] = {it["id"]: ol[it["id"]] for it in sel[:3]}
            print(f"{row} L{L}: carries {carries[L]:.3f} names {names[L]:.3f} ({time.time() - t_row:.0f}s)", flush=True)
        stage1 = json.loads(a_path.read_text())
        res = merge(stage1, {"carries": carries, "names": names, "examples": examples,
                             "wall_s_recon": round(time.time() - t_row, 1)})
        Path(path).write_text(json.dumps(res, indent=1))
        VOLUMES["/results"].commit()
        out_all[row] = {"carries": carries, "names": names}
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str = "ol", features: str = "") -> None:
    from lib26 import ol_templates

    RES.mkdir(exist_ok=True)
    rows = features.split(",") if features else list(ROWS)
    if stage == "ol":
        frozen = json.loads((HERE.parent / "22_public_bank" / "templates_frozen.json").read_text())
        r = r6tp.remote([(row, *ol_templates(frozen, row)) for row in rows])
    elif stage == "recon":
        r = x9ae.remote(rows)
    else:
        raise SystemExit(f"unknown stage {stage!r}: use ol or recon")
    print(stage, json.dumps(r, indent=1, default=str)[:3000])
