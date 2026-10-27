"""Exp 28 Modal stage (app `y4hb`, function `q8nr`, H200): the statement shift with eight paraphrase pairs per row.

For every row in /results/28_stmt_curve/rows28.json (uploaded by run28.sh with paraphrases28.json) the stage loads the
row's selection items and layer-42 activations from the volume paths the record names (exp 22: acts22_{row}.pt; exp
19: acts19_{row}.pt; exp 18: acts18_{row}.pt; exps 12/13: 07_template_mining/acts_{row}.pt and 13_new_rows/acts_{row}.pt
with the items inside gen_ol_{row}.json), takes the true class direction d (marked minus unmarked mean over the
selection split) and its gap, rebuilds the fixed statement pair and the eight paraphrase pairs through the NLA critic
(`load_nla(load_ar=True)`, `reconstruct`), and reports the shift ((rec(A) - rec(B)) . d_hat) / gap for each pair,
exactly as modal22's c7lx computes `stmt`. It also rebuilds the true last-32-token spans and reports `span` (exp 14's
frozen-probe AUROC, lib22.frozen_probe_auroc), `span_gap` (modal22's ratio) and `readable` (grouped 5-fold probe AUROC
on the true activations) for every row, so the rows whose source did not record them get them and the others get a
consistency check. One file per row, /results/28_stmt_curve/stmt28_{experiment}_{row}.json, committed after each
row; rows whose file exists are skipped, so a relaunch resumes.

    cd experiments/28_stmt_curve && modal run --detach modal28.py --stage stmt [--rows 22_py_string,13_tb_paren]
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, gpu_image

app = modal.App("y4hb")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)
RECON_BATCH = 16


def _out(key: str) -> str:
    return f"{OUT}/stmt28_{key}.json"


@app.function(image=gpu_image, gpu="H200", **GPU)
def q8nr(keys: list[str]) -> dict:
    import torch
    from transformers import AutoTokenizer

    from lib22 import frozen_probe_auroc
    from lib28 import (SPAN_TOKENS, acts_from_blob, class_direction, items_from, select_split, shift_summary,
                       statement_shift)
    from mats import probes as P
    from mats.readers import nla as NL

    t0 = time.time()
    VOLUMES["/results"].reload()
    rows = {r["key"]: r for r in json.load(open(f"{OUT}/rows28.json"))}
    paras = json.load(open(f"{OUT}/paraphrases28.json"))["rows"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    out_all = {}
    for key in keys:
        path = _out(key)
        if Path(path).exists():
            out_all[key] = {"skipped": True}
            continue
        t1 = time.time()
        row = rows[key]
        pairs = [list(row["fixed"])] + [list(p) for p in paras[key]["pairs"]]
        items = items_from(json.load(open(row["items"]["vol"])), row["items"]["fmt"], row["feature"])
        blob = torch.load(row["acts"]["vol"], weights_only=False)
        acts = acts_from_blob(blob, row["acts"]["layer"], [it["id"] for it in items])
        idx, y, groups = select_split(items, row["values"][0])
        true = acts[idx]
        d, gap = class_direction(true, y)
        rs = nla.reconstruct([s for p in pairs for s in p], batch=RECON_BATCH).float()
        shifts = [statement_shift(rs[2 * i], rs[2 * i + 1], d) for i in range(len(pairs))]
        cos = [float(torch.nn.functional.cosine_similarity(rs[2 * i] - rs[2 * i + 1], d, dim=0)) for i in range(len(pairs))]
        spans = [tok.decode(items[i]["ids"][-SPAN_TOKENS:]) for i in idx]
        rec = nla.reconstruct(spans, batch=RECON_BATCH).float()
        d_rec, _ = class_direction(rec, y)
        res = {"experiment": row["experiment"], "feature": row["feature"], "key": key, "type": row["type"], "stated": row["stated"],
               "n_select": len(idx), "n_marked": int(y.sum()), "gap": gap,
               "shift": shift_summary(shifts[0], shifts[1:]), "shifts_para": shifts[1:], "stmt_cos_fixed": cos[0],
               "stmt_cos_para": cos[1:], "statements": pairs, "registers": paras[key].get("registers"),
               "span": frozen_probe_auroc(true, rec, y, groups), "span_gap": statement_shift(d_rec, torch.zeros_like(d_rec), d),
               "readable": float(P.cv_auroc(true, y, groups)["auroc"]), "prior": row.get("prior", {}),
               "wall_s": round(time.time() - t1, 1)}
        torch.save({"ids": blob["ids"], "sel_idx": idx, "rec_span": rec, "rec_stmt": rs, "statements": pairs}, f"{OUT}/recon28_{key}.pt")
        Path(path).write_text(json.dumps(res, indent=1))
        VOLUMES["/results"].commit()
        out_all[key] = {k: res[k] for k in ("shift", "span", "readable", "wall_s")}
        print(key, json.dumps(out_all[key]), flush=True)
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str = "stmt", rows: str = "") -> None:
    RES.mkdir(exist_ok=True)
    keys = rows.split(",") if rows else [r["key"] for r in json.loads((RES / "rows28.json").read_text())]
    if stage != "stmt":
        raise SystemExit(f"unknown stage {stage!r}; the only stage is 'stmt'")
    r = q8nr.remote(keys)
    print(json.dumps(r, indent=1, default=str)[:4000])
