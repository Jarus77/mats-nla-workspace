"""Exp 18 Modal stages (app `h5tq`, opaque function names). Stage 1 = the item bank:
  instr1 p7vd (H100): capture L36/42/44/52 at the read token (all 200 items per row); on the 160 selection items:
                      probe AUROC (readable), J-enrichment of the class mean, and the USED push (centred class-mean
                      push at r11 +/- vs 20 isotropic nulls, next-token log-odds on the row's declared token set)
                      -> acts18_{row}.pt, instr1_{row}.json
  instr2 c4nx (H200): NLA reconstructor on the true 32-token spans (sayable by span) and on the fixed statement
                      pair (sayable by statement)                                          -> instr2_{row}.json
  build  w3ke (CPU): synthetic rows + stance row (fresh wikitext paragraphs) + the two exp 07 control rows
                     (language FR/DE, quote state), tokenised, split 20+20 mine / 80+80 select, frozen with a
                     sha256 -> /results/18_prediction_test/items_base.json, results/build_info.json

    cd experiments/18_prediction_test && modal run modal18.py --stage build
"""
import hashlib
import json
import time
from pathlib import Path

import modal

from env import BASIS, OUT, OUT07, OUT08, OUT13, VOLUMES, build_image, gpu_image
from feature_spec18 import (FEATURES, FIXED, K_NULL, LAYER, LAYERS_CAPTURE, N_GEN, N_MINE_PER_VALUE, N_SELECT_PER_VALUE,
                            REL_DOSE, SEED_NULL)

app = modal.App("h5tq")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def _load_07_items_build():  # noqa: ANN202
    import importlib.util
    import sys

    def load(name: str, path: str):  # noqa: ANN202
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    saved = sys.modules.get("feature_spec")
    sys.modules["feature_spec"] = load("feature_spec07", "/root/ext07/feature_spec.py")
    try:
        return load("items_build07", "/root/ext07/items_build.py")
    finally:
        if saved is not None:
            sys.modules["feature_spec"] = saved


def assign_splits(items: list[dict], rng, n_mine: int, n_sel: int) -> list[dict]:  # noqa: ANN001
    """Per value: shuffle, first n_mine -> mine, next n_sel -> select (exp 07's rule)."""
    out, by_val = [], {}
    for it in items:
        by_val.setdefault(it["value"], []).append(it)
    for _, lst in sorted(by_val.items()):
        rng.shuffle(lst)
        out += [{**it, "split": "mine" if k < n_mine else "select"} for k, it in enumerate(lst[: n_mine + n_sel])]
    return out


@app.function(image=build_image, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def w3ke() -> dict:
    import collections
    import random

    from transformers import AutoTokenizer

    import items_build18 as IB

    t0 = time.time()
    VOLUMES["/results"].reload()
    IB07 = _load_07_items_build()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    s1 = IB07.load_02("s1_data")
    paras = IB07.fresh_paragraphs(s1)
    used = set()
    for path in (f"{OUT07}/items_base.json", f"{OUT07}/items_extra.json", f"{OUT08}/items.json", f"{OUT13}/items_base.json"):
        try:
            used |= {IB07._norm(it["text"]) for it in json.load(open(path)) if it.get("text")}
        except FileNotFoundError:
            pass
    paras = [p for p in paras if IB07._norm(p) not in used]
    rng = random.Random(IB.SEED)
    NM, NS = N_MINE_PER_VALUE, N_SELECT_PER_VALUE
    items: list[dict] = []
    for f, builder in IB.BUILDERS.items():
        items += assign_splits(builder(rng, N_GEN), rng, NM, NS)
    items += assign_splits(IB.build_stance_word(paras[0:400], rng), rng, NM, NS)
    items = [{**it, "ids": tok(it["text"], add_special_tokens=False)["input_ids"]} for it in items]
    lang = [{**it, "feature": "ij2_lang"} for it in IB07.build_lang(tok, rng)]          # exp 07 generator, fresh draw
    quote = IB.length_band([{**it, "feature": "bk_quote"} for it in IB07.build_quote(tok, paras[400:2400], rng)], 40, 250)
    items += assign_splits(lang, rng, NM, NS) + assign_splits(quote, rng, NM, NS)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    blob = json.dumps(items)
    Path(f"{OUT}/items_base.json").write_text(blob)
    VOLUMES["/results"].commit()
    summ = collections.Counter((it["feature"], it["split"], it["value"]) for it in items)
    by_val = {}
    for it in items:
        by_val.setdefault(it["feature"] + "|" + it["value"], []).append(len(it["ids"]))
    toklen = {f: sorted(len(it["ids"]) for it in items if it["feature"] == f) for f in {it["feature"] for it in items}}
    return {"n_items": len(items), "n_paragraphs": len(paras), "sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "counts": {"|".join(map(str, k)): v for k, v in sorted(summ.items())},
            "token_len_q10_q50_q90": {f: [L[len(L) // 10], L[len(L) // 2], L[9 * len(L) // 10]] for f, L in toklen.items()},
            "median_token_len_by_value": {k: sorted(v)[len(v) // 2] for k, v in sorted(by_val.items())},
            "wall_s": round(time.time() - t0, 1)}


GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)
ALL_ROWS = list(FEATURES)


def _items(row: str) -> list[dict]:
    return [it for it in json.load(open(f"{OUT}/items_base.json")) if it["feature"] == row]


def _sel(items: list[dict]):  # noqa: ANN202
    import numpy as np

    idx = [i for i, it in enumerate(items) if it["split"] == "select"]
    y = np.array([items[i]["value"] == FEATURES[items[i]["feature"]]["values"][0] for i in idx], int)  # 1 = marked value
    return idx, y, np.array([items[i]["cluster"] for i in idx])


@app.function(image=gpu_image, gpu="H100", **GPU)
def p7vd(rows: list[str]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lib18 import band, centred_push, class_mean_direction, iso_nulls, set_logodds
    from mats import capture as C
    from mats import jspace as JS
    from mats import probes as P
    from mats.capture import decoder_blocks

    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    basis = JS.load_basis_file(BASIS)
    blk = decoder_blocks(model)[LAYER]
    state: dict = {"H": None}

    def hook(_m, _i, o):  # noqa: ANN001, ANN202
        if state["H"] is None:
            return o
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] == 1:
            return o
        h = h.clone()
        h[:, -1] = state["H"].to(h.dtype)
        return (h, *o[1:]) if isinstance(o, tuple) else h

    out_all = {}
    for row in rows:
        path = f"{OUT}/instr1_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items = _items(row)
        seqs = [it["ids"] for it in items]
        acts = C.capture(model, seqs, list(LAYERS_CAPTURE), [[-1]] * len(seqs), batch_size=8)
        acts = {L: a[:, 0].contiguous() for L, a in acts.items()}
        torch.save({"ids": [it["id"] for it in items], "acts": acts}, f"{OUT}/acts18_{row}.pt")
        idx, y, groups = _sel(items)
        X = acts[LAYER][idx].float()
        probe = float(P.cv_auroc(X, y, groups)["auroc"])
        v = class_mean_direction(X, y)
        jenr = float(JS.enrichment(basis, v, 1536))
        res = {"feature": row, "n_select": len(idx), "readable": probe, "jenr": jenr}
        toks = FEATURES[row]["next_tokens"]
        if toks:
            set_ids = [i for i in range(len(tok)) if any(t in tok.decode([i]) for t in toks)]
            sel_seqs = [seqs[i] for i in idx]
            hd = blk.register_forward_hook(hook)
            try:
                def lo_for(H):  # noqa: ANN001, ANN202
                    outs = []
                    for s in range(0, len(sel_seqs), 64):
                        rows_ = list(range(s, min(len(sel_seqs), s + 64)))
                        enc = tok.pad({"input_ids": [sel_seqs[r] for r in rows_]}, return_tensors="pt")
                        state["H"] = H[rows_].cuda()
                        with torch.no_grad():
                            lg = model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(),
                                       logits_to_keep=1).logits[:, -1].float()
                        outs.append(set_logodds(torch.log_softmax(lg, -1), set_ids).cpu())
                    state["H"] = None
                    return torch.cat(outs)
                real = lo_for(centred_push(X, v, REL_DOSE, +1)) - lo_for(centred_push(X, v, REL_DOSE, -1))
                nulls = []
                for u in iso_nulls(X.shape[1], K_NULL, SEED_NULL):
                    nulls.append(float((lo_for(centred_push(X, u, REL_DOSE, +1)) - lo_for(centred_push(X, u, REL_DOSE, -1))).mean()))
            finally:
                hd.remove()
            b = band(float(real.mean()), nulls)
            res.update({"used_shift": b["stat"], "used_z": b["z"], "used_null_sd": b["null_sd"], "n_set_tokens": len(set_ids),
                        "used_sign_note": "+ = the marked value raises the declared set's log-odds"})
        else:
            res.update({"used_shift": None, "used_z": None, "used_note": "no declared next-token set: not used by construction"})
        Path(path).write_text(json.dumps(res, indent=1))
        VOLUMES["/results"].commit()
        out_all[row] = res
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def c4nx(rows: list[str]) -> dict:
    import torch
    from transformers import AutoTokenizer

    from lib18 import fill
    from mats.readers import nla as NL

    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    out_all = {}
    for row in rows:
        path = f"{OUT}/instr2_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items = _items(row)
        blob = torch.load(f"{OUT}/acts18_{row}.pt", weights_only=False)
        assert blob["ids"] == [it["id"] for it in items]
        idx, y, _ = _sel(items)
        true = blob["acts"][LAYER][idx].float()
        spans = [tok.decode(items[i]["ids"][-32:]) for i in idx]
        rec = nla.reconstruct(spans, batch=16).float()
        a, b = FEATURES[row]["values"]                      # a = marked value (y == 1)
        stmts = [fill(FIXED["nla"][row], a), fill(FIXED["nla"][row], b)]
        rs = nla.reconstruct(stmts, batch=2).float()
        yb = torch.as_tensor(y == 1)
        d_true = true[yb].mean(0) - true[~yb].mean(0)
        u = d_true / d_true.norm().clamp_min(1e-9)
        d_rec = rec[yb].mean(0) - rec[~yb].mean(0)
        res = {"feature": row, "span": float((d_rec @ u) / d_true.norm().clamp_min(1e-9)),
               "span_cos": float(torch.nn.functional.cosine_similarity(d_rec, d_true, dim=0)),
               "stmt": float(((rs[0] - rs[1]) @ u) / d_true.norm().clamp_min(1e-9)),
               "stmt_cos": float(torch.nn.functional.cosine_similarity(rs[0] - rs[1], d_true, dim=0)),
               "statements": stmts}
        torch.save({"ids": blob["ids"], "sel_idx": idx, "rec_span": rec, "rec_stmt": rs}, f"{OUT}/recon18_{row}.pt")
        Path(path).write_text(json.dumps(res, indent=1))
        VOLUMES["/results"].commit()
        out_all[row] = res
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str = "build", features: str = "") -> None:
    RES.mkdir(exist_ok=True)
    rows = features.split(",") if features else ALL_ROWS
    if stage == "build":
        r = w3ke.remote()
        (RES / "build_info.json").write_text(json.dumps(r, indent=1, default=str))
    else:
        r = {"instr1": p7vd, "instr2": c4nx}[stage].remote(rows)
        for row, res in r["rows"].items():
            if not res.get("skipped"):
                (RES / f"{stage}18_{row}.json").write_text(json.dumps(res, indent=1, default=float))
    print(stage, json.dumps(r, indent=1, default=str)[:3000])
