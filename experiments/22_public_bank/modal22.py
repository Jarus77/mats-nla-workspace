"""Exp 19 Modal stages (app `k3rw`, opaque function names), the exp 18 pipeline on the natural bank. The bank is
built on the laptop by build22.py (public package sources and arXiv e-prints on this machine) and uploaded to
/results/22_public_bank/items_base.json with `modal volume put`; there is no build stage here.
  instr1 a4wn (H100): capture L36/42/44/52 at the read token; probe AUROC (readable), J-enrichment, USED push -> acts22_{row}.pt, instr1_{row}.json
  instr2 c7lx (H200): NLA reconstructor on true 32-token spans (span) and the fixed statement pair (stmt)  -> instr2_{row}.json
  gen_ol d2kq / gen_nla h9vt: free readouts on the mining split; score_ol j6rb / score_nla n3yc: prompted log-odds on the selection split.

    cd experiments/22_public_bank && modal run modal22.py --stage instr1
"""
import json
import time
from pathlib import Path

import modal

from env import BASIS, OUT, VOLUMES, gpu_image
from feature_spec22 import FEATURES, FIXED, K_NULL, LAYER, LAYERS_CAPTURE, REL_DOSE, SEED_NULL

app = modal.App("p8sm")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"


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
def a4wn(rows: list[str]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lib22 import band, centred_push, class_mean_direction, iso_nulls, set_logodds
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
        torch.save({"ids": [it["id"] for it in items], "acts": acts}, f"{OUT}/acts22_{row}.pt")
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
def c7lx(rows: list[str]) -> dict:
    import torch
    from transformers import AutoTokenizer

    from lib22 import fill
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
        blob = torch.load(f"{OUT}/acts22_{row}.pt", weights_only=False)
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
        torch.save({"ids": blob["ids"], "sel_idx": idx, "rec_span": rec, "rec_stmt": rs}, f"{OUT}/recon22_{row}.pt")
        Path(path).write_text(json.dumps(res, indent=1))
        VOLUMES["/results"].commit()
        out_all[row] = res
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


# ============================================================================ readers (prereg §6 step 4)
K_READOUTS = 8
OL_LAYERS = (42, 36, 44, 52)


def _strip_expl(t: str) -> str:
    return t.split("</br>")[0].strip() if "</br>" in t else t.strip()


def _acts(row: str):  # noqa: ANN202
    import torch

    items = _items(row)
    blob = torch.load(f"{OUT}/acts22_{row}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    return items, blob["acts"]


@app.function(image=gpu_image, gpu="H100", **GPU)
def d2kq(rows: list[str]) -> dict:
    """OLens free readouts (k=8, layer 42) and J-lens top-50 on the mining split -> gen_ol_{row}.json."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats.readers.jlens import load_jlens
    from mats.readers.olens import WSBENCH_SAMPLING, load_olens

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    jl = load_jlens(ol.model, tok, layers=[LAYER])
    out_all = {}
    for row in rows:
        path = f"{OUT}/gen_ol_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items, acts = _acts(row)
        mine = [i for i, it in enumerate(items) if it["split"] == "mine"]
        g = ol.verbalize(acts[LAYER][mine], LAYER, k=K_READOUTS, sampling=WSBENCH_SAMPLING, batch=32)
        jtop = jl.readout(acts[LAYER][mine], LAYER, k=50)
        out = {"feature": row, "items": items, "ol_gens": {str(LAYER): {items[i]["id"]: g[r] for r, i in enumerate(mine)}},
               "jlens_top50_L42": {items[i]["id"]: jtop[r] for r, i in enumerate(mine)}, "wall_s": round(time.time() - t0, 1)}
        Path(path).write_text(json.dumps(out))
        VOLUMES["/results"].commit()
        out_all[row] = {"n_mine": len(mine)}
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def h9vt(rows: list[str]) -> dict:
    """NLA free readouts (k=8) on the mining split -> gen_nla_{row}.json."""
    import torch

    from mats.readers import nla as NL

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    out_all = {}
    for row in rows:
        path = f"{OUT}/gen_nla_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items, acts = _acts(row)
        mine = [i for i, it in enumerate(items) if it["split"] == "mine"]
        raw = nla.verbalize_raw(acts[LAYER][mine], k=K_READOUTS, batch=64)
        out = {"feature": row, "nla_gens": {items[i]["id"]: [_strip_expl(t) for t in raw[r]] for r, i in enumerate(mine)},
               "wall_s": round(time.time() - t0, 1)}
        Path(path).write_text(json.dumps(out))
        VOLUMES["/results"].commit()
        out_all[row] = {"n_mine": len(mine)}
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


def _cands_for(templates: list[dict], vals: list[str]) -> list[str]:
    from feature_spec import fill

    return [fill(t, v) for t in templates for v in vals]


def _score_generic(lp_fn, row: str, templates: list[dict], items, acts_L, sel) -> dict:  # noqa: ANN001
    vals = FEATURES[row]["values"]
    lp = lp_fn(acts_L[sel], _cands_for(templates, vals))
    lo = (lp[:, 0::2] - lp[:, 1::2]).T
    return {"lo": {t["id"]: lo[k].tolist() for k, t in enumerate(templates)},
            "y": [int(items[i]["value"] == vals[0]) for i in sel], "ids": [items[i]["id"] for i in sel]}


@app.function(image=gpu_image, gpu="H100", **GPU)
def j6rb(jobs: list[tuple]) -> dict:
    """jobs = [(row, ol_templates, jlens_sets)]: OLens at 42/36/44/52 and J-lens at 42 on the selection split."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats.readers.jlens import load_jlens, token_variants
    from mats.readers.olens import load_olens

    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    jl = load_jlens(ol.model, tok, layers=[LAYER])
    out_all = {}
    for row, templates, jsets in jobs:
        path = f"{OUT}/score_ol_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items, acts = _acts(row)
        sel = [i for i, it in enumerate(items) if it["split"] == "select"]
        res = {}
        for L in OL_LAYERS:
            fn = lambda A, cands, L=L: ol.logprob_of_strings(A, L, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
            res[f"ol@{L}"] = _score_generic(fn, row, templates, items, acts[L], sel)
        vals = FEATURES[row]["values"]
        y = [int(items[i]["value"] == vals[0]) for i in sel]
        lo = {}
        for js in jsets:
            a, b = token_variants(tok, js["words"][vals[0]]), token_variants(tok, js["words"][vals[1]])
            if a and b:
                lo[js["id"]] = jl.paired_logodds(acts[LAYER][sel], LAYER, a, b).tolist()
        res[f"jlens@{LAYER}"] = {"lo": lo, "y": y, "ids": [items[i]["id"] for i in sel]}
        out = {"feature": row, "scores": res, "wall_s": round(time.time() - t0, 1)}
        Path(path).write_text(json.dumps(out))
        VOLUMES["/results"].commit()
        out_all[row] = {"n_select": len(sel), "n_templates": len(templates)}
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def n3yc(jobs: list[tuple]) -> dict:
    """jobs = [(row, nla_templates)]: NLA prompted log-odds at 42 on the selection split."""
    from mats.readers import nla as NL

    t0 = time.time()
    VOLUMES["/results"].reload()
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    out_all = {}
    for row, templates in jobs:
        path = f"{OUT}/score_nla_{row}.json"
        if Path(path).exists():
            out_all[row] = {"skipped": True}
            continue
        items, acts = _acts(row)
        sel = [i for i, it in enumerate(items) if it["split"] == "select"]
        fn = lambda A, cands: NL.logprob_of_strings_batch(nla, A, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
        out = {"feature": row, "scores": {f"nla@{LAYER}": _score_generic(fn, row, templates, items, acts[LAYER], sel)},
               "wall_s": round(time.time() - t0, 1)}
        Path(path).write_text(json.dumps(out))
        VOLUMES["/results"].commit()
        out_all[row] = {"n_select": len(sel), "n_templates": len(templates)}
    return {"rows": out_all, "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str = "instr1", features: str = "") -> None:
    RES.mkdir(exist_ok=True)
    rows = features.split(",") if features else ALL_ROWS
    if stage == "build":
        r = w3ke.remote()
        (RES / "build_info.json").write_text(json.dumps(r, indent=1, default=str))
    elif stage in ("gen_ol", "gen_nla"):
        r = {"gen_ol": d2kq, "gen_nla": h9vt}[stage].remote(rows)
    elif stage in ("score_ol", "score_nla"):
        cand = json.loads((RES / "candidates.json").read_text())
        if stage == "score_ol":
            r = j6rb.remote([(row, cand["ol"][row], cand["jlens"][row]) for row in rows])
        else:
            r = n3yc.remote([(row, cand["nla"][row]) for row in rows])
    else:
        r = {"instr1": a4wn, "instr2": c7lx}[stage].remote(rows)
        for row, res in r["rows"].items():
            if not res.get("skipped"):
                (RES / f"{stage}22_{row}.json").write_text(json.dumps(res, indent=1, default=float))
    print(stage, json.dumps(r, indent=1, default=str)[:3000])
