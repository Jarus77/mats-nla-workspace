"""Exp 13 Modal stages (app `tq7`; functions s0-s4 so the dashboard shows no task names):
s0 = build (CPU), s1 = OL/J-lens readouts (H100), s2 = NLA readouts (H200), s3/s4 = prompted scoring.
Same stage logic as exp 07's tm_modal.py, pointed at /results/13_new_rows and the exp 13 features.

    modal run modal13.py --stage build
    modal run modal13.py --stage gen_ol  [--features a,b]
    modal run modal13.py --stage gen_nla [--features a,b]
    modal run modal13.py --stage score   [--features a,b] [--readers ol,nla]
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, OUT07, OUT08, VOLUMES, build_image, gpu_image
from feature_spec import FEATURES, K_READOUTS, NLA_LAYER, OL_LAYERS

app = modal.App("tq7")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=2)


# ============================================================================ build (CPU)
@app.function(image=build_image, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def s0() -> dict:
    import collections
    import random

    from transformers import AutoTokenizer

    import items_build as IB
    from feature_spec import N_MINE_PER_VALUE as NM
    from feature_spec import N_SELECT_PER_VALUE as NS

    IB07 = _load_07_items_build()

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    s1 = IB07.load_02("s1_data")
    paras = IB07.fresh_paragraphs(s1)  # 02's cleaner, 02's paragraphs excluded
    used = set()
    for path in (f"{OUT07}/items_base.json", f"{OUT07}/items_extra.json"):
        try:
            used |= {IB07._norm(it["text"]) for it in json.load(open(path)) if it.get("text")}
        except FileNotFoundError:
            pass
    try:
        used |= {IB07._norm(it["text"]) for it in json.load(open(f"{OUT08}/items.json")) if it.get("text")}
    except FileNotFoundError:
        pass
    paras = [p for p in paras if IB07._norm(p) not in used]
    print("fresh paragraphs after 07/08 exclusion", len(paras), flush=True)
    rng = random.Random(IB.SEED)
    blk = {"sent": paras[0:2000], "paren": paras[2000:4000], "wl": paras[4000:4600],
           "qa": paras[4600:5000], "col": paras[5000:5600]}
    items: list[dict] = []
    items += IB.assign_splits(IB.build_sentcount(blk["sent"], rng), rng, NM, NS)
    items += IB.assign_splits(IB.build_paren(blk["paren"], rng), rng, NM, NS)
    items += IB.assign_splits(IB.build_wordsleft(blk["wl"], rng), rng, NM, NS)
    items += IB.assign_splits(IB.build_qa(blk["qa"], rng), rng, NM, NS)
    items += IB.assign_splits(IB.build_tablecol(rng, 2 * (NM + NS) + 20), rng, NM, NS)
    items += IB.assign_splits(IB.build_codeline(rng, 2 * (NM + NS) + 20), rng, NM, NS)
    items += IB.assign_splits(IB.build_listdepth(rng, 2 * (NM + NS) + 20), rng, NM, NS)
    items = IB.tokenise(tok, items)
    col = IB07.build_col(tok, s1, blk["col"], rng)  # exp 07's generator (02's build_wrap recipe)
    col = [{**it, "id": it["id"].replace("col_", "colf_", 1), "feature": "nj1_col_fresh"} for it in col]
    items += [{**it, "split": "select"} for it in IB.assign_splits(col, rng, 0, NS)]
    Path(OUT).mkdir(parents=True, exist_ok=True)
    json.dump(items, open(f"{OUT}/items_base.json", "w"))
    VOLUMES["/results"].commit()
    summ = collections.Counter((it["feature"], it["split"], it["value"]) for it in items)
    info = {"n_items": len(items), "n_paragraphs": len(paras),
            "counts": {"|".join(map(str, k)): v for k, v in sorted(summ.items())}}
    print(json.dumps(info, indent=1))
    return info


def _load_07_items_build():  # noqa: ANN202
    """Exp 07's items_build with exp 07's feature_spec (both shipped to /root/ext07), loaded under
    private module names so they do not collide with this experiment's modules of the same name."""
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


def _load_items(feature: str) -> list[dict]:
    return [it for it in json.load(open(f"{OUT}/items_base.json")) if it["feature"] == feature]


def _strip_expl(t: str) -> str:
    return t.split("</br>")[0].strip() if "</br>" in t else t.strip()


# ============================================================================ gen_ol (H100)
@app.function(image=gpu_image, gpu="H100", **GPU)
def s1(feature: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C
    from mats.readers.jlens import load_jlens
    from mats.readers.olens import WSBENCH_SAMPLING, load_olens

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    if Path(f"{OUT}/gen_ol_{feature}.json").exists():
        return json.load(open(f"{OUT}/gen_ol_{feature}.json"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    items = _load_items(feature)
    seqs = [it["ids"] for it in items]
    with ol.base():
        acts = C.capture(ol.model, seqs, list(OL_LAYERS), [[-1]] * len(seqs), batch_size=8)
    acts = {L: a[:, 0].contiguous() for L, a in acts.items()}
    torch.save({"ids": [it["id"] for it in items], "acts": acts}, f"{OUT}/acts_{feature}.pt")
    VOLUMES["/results"].commit()
    mine = [i for i, it in enumerate(items) if it["split"] == "mine"]
    gens, jtop = {}, []
    if mine:
        for L in (42,):  # free readouts at the primary layer only (40/44 shown interchangeable in exp 12)
            g = ol.verbalize(acts[L][mine], L, k=K_READOUTS, sampling=WSBENCH_SAMPLING, batch=32)
            gens[L] = {items[i]["id"]: g[r] for r, i in enumerate(mine)}
        jl = load_jlens(ol.model, tok, layers=[42])
        jtop = jl.readout(acts[42][mine], 42, k=50)
    out = {"feature": feature, "items": items, "ol_gens": gens,
           "jlens_top50_L42": {items[i]["id"]: jtop[r] for r, i in enumerate(mine)},
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/gen_ol_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return out


# ============================================================================ gen_nla (H200)
@app.function(image=gpu_image, gpu="H200", **GPU)
def s2(feature: str) -> dict:
    import torch

    from mats.readers import nla as NL

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    if Path(f"{OUT}/gen_nla_{feature}.json").exists():
        return json.load(open(f"{OUT}/gen_nla_{feature}.json"))
    items = _load_items(feature)
    blob = torch.load(f"{OUT}/acts_{feature}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    mine = [i for i, it in enumerate(items) if it["split"] == "mine"]
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    raw = nla.verbalize_raw(blob["acts"][NLA_LAYER][mine], k=K_READOUTS, batch=64) if mine else []
    out = {"feature": feature,
           "nla_gens": {items[i]["id"]: [_strip_expl(t) for t in raw[r]] for r, i in enumerate(mine)},
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/gen_nla_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return out


# ============================================================================ scoring
def _cands_for(templates: list[dict], vals: list[str]) -> list[str]:
    from feature_spec import fill

    return [fill(t, v) for t in templates for v in vals]


def _selection(feature: str):  # noqa: ANN202
    import torch

    items = _load_items(feature)
    blob = torch.load(f"{OUT}/acts_{feature}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    return items, blob["acts"], [i for i, it in enumerate(items) if it["split"] == "select"]


def _score_generic(lp_fn, feature: str, templates: list[dict], items, acts_L, sel) -> dict:  # noqa: ANN001
    vals = FEATURES[feature]["values"]
    lp = lp_fn(acts_L[sel], _cands_for(templates, vals))
    lo = (lp[:, 0::2] - lp[:, 1::2]).T
    return {"lo": {t["id"]: lo[k].tolist() for k, t in enumerate(templates)},
            "y": [int(items[i]["value"] == vals[0]) for i in sel], "ids": [items[i]["id"] for i in sel]}


@app.function(image=gpu_image, gpu="H100", **GPU)
def s3(feature: str, templates: list[dict], jsets: list[dict]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats.readers.jlens import load_jlens, token_variants
    from mats.readers.olens import load_olens

    t0 = time.time()
    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    items, acts, sel = _selection(feature)
    res = {}
    for L in OL_LAYERS:
        fn = lambda A, cands, L=L: ol.logprob_of_strings(A, L, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
        res[f"ol@{L}"] = _score_generic(fn, feature, templates, items, acts[L], sel)
    jl = load_jlens(ol.model, tok, layers=list(OL_LAYERS))
    vals = FEATURES[feature]["values"]
    y = [int(items[i]["value"] == vals[0]) for i in sel]
    idl = [items[i]["id"] for i in sel]
    for L in OL_LAYERS:
        lo = {}
        for js in jsets:
            a, b = token_variants(tok, js["words"][vals[0]]), token_variants(tok, js["words"][vals[1]])
            if a and b:
                lo[js["id"]] = jl.paired_logodds(acts[L][sel], L, a, b).tolist()
        res[f"jlens@{L}"] = {"lo": lo, "y": y, "ids": idl}
    out = {"feature": feature, "scores": res, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/score_ol_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return out


@app.function(image=gpu_image, gpu="H200", **GPU)
def s4(feature: str, templates: list[dict]) -> dict:
    from mats.readers import nla as NL

    t0 = time.time()
    VOLUMES["/results"].reload()
    items, acts, sel = _selection(feature)
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    fn = lambda A, cands: NL.logprob_of_strings_batch(nla, A, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
    out = {"feature": feature,
           "scores": {f"nla@{NLA_LAYER}": _score_generic(fn, feature, templates, items, acts[NLA_LAYER], sel)},
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/score_nla_{feature}.json", "w"))
    VOLUMES["/results"].commit()
    return out


# ============================================================================ entrypoint
@app.local_entrypoint()
def main(stage: str = "gen_ol", features: str = "", readers: str = "ol,nla") -> None:
    feats = features.split(",") if features else list(FEATURES)
    RES.mkdir(exist_ok=True)
    if stage == "build":
        (RES / "build_info.json").write_text(json.dumps(s0.remote(), indent=1, default=str))
    elif stage in ("gen_ol", "gen_nla"):
        fn = s1 if stage == "gen_ol" else s2
        for r in fn.map(feats, return_exceptions=True):
            _save(r, stage)
    elif stage == "score":
        cand = json.loads((RES / "candidates.json").read_text())
        rd = readers.split(",")
        hs = [("score_ol", s3.spawn(f, cand["ol"][f], cand["jlens"][f])) for f in feats] if "ol" in rd else []
        hs += [("score_nla", s4.spawn(f, cand["nla"][f])) for f in feats] if "nla" in rd else []
        for kind, h in hs:
            try:
                _save(h.get(), kind)
            except Exception as e:  # noqa: BLE001
                _save(e, kind)


def _save(r, kind: str) -> None:  # noqa: ANN001
    if isinstance(r, Exception):
        print(f"[{kind}] FAILED: {r!r}")
        return
    (RES / f"{kind}_{r['feature']}.json").write_text(json.dumps(r, default=str))
    print(f"[{kind}] {r['feature']} done in {r.get('wall_s')} s")
