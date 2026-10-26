"""Exp 17 Modal stages (app `b7kx`), opaque function names; CLI keys are descriptive:
  mc_ol   q2vn (H100): OLens@42 + J-lens@42 multi-candidate log-probs on the exp 13 selection items -> mc_ol_{row}.pt
  mc_nla  f8rt (H200): NLA multi-candidate log-probs                                                 -> mc_nla_{row}.pt
  vocab   y4lm (CPU):  unembedding readout of the exp 15 directions                                  -> vocab17_{row}.json
  xs_ol   n6wd (H100), xs_nla k9zc (H200): cross-scoring with Guiv's bank (skip when inputs absent)  -> xs_{ol,nla}_{row}.pt
  noop    g1ps (CPU):  input check

    cd experiments/17_harden && RFM_MAXC=1 modal run modal17.py --stage mc_ol --features tb_tablecol,tb_paren
"""
import json
import os
import time
from pathlib import Path

import modal

from env import OUT, SRC13, SRC15, VOLUMES, gpu_image, templates_path
from spec17 import CLASS_SETS, LAYER, MC_TEMPLATES, ROWS

app = modal.App("b7kx")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GPU = dict(volumes=VOLUMES, timeout=3 * 3600, memory=131072, single_use_containers=True,
           max_containers=int(os.environ.get("RFM_MAXC", "1")))  # GPU cap: 1 by default, never more than 2


def _done(path: str) -> bool:
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    return Path(path).exists()


def _save(path: str, obj) -> None:  # noqa: ANN001
    import torch

    torch.save(obj, path)
    VOLUMES["/results"].commit()


def _selection(row: str):  # noqa: ANN202
    """The 160 exp 13 selection items of a row with their L42 activations, parenthesis items filtered."""
    import numpy as np
    import torch

    from lib17 import paren_item_ok, true_index

    items = json.load(open(f"{SRC13}/gen_ol_{row}.json"))["items"]
    blob = torch.load(f"{SRC13}/acts_{row}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    keep, excluded = [], []
    for i, it in enumerate(items):
        if it["split"] != "select":
            continue
        if row == "tb_paren" and not paren_item_ok(it["text"]):
            excluded.append(it["id"])
            continue
        keep.append(i)
    sel = [items[i] for i in keep]
    meta = {"ids": [it["id"] for it in sel],
            "truth": np.array([true_index(row, it) for it in sel]),
            "y": np.array([it["value"] == ROWS[row]["values"][1] for it in sel], int),
            "clusters": [it["cluster"] for it in sel], "excluded": excluded}
    return meta, blob["acts"][LAYER][keep].float()


def _mc_stage(row: str, reader: str, lp_fn, jl=None, tok=None) -> dict:  # noqa: ANN001
    """Score every value-slot template of `reader` (and the J-lens when given) -> {"lp": {id: [N, C]}}."""
    import torch

    from lib17 import candidates
    from mats.readers.jlens import token_variants

    meta, acts = _selection(row)
    lp = {}
    for t in MC_TEMPLATES[row][reader]:
        lp[t["id"]] = lp_fn(acts, candidates(t))
    if jl is not None:
        for t in MC_TEMPLATES[row]["jlens"]:
            cols = []
            for words in candidates(t):
                ids = token_variants(tok, words)
                if not ids:
                    raise ValueError(f"no single-token variant for {words}")
                cols.append(jl.scores(acts, LAYER, ids=ids).max(1).values)
            lp[t["id"]] = torch.stack(cols, 1)
    return {**meta, "lp": lp}


@app.function(image=gpu_image, gpu="H100", **GPU)
def q2vn(row: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats.readers.jlens import load_jlens
    from mats.readers.olens import load_olens

    t0 = time.time()
    path = f"{OUT}/mc_ol_{row}.pt"
    if _done(path):
        return {"feature": row, "skipped": True}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    jl = load_jlens(ol.model, tok, layers=[LAYER])
    fn = lambda A, cands: ol.logprob_of_strings(A, LAYER, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
    out = _mc_stage(row, "ol", fn, jl=jl, tok=tok)
    _save(path, out)
    return {"feature": row, "n": len(out["ids"]), "n_excluded": len(out["excluded"]),
            "templates": list(out["lp"]), "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def f8rt(row: str) -> dict:
    from mats.readers import nla as NL

    t0 = time.time()
    path = f"{OUT}/mc_nla_{row}.pt"
    if _done(path):
        return {"feature": row, "skipped": True}
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    fn = lambda A, cands: NL.logprob_of_strings_batch(nla, A, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
    out = _mc_stage(row, "nla", fn)
    _save(path, out)
    return {"feature": row, "n": len(out["ids"]), "n_excluded": len(out["excluded"]),
            "templates": list(out["lp"]), "wall_s": round(time.time() - t0, 1)}


TOKENS_OF_INTEREST = {"tb_tablecol": ["\n", " |", "|", " first", " fifth", " column"],
                      "tb_paren": [")", " )", " inside", " outside", " parenthesis"]}
CONTENT_WORDS = {"tb_tablecol": ["first", "fifth", "column"], "tb_paren": ["inside", "outside", "parenthesis"]}


@app.function(image=gpu_image, volumes=VOLUMES, timeout=1800, memory=24576, cpu=4)
def y4lm(row: str) -> dict:
    """Vocabulary readout of every exp 15 direction and null: logits = W_U @ ((1 + w_norm) * v)."""
    import glob

    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from lib17 import pick_unembedding, vocab_stats

    t0 = time.time()
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    path = f"{OUT}/vocab17_{row}.json"
    if Path(path).exists():
        return {"feature": row, "skipped": True}
    snap = sorted(glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*/model.safetensors.index.json"))[-1]
    wmap = json.load(open(snap))["weight_map"]
    key, shard, norm_shard = pick_unembedding(wmap)
    sdir = Path(snap).parent
    with safe_open(str(sdir / shard), framework="pt") as f:
        W = f.get_tensor(key).float()                                     # [V, d]
    gain = torch.ones(W.shape[1])
    if norm_shard is not None:
        with safe_open(str(sdir / norm_shard), framework="pt") as f:
            nk = next(k for k in f.keys() if k.endswith("norm.weight") and ("language_model.norm" in k or k == "model.norm.weight"))
            gain = 1.0 + f.get_tensor(nk).float()                        # Qwen3.5 RMSNorm: (1 + w) (D10)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    spec = torch.load(f"{SRC15}/spec15_{row}.pt", weights_only=False)
    ids_of_interest = {s: tok.encode(s, add_special_tokens=False) for s in TOKENS_OF_INTEREST[row]}

    def readout(v: torch.Tensor) -> dict:
        logits = (W @ (gain * v.float())).numpy()
        st = vocab_stats(logits, top_k=30)
        order = np.argsort(-logits)
        rank = {i: int(r) for r, i in enumerate(order)}
        return {"peak_ratio": st["peak_ratio"], "top_mass": st["top_mass"],
                "top_pos_tokens": [tok.decode([i]) for i, _ in st["top_pos"]],
                "top_pos": st["top_pos"], "top_neg": st["top_neg"],
                "top_neg_tokens": [tok.decode([i]) for i, _ in st["top_neg"]],
                "rank_of_interest": {s: min(rank[i] for i in ids) for s, ids in ids_of_interest.items() if ids}}

    out = {"feature": row, "unembedding": key, "gain": "1+w" if norm_shard else "none", "vocab": int(W.shape[0]),
           "content_words": CONTENT_WORDS[row],
           "dirs": {d: readout(v) for d, v in spec["dirs"].items()},
           "nulls": {k: readout(v) for k, v in spec["nulls"].items()},
           "wall_s": round(time.time() - t0, 1)}
    Path(path).write_text(json.dumps(out, indent=1, default=float))
    VOLUMES["/results"].commit()
    return {"feature": row, "unembedding": key, "gain": out["gain"], "wall_s": out["wall_s"],
            "arstmt_top10": out["dirs"]["arstmt_raw"]["top_pos_tokens"][:10],
            "peak": {d: round(e["peak_ratio"], 1) for d, e in out["dirs"].items()}}


# ============================================================================ cross-scoring with Guiv's bank
INPUTS = HERE / "inputs"


def _our_pairs(row: str, reader: str) -> list[list[str]]:
    """Our frozen phrasings (fixed label + selected), exp 16 shape."""
    t = json.loads(templates_path("13_new_rows").read_text())["templates"][reader][row]
    a, b = ROWS[row]["values"]
    return [[r["id"], r["candidates"][a], r["candidates"][b]] for r in [t["fixed_label"]] + t["selected"]]


def _our_jlens_words(row: str) -> dict:
    t = json.loads(templates_path("13_new_rows").read_text())["templates"]["jlens"][row]["fixed_label"]
    return t["words"]


def _xs_inputs(row: str):  # noqa: ANN202
    from lib17 import load_guiv_items

    ip = INPUTS / f"guiv_items_{row}.json"
    pp = INPUTS / "guiv_phrasings.json"
    if not ip.exists():
        return None, None, "no guiv items"
    items = load_guiv_items(ip, row)
    phr = json.loads(pp.read_text()) if pp.exists() else {}
    return items, phr, None


def _lo_pairs(lp_fn, acts, pairs: list[list[str]]):  # noqa: ANN001
    """{pair_id: [N] B - A} for a reader's string-scoring function."""
    import torch

    cands = [x for _, a, b in pairs for x in (a, b)]
    if not cands:
        return {}
    lp = lp_fn(acts, cands)
    return {pid: (lp[:, 2 * t + 1] - lp[:, 2 * t]) for t, (pid, _, _) in enumerate(pairs)}


@app.function(image=gpu_image, gpu="H100", **GPU)
def n6wd(row: str) -> dict:
    """Capture Guiv's items at L42, score OLens + J-lens: his items x {our, his} phrasings, our items x his."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lib17 import guiv_pairs
    from mats import capture as C
    from mats.readers.jlens import load_jlens
    from mats.readers.olens import load_olens

    t0 = time.time()
    path = f"{OUT}/xs_ol_{row}.pt"
    if _done(path):
        return {"feature": row, "skipped": True}
    items, phr, why = _xs_inputs(row)
    if items is None:
        return {"feature": row, "skipped": True, "reason": why}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    seqs = [tok.encode(it["text"], add_special_tokens=False) for it in items]
    with ol.base():
        his = C.capture(ol.model, seqs, [LAYER], [[-1]] * len(seqs), batch_size=8)[LAYER][:, 0].contiguous().float()
    ours_meta, ours = _selection(row)
    fn = lambda A, cands: ol.logprob_of_strings(A, LAYER, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
    his_pairs, our_pairs = guiv_pairs(phr, row, "ol"), _our_pairs(row, "ol")
    lo = {"his_x_ours": _lo_pairs(fn, his, our_pairs), "his_x_his": _lo_pairs(fn, his, his_pairs),
          "ours_x_his": _lo_pairs(fn, ours, his_pairs)}
    jl = load_jlens(ol.model, tok, layers=[LAYER])
    w = _our_jlens_words(row)
    a, b = ROWS[row]["values"]
    lo["jlens_his_x_ours"] = {"jlens:fixed": jl.paired_logodds(his, LAYER, w[a], w[b]).neg()}  # B - A convention
    _save(path, {"his_ids": [it["id"] for it in items], "his_y": [int(it["value"] == b) for it in items],
                 "his_clusters": [it["cluster"] for it in items], "his_acts42": his,
                 "our_ids": ours_meta["ids"], "our_y": ours_meta["y"].tolist(), "our_clusters": ours_meta["clusters"],
                 "pairs": {"ours": our_pairs, "his": his_pairs}, "lo": lo})
    return {"feature": row, "n_his": len(items), "n_his_pairs": len(his_pairs), "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def k9zc(row: str) -> dict:
    """NLA mirror of n6wd; reads Guiv's activations from the OL file (so xs_ol runs first)."""
    import torch

    from lib17 import guiv_pairs
    from mats.readers import nla as NL

    t0 = time.time()
    path = f"{OUT}/xs_nla_{row}.pt"
    if _done(path):
        return {"feature": row, "skipped": True}
    olp = f"{OUT}/xs_ol_{row}.pt"
    if not Path(olp).exists():
        return {"feature": row, "skipped": True, "reason": "xs_ol output absent; run xs_ol first"}
    items, phr, why = _xs_inputs(row)
    if items is None:
        return {"feature": row, "skipped": True, "reason": why}
    s = torch.load(olp, weights_only=False)
    assert s["his_ids"] == [it["id"] for it in items], "guiv items changed since xs_ol ran"
    _, ours = _selection(row)
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    fn = lambda A, cands: NL.logprob_of_strings_batch(nla, A, cands, batch=max(64, 2 * len(cands)))  # noqa: E731
    his_pairs, our_pairs = guiv_pairs(phr, row, "nla"), _our_pairs(row, "nla")
    lo = {"his_x_ours": _lo_pairs(fn, s["his_acts42"], our_pairs), "his_x_his": _lo_pairs(fn, s["his_acts42"], his_pairs),
          "ours_x_his": _lo_pairs(fn, ours, his_pairs)}
    _save(path, {"his_ids": s["his_ids"], "his_y": s["his_y"], "his_clusters": s["his_clusters"],
                 "our_ids": s["our_ids"], "our_y": s["our_y"], "our_clusters": s["our_clusters"],
                 "pairs": {"ours": our_pairs, "his": his_pairs}, "lo": lo})
    return {"feature": row, "n_his": len(items), "n_his_pairs": len(his_pairs), "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, volumes=VOLUMES, timeout=600, memory=8192, cpu=2)
def g1ps(row: str) -> dict:
    """Warms the image build, checks the inputs exist, and records the token length of every multi-candidate
    string (top-1 on summed log-probs is biased toward shorter candidates; the lengths go in the write-up)."""
    from transformers import AutoTokenizer

    from lib17 import candidates

    VOLUMES["/results"].reload()
    need = [f"{SRC13}/gen_ol_{row}.json", f"{SRC13}/acts_{row}.pt", f"{SRC15}/spec15_{row}.pt"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    lengths = {t["id"]: [len(tok.encode(c, add_special_tokens=False)) for c in candidates(t)]
               for reader in ("nla", "ol") for t in MC_TEMPLATES[row][reader]}
    Path(f"{OUT}").mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/cand_lengths_{row}.json").write_text(json.dumps(lengths))
    VOLUMES["/results"].commit()
    return {"feature": row, "missing": [p for p in need if not Path(p).exists()], "skipped": True, "cand_token_lengths": lengths,
            "guiv_inputs_present": {"items": (INPUTS / f"guiv_items_{row}.json").exists(),
                                    "phrasings": (INPUTS / "guiv_phrasings.json").exists()}}


STAGES = {"mc_ol": q2vn, "mc_nla": f8rt, "vocab": y4lm, "xs_ol": n6wd, "xs_nla": k9zc, "noop": g1ps}


@app.local_entrypoint()
def main(stage: str = "noop", features: str = "") -> None:
    feats = features.split(",") if features else list(ROWS)
    RES.mkdir(exist_ok=True)
    for r in STAGES[stage].map(feats, return_exceptions=True):
        if isinstance(r, Exception):
            print("ERROR", stage, r)
            continue
        print(stage, json.dumps(r, default=str)[:600])
        if not r.get("skipped"):
            (RES / f"{stage}17_{r['feature']}.json").write_text(json.dumps(r, indent=1, default=float))
