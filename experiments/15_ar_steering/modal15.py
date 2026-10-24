"""Exp 15 Modal stages (app `vx4`):
  prep      (H200, NLA + its AR): directions, doses, nulls, NLA-AR span reconstructions -> spec15_{f}.pt
  behave    (H100, base Qwen):   steered next-token logits + 32-token greedy continuations for every
                                 condition, plus J-lens@42 prompted log-odds        -> behave15_{f}.pt
  score_nla (H200, NLA):         NLA prompted log-odds on every condition           -> score15_nla_{f}.pt
  score_ol  (H100, OLens@42):    OLens prompted log-odds on every condition         -> score15_ol_{f}.pt
  analyze   (CPU):               bands, z, tables                                   -> analysis15_{f}.json

    modal run modal15.py --stage prep --features tb_tablecol,tb_paren
    modal run modal15.py --stage behave      (one GPU)   | modal run modal15.py --stage score_nla (other GPU)
    modal run modal15.py --stage score_ol
    modal run modal15.py --stage analyze
Every stage skips a feature whose output already exists.
"""
import json
import time
from pathlib import Path

import modal

from env import BASIS, OUT, SRC13, SRC14, VOLUMES, gpu_image
from spec15 import DIR_NAMES, GEN_TOKENS, JLENS_WORDS, JREM_RANK, K_NULL, LAYER, RANKS, ROWS, SPAN_TOKENS

app = modal.App("vx4")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
GPU = dict(volumes=VOLUMES, timeout=4 * 3600, memory=131072, single_use_containers=True, max_containers=1)
BASE = "Qwen/Qwen3.6-27B"


# ============================================================================ shared
def _items(feature: str):  # noqa: ANN202
    """Exp 13 items (token ids, labels) and the L42 activations in item order."""
    import numpy as np
    import torch

    items = json.load(open(f"{SRC13}/gen_ol_{feature}.json"))["items"]
    blob = torch.load(f"{SRC13}/acts_{feature}.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items], "activation order != item order"
    h = blob["acts"][LAYER].float()
    y = np.array([it["value"] == ROWS[feature]["values"][1] for it in items], int)
    return items, h, y


def _pairs(feature: str, reader: str) -> list[list[str]]:
    """[id, A, B] phrasing pairs: exp 13's frozen fixed label plus the selected mined templates."""
    t = json.loads((HERE.parent / "13_new_rows" / "templates_frozen.json").read_text())["templates"][reader][feature]
    a, b = ROWS[feature]["values"]
    return [[r["id"], r["candidates"][a], r["candidates"][b]] for r in [t["fixed_label"]] + t["selected"]]


def _done(path: str) -> bool:
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    return Path(path).exists()


def _save(path: str, obj) -> None:  # noqa: ANN001
    import torch

    torch.save(obj, path)
    VOLUMES["/results"].commit()


# ============================================================================ prep (H200)
@app.function(image=gpu_image, gpu="H200", **GPU)
def prep(feature: str) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from lib15 import class_mean_direction, cosines, q90_dose, unit
    from mats import jspace as JS
    from mats import probes as P
    from mats.metrics import auroc_labels
    from mats.readers import nla as NL

    t0 = time.time()
    path = f"{OUT}/spec15_{feature}.pt"
    if _done(path):
        return {"feature": feature, "skipped": True}
    items, h, y = _items(feature)
    tok = AutoTokenizer.from_pretrained(BASE)
    basis = JS.load_basis_file(BASIS)
    r14 = torch.load(f"{SRC14}/recon14_nla_{feature}.pt", weights_only=False)
    assert r14["ids"] == [it["id"] for it in items]
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    spans = [tok.decode(it["ids"][-SPAN_TOKENS:]) for it in items]
    rec_span = nla.reconstruct(spans, batch=16).float()                 # [N, d]
    rec_expl = r14["recon"].float().mean(1)                               # [N, d] (k=2 readouts averaged)
    stmt = r14["stmt_recon"][0].float()                                   # fixed label: [2, d] (A, B)

    raw = {"cm": class_mean_direction(h, y), "arexpl": class_mean_direction(rec_expl, y),
           "arspan": class_mean_direction(rec_span, y), "arstmt": unit(stmt[1] - stmt[0])}
    dirs = {}
    for k, v in raw.items():
        dirs[f"{k}_raw"] = v
        dirs[f"{k}_jrem4"] = JS.j_removed_direction(basis, v, JREM_RANK)
    assert set(dirs) == set(DIR_NAMES)
    nulls = {}
    for k in range(K_NULL):
        nulls[f"iso_{k}"] = JS.random_direction(basis, JREM_RANK, "iso", seed=9100 + 3 * k)
        nulls[f"nonj4_{k}"] = JS.random_direction(basis, JREM_RANK, "nonj", seed=9101 + 3 * k)
    q90 = {k: q90_dose(h, v) for k, v in dirs.items()}

    def retrained(X):  # noqa: ANN001, ANN202
        groups = np.array([it["cluster"] for it in items])
        return float(P.cv_auroc(X.float(), y, groups)["auroc"])

    med = float(h.norm(dim=1).median())
    geom = {"cosines": cosines(dirs),
            "enrichment": {k: {f"r{r}": float(JS.enrichment(basis, v, r)) for r in RANKS} for k, v in dirs.items()},
            "q90_over_median_norm": {k: v / med for k, v in q90.items()},
            "natural_auroc_on_dir": {k: auroc_labels((h @ v).numpy(), y) for k, v in dirs.items()},
            "recon_probe_auroc": {"true": retrained(h), "arexpl": retrained(rec_expl), "arspan": retrained(rec_span)},
            "recon_fve": {"arexpl": float(NL.fve(rec_expl, h, NL.predict_mean_baseline(h)).mean()),
                          "arspan": float(NL.fve(rec_span, h, NL.predict_mean_baseline(h)).mean())},
            "median_norm": med, "n": len(items), "n_pos": int(y.sum())}
    _save(path, {"ids": [it["id"] for it in items], "y": y, "h": h, "dirs": dirs, "q90": q90, "nulls": nulls,
                 "rec_span": rec_span, "spans": spans, "geometry": geom})
    return {"feature": feature, "geometry": geom, "wall_s": round(time.time() - t0, 1)}


# ============================================================================ behave (H100)
def _steered(spec: dict, cond: tuple):  # noqa: ANN202
    from lib15 import build

    kind, name, q, _s = cond
    v = spec["dirs"][name] if kind == "real" else spec["nulls"][name] if kind == "null" else spec["dirs"][DIR_NAMES[0]]
    return build(spec["h"], v, cond, q90=spec["q90"].get(name) if kind == "real" else None)


@app.function(image=gpu_image, gpu="H100", **GPU)
def behave(feature: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lib15 import cells_to_newline, chars_to_close, conditions, tokens_to_close
    from mats.capture import decoder_blocks
    from mats.readers.jlens import load_jlens, token_variants

    t0 = time.time()
    path = f"{OUT}/behave15_{feature}.pt"
    if _done(path):
        return {"feature": feature, "skipped": True}
    spec = torch.load(f"{OUT}/spec15_{feature}.pt", weights_only=False)
    items, _h, _y = _items(feature)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    blk = decoder_blocks(model)[LAYER]
    state: dict = {"H": None}

    def hook(_m, _i, o):  # noqa: ANN001, ANN202
        if state["H"] is None:
            return o
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] == 1:  # decode step: prompt already edited
            return o
        h = h.clone()
        h[:, -1] = state["H"].to(h.dtype)
        return (h, *o[1:]) if isinstance(o, tuple) else h

    close_ids = [i for i in range(len(tok)) if ")" in tok.decode([i])]
    nl_ids = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
    metric = ROWS[feature]["model_metric"]
    seqs = [it["ids"] for it in items]
    B = 32
    pad = tok.pad_token_id or tok.eos_token_id

    def run(H):  # noqa: ANN001, ANN202
        """H [N, d] or None (clean): next-token log-odds + greedy continuation metrics."""
        lo_close, lo_nl, m_val, m_found, texts = [], [], [], [], []
        for s in range(0, len(seqs), B):
            rows = list(range(s, min(len(seqs), s + B)))
            enc = tok.pad({"input_ids": [seqs[r] for r in rows]}, return_tensors="pt")
            ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
            state["H"] = None if H is None else H[rows].cuda()
            with torch.no_grad():
                lg = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
                lp = torch.log_softmax(lg, -1)
                a = torch.logsumexp(lp[:, close_ids], -1)
                lo_close.append((a - torch.log1p(-a.exp().clamp(max=1 - 1e-6))).cpu())
                b = torch.logsumexp(lp[:, nl_ids], -1)
                lo_nl.append((b - torch.log1p(-b.exp().clamp(max=1 - 1e-6))).cpu())
                g = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=GEN_TOKENS,
                                   pad_token_id=pad)
            for r_i in range(len(rows)):
                new = g[r_i, ids.shape[1]:].tolist()
                toks = [tok.decode([t]) for t in new]
                text = tok.decode(new, skip_special_tokens=True)
                if metric == "cells_to_newline":
                    v, f = cells_to_newline(text)
                else:
                    v, f = tokens_to_close(toks)
                m_val.append(v); m_found.append(f); texts.append(text)
        state["H"] = None
        return {"lo_close": torch.cat(lo_close), "lo_nl": torch.cat(lo_nl), "metric": torch.tensor(m_val, dtype=torch.float32),
                "found": torch.tensor(m_found), "chars_to_close": torch.tensor([chars_to_close(t)[0] for t in texts], dtype=torch.float32),
                "texts": texts}

    hd = blk.register_forward_hook(hook)
    out: dict = {"conds": conditions(), "model": {}, "metric": metric}
    try:
        out["clean_nohook"] = run(None)
        for ci, c in enumerate(out["conds"]):
            out["model"][ci] = run(_steered(spec, c))
    finally:
        hd.remove()
    # J-lens@42 prompted log-odds on every condition (cheap)
    jl = load_jlens(model, tok, layers=[LAYER])
    wa, wb = JLENS_WORDS[feature][ROWS[feature]["values"][0]], JLENS_WORDS[feature][ROWS[feature]["values"][1]]
    ia, ib = token_variants(tok, wa), token_variants(tok, wb)
    out["jlens"] = {ci: (jl.paired_logodds(_steered(spec, c).cuda(), LAYER, ib, ia)).cpu()  # B minus A: + = toward values[1]
                    for ci, c in enumerate(out["conds"])}
    out["wall_s"] = round(time.time() - t0, 1)
    _save(path, out)
    return {"feature": feature, "n_conds": len(out["conds"]), "wall_s": out["wall_s"]}


# ============================================================================ reader scoring
def _score(feature: str, reader: str) -> dict:
    import torch

    from lib15 import conditions

    t0 = time.time()
    path = f"{OUT}/score15_{reader}_{feature}.pt"
    if _done(path):
        return {"feature": feature, "reader": reader, "skipped": True}
    spec = torch.load(f"{OUT}/spec15_{feature}.pt", weights_only=False)
    pairs = _pairs(feature, reader)
    cands = [x for _, a, b in pairs for x in (a, b)]
    if reader == "ol":
        from mats.readers.olens import load_olens

        ol = load_olens()
        score = lambda A: ol.logprob_of_strings(A, LAYER, cands, batch=112)  # noqa: E731
    else:
        from mats.readers import nla as NL

        nla = NL.load_nla(load_ar=False)
        score = lambda A: NL.logprob_of_strings_batch(nla, A, cands, batch=112)  # noqa: E731
    conds = conditions()
    lo = {}
    for ci, c in enumerate(conds):
        lp = score(_steered(spec, c))
        lo[ci] = torch.stack([lp[:, 2 * t + 1] - lp[:, 2 * t] for t in range(len(pairs))])  # B - A: + = toward values[1]
    out = {"conds": conds, "pairs": pairs, "lo": lo, "wall_s": round(time.time() - t0, 1)}
    _save(path, out)
    return {"feature": feature, "reader": reader, "wall_s": out["wall_s"]}


@app.function(image=gpu_image, gpu="H200", **GPU)
def score_nla(feature: str) -> dict:
    return _score(feature, "nla")


@app.function(image=gpu_image, gpu="H100", **GPU)
def score_ol(feature: str) -> dict:
    return _score(feature, "ol")


# ============================================================================ analyze (CPU)
@app.function(image=gpu_image, volumes=VOLUMES, timeout=1800, memory=32768, cpu=4)
def analyze(feature: str) -> dict:
    import torch

    from analyze15 import analyze_feature

    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/spec15_{feature}.pt", weights_only=False)
    beh = torch.load(f"{OUT}/behave15_{feature}.pt", weights_only=False)
    sc = {r: torch.load(f"{OUT}/score15_{r}_{feature}.pt", weights_only=False) for r in ("nla", "ol")}
    res = analyze_feature(feature, spec, beh, sc)
    Path(f"{OUT}/analysis15_{feature}.json").write_text(json.dumps(res, indent=1, default=float))
    VOLUMES["/results"].commit()
    return res


# ============================================================================ entrypoint
@app.local_entrypoint()
def main(stage: str = "prep", features: str = "") -> None:
    feats = features.split(",") if features else list(ROWS)
    RES.mkdir(exist_ok=True)
    fn = {"prep": prep, "behave": behave, "score_nla": score_nla, "score_ol": score_ol, "analyze": analyze}[stage]
    for r in fn.map(feats, return_exceptions=True):
        if isinstance(r, Exception):
            print("ERROR", stage, r)
            continue
        print(stage, json.dumps(r, default=str)[:600])
        if stage in ("prep", "analyze") and not r.get("skipped"):
            (RES / f"{stage}15_{r['feature']}.json").write_text(json.dumps(r, indent=1, default=float))
