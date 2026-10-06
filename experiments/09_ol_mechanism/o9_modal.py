"""09 OL-mechanism pilot (D25 tests a/b/c) -- DECLARED PILOT SPLIT, EXCLUDED FROM MAIN ANALYSIS.

Stages:
  build   fresh items (o9_items.py) -> /results/09_ol_mechanism/items.json; local pilot_split.json
  prep    capture h@L42, directions (col, tense, indent, nextind), random-direction selection by the
          FIXED rule (o9_templates.RANDOM_RULE) on sel09 -> spec.pt, random_selection.json (BEFORE OL scoring)
  run     model behaviour for every condition (4 sets) + OL channels, adapter ON and OFF, all conditions
          + (c) r5 tense120 adapter-off re-score

    modal run o9_modal.py                     # all stages
    modal run o9_modal.py --stage run         # after build+prep
    ../../.venv/bin/python o9_analyze.py
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, OUT02, OUT04, OUT07, OUT08, VOLUMES, app_image, build_image

app = modal.App("mats-09-ol-mech")
L = 42
NL_ID = 198  # "\n"
OL_CHUNKS = {"codeNL": 2, "codeIN": 2, "col": 6, "tense": 7}


def _sec(t0: float) -> float:
    return round(time.time() - t0, 1)


# ============================================================================= build
@app.function(image=build_image, volumes=VOLUMES, timeout=3 * 3600, memory=32768, cpu=8)
def build_items() -> dict:
    import sys

    from transformers import AutoTokenizer

    sys.path.insert(0, "/root/ext08")
    import o9_items as OI
    import pilot_items as PI

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    heads, sha = OI.used_heads(OUT02, OUT04, OUT07, OUT08)
    s1 = PI.load_02("s1_data", OUT02)
    col = OI.build_col(tok, s1, heads, sha)
    sha |= {it["passage_sha1_16"] for it in col}
    sel = OI.build_sel(tok, s1, heads, sha)
    sha |= {it["source_sha1_16"] for it in sel}
    tense = OI.build_tense(tok, s1, OUT02, OUT07, heads, sha)
    code = OI.build_code(tok, OUT04)
    items = {"col": col, "sel": sel, "tense": tense, "codeNL": code["codeNL"], "codeIN": code["codeIN"],
             "codeTR": code["codeTR"],
             "meta": {"n_used_heads": len(heads), "n_used_sha": len(sha), "code_skip_used": code["_n_skip_used"],
                      "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/items.json").write_text(json.dumps(items))
    VOLUMES["/results"].commit()
    return items


# ============================================================================= GPU helpers
def _logprobs(model, seqs, batch: int = 16):  # noqa: ANN001, ANN202
    """[N, V] fp32 log-softmax at each sequence's last token (edits in scope apply)."""
    import torch

    from mats import capture as C

    out = []
    with torch.no_grad():
        for s in range(0, len(seqs), batch):
            rows = list(range(s, min(len(seqs), s + batch)))
            ids, mask = C._pad([seqs[r] for r in rows], 0, "cuda")
            with C._batch_context(rows, [len(seqs[r]) for r in rows]):
                hs = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            last = hs[torch.arange(len(rows)), mask.sum(1) - 1]
            out.append(torch.log_softmax(model.lm_head(last).float(), -1))
    return torch.cat(out)


def _edit_fn(v, q: str, sg: int, scalar: float | None):  # noqa: ANN001, ANN202
    from o9_lib import REL

    def fn(h, _r):  # noqa: ANN001, ANN202
        vv = v.to(h.device).float()
        dd = REL[q] * sg * h.norm(dim=-1, keepdim=True) if q in REL else scalar * sg
        return h - (h @ vv)[:, None] * vv[None] + dd * vv[None]
    return fn


def _run_cond(model, seqs, pos: int, v, q, sg, scalar):  # noqa: ANN001, ANN202
    from mats import capture as C

    if v is None:
        return _logprobs(model, seqs)
    with C.edit(model, L, _edit_fn(v, q, sg, scalar), positions=[[pos]] * len(seqs)):
        return _logprobs(model, seqs)


def _load_model():  # noqa: ANN202
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


# ============================================================================= prep
@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=196608)
def prep() -> dict:
    import re

    import numpy as np
    import torch

    from mats import capture as C
    from mats import jspace as JS
    from mats import probes as P
    from mats.metrics import auroc_labels
    from o9_templates import EXCLUDE_RAND_TOKENS, RANDOM_RULE, RANDOM_RULE_A1

    t0 = time.time()
    VOLUMES["/results"].reload()
    items = json.loads(Path(f"{OUT}/items.json").read_text())
    model, tok = _load_model()
    b42 = JS.load_basis_file(f"/results/03_pipeline_smoke/jspace_basis_L{L}.pt")
    colspec = torch.load(f"{OUT08}/spec.pt", weights_only=False)
    calspec = torch.load(f"{OUT08}/calib_spec.pt", weights_only=False)
    nulls = {k: v.float() for k, v in colspec["nulls"][L].items() if k.startswith(("iso_", "nonj4_"))}
    assert len(nulls) == 40, len(nulls)

    H = {}
    for S in ("col", "tense", "sel", "codeNL", "codeIN", "codeTR"):
        its = items[S]
        H[S] = C.capture(model, [it["ids"] for it in its], [L], [[-1]] * len(its), batch_size=8)[L][:, 0]
    checks: dict = {"mean_norm": {S: float(H[S].norm(dim=1).mean()) for S in H}}

    # ---- directions
    D4 = torch.load(f"{OUT04}/gate/indent_dirs.pt", weights_only=False)
    ind = D4[(42, 0)]["dirs"]["raw"].float() + D4[(42, 1)]["dirs"]["raw"].float()
    ind = ind / ind.norm()
    ytr = np.array([it["y"] for it in items["codeTR"]])
    nxt = P.mean_diff_direction(H["codeTR"], ytr).cpu().float()
    code_dirs = {"indent_raw": ind, "indent_jrem4": JS.j_removed_direction(b42, ind, 1536).float(),
                 "nextind_raw": nxt, "nextind_jrem4": JS.j_removed_direction(b42, nxt, 1536).float()}
    cd, cdz = colspec["dirs"][L], colspec["doses"][L]
    col_dirs = {"reg_raw": cd["raw"].float(), "reg_jrem4": cd["jrem4"].float()}
    col_doses = {"reg_raw": {"q90": float(cdz["raw"]["q90"])}, "reg_jrem4": {"q90": float(cdz["jrem4"]["q90"])}}
    td, tdz = calspec["tense"]["dirs"][L], calspec["tense"]["doses"][L]
    ten_dirs = {"raw": td["raw"].float(), "jrem4": td["jrem4"].float()}
    ten_doses = {k: {"q90": float(tdz[k]["q90"])} for k in ("raw", "jrem4")}

    allv = {**code_dirs, **{f"col_{k}": v for k, v in col_dirs.items()}, **{f"tense_{k}": v for k, v in ten_dirs.items()}}
    checks["enrichment"] = {k: {f"r{r}": float(JS.enrichment(b42, v, r)) for r in (384, 1536)} for k, v in allv.items()}
    checks["cos"] = {"indent_raw.nextind_raw": float(ind @ nxt),
                     "indent_jrem4.nextind_jrem4": float(code_dirs["indent_jrem4"] @ code_dirs["nextind_jrem4"]),
                     "indent_fold0.fold1": float(D4[(42, 0)]["dirs"]["raw"].float() @ D4[(42, 1)]["dirs"]["raw"].float())}
    lab = {"codeNL_next8": np.array([it["y"] for it in items["codeNL"]]),
           "codeNL_curlevel2": np.array([int(it["cur_indent"] >= 8) for it in items["codeNL"]]),
           "codeIN_curlevel2": np.array([it["y"] for it in items["codeIN"]]),
           "codeIN_next8": np.array([int(it["next_indent"] == 8) for it in items["codeIN"]])}
    checks["natural_auroc"] = {f"{k}|{lk}": auroc_labels((H[lk.split('_')[0]] @ v).numpy(), y)
                               for k, v in code_dirs.items() for lk, y in lab.items()}
    yt = np.array([it["y"] for it in items["tense"]])
    checks["natural_auroc"] |= {f"tense_{k}|past": auroc_labels((H["tense"] @ v).numpy(), yt) for k, v in ten_dirs.items()}
    ycol = np.array([it["col"] for it in items["col"]], float)
    from scipy.stats import spearmanr
    checks["natural_spearman_col"] = {k: float(spearmanr((H["col"] @ v).numpy(), ycol).statistic) for k, v in col_dirs.items()}
    checks["t_dirs"] = _sec(t0)

    # ---- random-direction selection (FIXED RULE) on sel09, model run only
    seqs = [it["ids"] for it in items["sel"]]
    lp0 = _run_cond(model, seqs, -1, None, None, 0, None)
    pnat = lp0.exp().mean(0).cpu()
    voc = [tok.decode([i]) for i in range(len(tok))]
    voc += [""] * (lp0.shape[1] - len(voc))
    elig = torch.tensor([bool(re.fullmatch(r" [a-z]{3,}", s)) and s not in EXCLUDE_RAND_TOKENS for s in voc])
    elig &= pnat >= 1e-4
    def pool(kind: str, seed0: int) -> list[dict]:
        cands = []
        for k in range(RANDOM_RULE["n_candidates"]):
            v = JS.random_direction(b42, 1536, kind, seed=seed0 + k).float().cpu()
            lpp = _run_cond(model, seqs, -1, v, "r108", 1, None).mean(0).cpu()
            lpm = _run_cond(model, seqs, -1, v, "r108", -1, None).mean(0).cpu()
            Dt = (lpp - lpm) / 2
            De = torch.where(elig, Dt, torch.tensor(float("nan")))
            a = torch.argsort(torch.nan_to_num(De, nan=-1e9), descending=True)[:3].tolist()
            b = torch.argsort(torch.nan_to_num(De, nan=1e9))[:3].tolist()
            score = float(np.mean([Dt[x] - Dt[y] for x, y in zip(a, b)]))
            cands.append({"kind": kind, "k": k, "seed": seed0 + k, "a": a, "b": b, "a_str": [voc[x] for x in a],
                          "b_str": [voc[x] for x in b], "D_a": [float(Dt[x]) for x in a],
                          "D_b": [float(Dt[x]) for x in b], "score": score,
                          "mean_abs_D_eligible": float(torch.nanmean(De.abs())), "v": v})
        return sorted(cands, key=lambda c: -c["score"])

    ranked = pool("nonj", 909000)
    ranked_iso = pool("iso", 919000)  # amendment A1
    chosen = {"randA": ranked[0], "randB": ranked[1], "randC": ranked_iso[0], "randD": ranked_iso[1]}
    strip = lambda c: {k: v for k, v in c.items() if k != "v"}  # noqa: E731
    sel = {"rule": RANDOM_RULE, "rule_A1": RANDOM_RULE_A1, "n_eligible_tokens": int(elig.sum()),
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "candidates": [strip(c) for c in ranked + ranked_iso],
           "chosen": {n: strip(c) for n, c in chosen.items()},
           "ol_templates": {n: [("-" + x, "-" + y) for x, y in zip(c["a_str"], c["b_str"])] for n, c in chosen.items()}}
    Path(f"{OUT}/random_selection.json").write_text(json.dumps(sel, indent=1))
    checks["t_select"] = _sec(t0)
    rand = {n: c["v"] for n, c in chosen.items()}
    checks["cos_rand"] = {f"{n}.{m}": float(rand[n] @ v) for n in rand for m, v in allv.items()}

    def meta(S: str) -> list[dict]:
        return [{k: v for k, v in it.items() if k not in ("ids",)} | {"n_tok": len(it["ids"])} for it in items[S]]

    spec = {
        "codeNL": {"h": H["codeNL"], "dirs": code_dirs, "doses": {}, "nulls": nulls, "items": meta("codeNL"),
                   "seqs": [it["ids"] for it in items["codeNL"]], "edit_pos": -1},
        "codeIN": {"h": H["codeIN"], "dirs": code_dirs, "doses": {}, "nulls": nulls, "items": meta("codeIN"),
                   "seqs": [it["ids"] + [NL_ID] for it in items["codeIN"]], "edit_pos": -2},
        "col": {"h": H["col"], "dirs": {**col_dirs, **rand}, "doses": col_doses, "nulls": nulls, "items": meta("col"),
                "seqs": [it["ids"] for it in items["col"]], "edit_pos": -1},
        "tense": {"h": H["tense"], "dirs": {**ten_dirs, **rand}, "doses": ten_doses, "nulls": nulls,
                  "items": meta("tense"), "seqs": [it["ids"] for it in items["tense"]], "edit_pos": -1},
        "random": {"chosen": {n: strip(c) for n, c in chosen.items()},
                   "ol_templates": sel["ol_templates"]},
        "checks": checks,
    }
    torch.save(spec, f"{OUT}/spec.pt")
    VOLUMES["/results"].commit()
    return {"checks": checks, "chosen": sel["chosen"], "ol_templates": sel["ol_templates"],
            "top5": [{k: c[k] for k in ("kind", "k", "score", "a_str", "b_str")} for c in ranked[:3] + ranked_iso[:5]]}


# ============================================================================= behaviour
def _readout_ids(tok, spec) -> dict:  # noqa: ANN001
    import sys

    sys.path.insert(0, "/root/ext08")
    from o9_templates import READOUTS
    from pilot_lib import eol_token_ids

    eol = eol_token_ids(tok)

    def ids(x):  # noqa: ANN001, ANN202
        if x == "NL":
            return eol["narrow"]
        if x == "NLB":
            return eol["broad"]
        out = []
        for s in x:
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                out.append(e[0])
            else:
                assert len(x) > 1, (s, e)  # single-token pairs must resolve exactly
        assert out, x
        return out

    R = {}
    for name, spec_ in READOUTS.items():
        R[name] = ids(spec_) if isinstance(spec_, str) else [(ids(a), ids(b)) for a, b in spec_]
    for n, c in spec["random"]["chosen"].items():
        R[n] = [([x], [y]) for x, y in zip(c["a"], c["b"])]
    return R


def _apply_readouts(lp, R) -> dict:  # noqa: ANN001
    import torch

    out = {}
    for name, r in R.items():
        if isinstance(r[0], int):  # logit of a token class
            a = torch.logsumexp(lp[:, r], -1)
            out[name] = (a - torch.log1p(-a.exp().clamp(max=1 - 1e-6))).cpu().numpy()
        else:
            out[name] = torch.stack([torch.logsumexp(lp[:, A], -1) - torch.logsumexp(lp[:, B], -1)
                                     for A, B in r]).mean(0).cpu().numpy()
    return out


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=131072)
def behave(S: str) -> dict:
    import torch

    from o9_lib import conditions, vec

    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    sp = spec[S]
    model, tok = _load_model()
    R = _readout_ids(tok, spec)
    conds = conditions(S)
    res = {"conds": conds, "readouts": list(R), "vals": []}
    for c in conds:
        if c[0] == "zero":
            lp = _run_cond(model, sp["seqs"], sp["edit_pos"], None, None, 0, None)
        else:
            q, ref = c[2], c[4]
            scalar = None if q.startswith("r") else float(sp["doses"][ref][q])
            lp = _run_cond(model, sp["seqs"], sp["edit_pos"], vec(sp, c), q, c[3], scalar)
        res["vals"].append(_apply_readouts(lp, R))
    res["t"] = _sec(t0)
    torch.save(res, f"{OUT}/behave_{S}.pt")
    VOLUMES["/results"].commit()
    return {"set": S, "n_conds": len(conds), "t": res["t"]}


# ============================================================================= OL scoring
def _channels(S: str, spec: dict) -> list[tuple]:
    from o9_templates import OL_CHANNELS, SET_CHANNELS

    ch = {**OL_CHANNELS, **{n: [tuple(p) for p in t] for n, t in spec["random"]["ol_templates"].items()}}
    return [(fam, a, b) for fam in SET_CHANNELS[S] for a, b in ch[fam]]


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=131072)
def score_ol(S: str, chunk: int, n_chunks: int, adapter: str) -> dict:
    import contextlib

    import torch

    from mats.readers.olens import load_olens
    from o9_lib import build, conditions

    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    sp = spec[S]
    conds = conditions(S)
    idx = list(range(len(conds)))[chunk::n_chunks]
    fp = _channels(S, spec)
    cands = [x for _, a, b in fp for x in (a, b)]
    ol = load_olens()
    A = torch.cat([build(sp, conds[i]) for i in idx])
    ctx = ol.base() if adapter == "off" else contextlib.nullcontext()
    with ctx:
        lp = ol.logprob_of_strings(A, L, cands, batch=120)
    lo = torch.stack([lp[:, 2 * t] - lp[:, 2 * t + 1] for t in range(len(fp))]).reshape(len(fp), len(idx), -1)
    la = torch.stack([lp[:, 2 * t] for t in range(len(fp))]).reshape(len(fp), len(idx), -1)
    out = {"idx": idx, "templates": fp, "lo": lo, "lp_a": la, "adapter": adapter, "t": _sec(t0)}
    torch.save(out, f"{OUT}/ol_{adapter}_{S}_chunk{chunk}.pt")
    VOLUMES["/results"].commit()
    return {"set": S, "chunk": chunk, "adapter": adapter, "n_acts": int(A.shape[0]), "t": out["t"]}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=131072)
def tense_r5() -> dict:
    """(c) symmetric to r5_mech: r5 tense120 activations (r5_spec.pt), every r5 tense condition, OL f07_tense +
    tense_tok with the adapter OFF; adapter ON re-score of zero + real cells (reproduction for f07_tense)."""
    import sys

    import torch

    sys.path.insert(0, "/root/ext08")
    from mats.readers.olens import load_olens
    from o9_templates import OL_CHANNELS
    from r5_lib import build, conditions

    t0 = time.time()
    VOLUMES["/results"].reload()
    sp = torch.load(f"{OUT08}/r5_spec.pt", weights_only=False)["tense"]
    conds = conditions("tense")
    fp = [(fam, a, b) for fam in ("f07_tense", "tense_tok") for a, b in OL_CHANNELS[fam]]
    cands = [x for _, a, b in fp for x in (a, b)]
    ol = load_olens()

    def score(acts):  # noqa: ANN001, ANN202
        lp = ol.logprob_of_strings(acts, L, cands, batch=120)
        return torch.stack([lp[:, 2 * t] - lp[:, 2 * t + 1] for t in range(len(fp))])

    A = torch.cat([build(sp, c) for c in conds])
    with ol.base():
        off = score(A).reshape(len(fp), len(conds), -1)
    on_idx = [i for i, c in enumerate(conds) if c[0] in ("zero", "real")]
    on = score(torch.cat([build(sp, conds[i]) for i in on_idx])).reshape(len(fp), len(on_idx), -1)
    out = {"conds": conds, "templates": fp, "ol_off": off, "ol_on": on, "on_idx": on_idx, "t": _sec(t0)}
    torch.save(out, f"{OUT}/tense_r5_mech.pt")
    VOLUMES["/results"].commit()
    return {"t": out["t"]}


# ============================================================================= entrypoint
@app.local_entrypoint()
def main(stage: str = "all") -> None:
    import subprocess

    here = Path(__file__).resolve().parent
    if stage in ("all", "build"):
        items = build_items.remote()
        split = {
            "status": "DECLARED PILOT SPLIT (09_ol_mechanism) -- EXCLUDED FROM THE MAIN ANALYSIS",
            "experiment": "09_ol_mechanism", "generator": "o9_items.py", "meta": items["meta"],
            "exclusion_rule": "wikitext paragraphs: no 6-word shingle shared with the first 6 words of any line of "
                              "any earlier item text (02 wrap/prose, 04 prose/lists/dialogue, 07 items_base/extra/"
                              "nj3_pool, 08 pilot/calib/round-5) and no sha1 match to an 08 passage; code from "
                              "codeparrot/codeparrot-clean-valid (unused before), one item per file.",
            "sets": {
                "col": [{k: it[k] for k in ("id", "passage_sha1_16", "passage_head", "width", "col")} for it in items["col"]],
                "sel": [{k: it[k] for k in ("id", "source_sha1_16", "source_head")} for it in items["sel"]],
                "tense": [{k: it[k] for k in ("id", "label", "source_sha1_16", "source_head")} for it in items["tense"]],
                **{S: [{k: it[k] for k in ("id", "repo", "path", "line_index", "next_indent")} for it in items[S]]
                   for S in ("codeNL", "codeIN", "codeTR")},
            }}
        (here / "pilot_split.json").write_text(json.dumps(split, indent=1))
        print({k: len(v) for k, v in items.items() if isinstance(v, list)}, items["meta"])
    if stage in ("all", "prep"):
        r = prep.remote()
        print(json.dumps(r, indent=1, default=str)[:12000])
    if stage in ("all", "run"):
        calls = [behave.spawn(S) for S in OL_CHUNKS] + [tense_r5.spawn()]
        for ad in ("on", "off"):
            for S, n in OL_CHUNKS.items():
                calls += [score_ol.spawn(S, i, n, ad) for i in range(n)]
        out = []
        for c in calls:
            try:
                out.append(c.get())
            except Exception as e:  # noqa: BLE001
                out.append({"error": repr(e)[:2000]})
        print(json.dumps(out, indent=1, default=str))
        (here / "results" / "run_summary.json").write_text(json.dumps(out, indent=1, default=str))
    (here / "results").mkdir(exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mats-results", "09_ol_mechanism", str(here / "results")],
                   check=False)
