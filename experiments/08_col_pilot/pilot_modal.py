"""08 col feasibility pilot -- DECLARED PILOT SPLIT, EXCLUDED FROM THE MAIN ANALYSIS.

Stages (app ``mats-08-col-pilot``):
  build_items (CPU)  60 fresh wrapped passages -> /results/08_col_pilot/items.json
  prep (H100)        capture L40/42/44 at the read token -> col directions (raw, J-removed at r*,
                     4r*) -> doses from 02's wrap bank -> K=20 null directions per class ->
                     J-Lens template LO + newline/EOL leak scores for EVERY condition at 40/42/44
                     -> 02 ridge-probe predictions -> TextNLA (local, prompt mode, natural only)
                     -> J-Lens natural top-20 -> spec.pt, jlens.pt, natural.pt
  score_ol (H100 xN) OL paired LO (10 candidates in one batch per activation) at L42 (all
                     conditions) and L40/L44 (real + jrem1 nulls)
  score_nla (H200xN) NLA paired LO at L42 (all conditions)
  fve (H200 xN)      NLA FVE guard natural vs steered (k=4, per-example paired) + explanations
  gen_ol (H100)      OL free generations (k=4), natural and steered, for the mention base rate

    cd experiments/08_col_pilot
    modal run pilot_modal.py
    ../../.venv/bin/python analyze.py
"""

import json
import time
from pathlib import Path

import modal

from env import OUT, OUT02, OUT03, OUT07, VOLUMES, app_image

app = modal.App("mats-08-col-pilot")
READ = (40, 42, 44)
NLA_BASELINE_00 = 0.4616  # predict-the-mean baseline on the 00 FineWeb set (as in 03)
FVE_CONDS = [("zero", None, "add", None, 0, None)] + [
    ("real", d, "centred", q, s, d) for d in ("jrem1", "jrem4", "raw") for q in ("q90", "q99")
    for s in (1, -1)] + [
    ("null", n, "centred", "q99", s, "jrem1") for n in ("iso_0", "nonj1_0") for s in (1, -1)]
GEN_CONDS = [("zero", None, "add", None, 0, None)] + [
    ("real", "jrem1", "centred", q, s, "jrem1") for q in ("q90", "q99") for s in (1, -1)] + [
    ("real", "raw", "centred", "q99", s, "raw") for s in (1, -1)]


def _sec(t0: float) -> float:
    return round(time.time() - t0, 1)


@app.function(image=app_image, volumes=VOLUMES, timeout=3600, memory=32768, cpu=4)
def build_items() -> list[dict]:
    from transformers import AutoTokenizer

    import pilot_items as PI

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    items = PI.build(tok, OUT02, OUT07)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/items.json").write_text(json.dumps(items))
    VOLUMES["/results"].commit()
    return items


def _bank_projections(dirs_by_layer: dict, rng_seed: int = 0) -> dict:
    """Natural projections of 02's wrap bank (col rows, <=40k, as in 02's s3) on each direction."""
    import pickle
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root/ext02")
    from features import FEATURES_BY_NAME, rows_for

    d = f"{OUT02}/acts/wrap"
    with open(f"{OUT02}/data/wrap.pkl", "rb") as f:
        items = pickle.load(f)
    index = dict(np.load(f"{d}/index.npz"))
    R = rows_for(FEATURES_BY_NAME["col"], items, index, np.random.default_rng(rng_seed))
    rows = R["rows"]
    out = {"n_rows": int(len(rows))}
    for L, dirs in dirs_by_layer.items():
        X = torch.from_numpy(np.load(f"{d}/hs{L + 1}.npy", mmap_mode="r")[rows].astype(np.float32)).cuda()
        out[L] = {k: (X @ v.cuda().float()).cpu() for k, v in dirs.items()}
        out[(L, "norm_median")] = float(X.norm(dim=1).median())
        if L == 42:
            out["y"] = torch.as_tensor(R["y"])
        del X
    return out


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=2 * 3600, memory=131072)
def prep() -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C
    from mats import jspace as JS
    from mats.readers.jlens import load_jlens, token_variants
    from mats.textnla import TextNLALocal, nla_instruction
    from pilot_lib import JLENS_CONDS, K_NULL, REAL_DIRS, build_all, eol_token_ids
    from templates import JLENS_WORDS, NLA_PAIRS, flat_pairs

    t0 = time.time()
    VOLUMES["/results"].reload()
    items = json.loads(Path(f"{OUT}/items.json").read_text())
    N = len(items)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    seqs = [it["ids"] for it in items]
    acts = C.capture(model, seqs, list(READ), [[-1]] * N, batch_size=16)
    # hook check on 4 items vs hidden_states[L+1]
    hook_check = {}
    with torch.no_grad():
        for i in range(4):
            hs = model(torch.tensor([seqs[i]], device="cuda"), output_hidden_states=True).hidden_states
            for L in READ:
                a, b = acts[L][i, 0].float(), hs[L + 1][0, -1].float().cpu()
                hook_check[f"{i}_L{L}"] = float((a - b).norm() / b.norm())
    timing = {"load_capture": _sec(t0)}

    rstar = {int(k): int(v["r_star"]) for k, v in
             json.loads(Path(f"{OUT02}/rstar.json").read_text())["layers"].items() if v.get("r_star")}
    ranks = {L: (rstar[L], 4 * rstar[L]) for L in READ}
    bases = {L: JS.load_basis_file(f"{OUT03}/jspace_basis_L{L}.pt") for L in READ}
    spec = {"h": {}, "dirs": {}, "doses": {}, "nulls": {}, "ranks": ranks, "probe": {},
            "items": items, "checks": {"hook_rel_l2": hook_check}}
    for L in READ:
        P = torch.load(f"{OUT02}/probes/col_hs{L + 1}.pt", weights_only=False)
        raw = P["steer_dir_unit"].float()
        r1, r4 = ranks[L]
        dirs = {"raw": raw, "jrem1": JS.j_removed_direction(bases[L], raw, r1),
                "jrem4": JS.j_removed_direction(bases[L], raw, r4)}
        if L == 42:  # 02's own exports (the gated directions) -- use them, record agreement
            for k, r in (("jrem1", 384), ("jrem4", 1536)):
                u02 = P[f"steer_dir_unit_jrem_r{r}"].float()
                spec["checks"][f"cos_02_vs_ours_{k}"] = float(u02 @ dirs[k])
                dirs[k] = u02 / u02.norm()
        spec["dirs"][L] = dirs
        spec["h"][L] = acts[L][:, 0].float()
        spec["probe"][L] = {"w": P["probe_w_raw"].float(), "b": float(P["probe_b_raw"])}
        spec["checks"][f"enrichment_L{L}"] = {
            k: {f"r{r}": float(JS.enrichment(bases[L], v, r)) for r in ranks[L]} for k, v in dirs.items()}
        spec["checks"][f"cos_raw_jrem_L{L}"] = {k: float(dirs["raw"] @ dirs[k]) for k in ("jrem1", "jrem4")}
        nulls = {}
        for k in range(K_NULL):
            nulls[f"iso_{k}"] = JS.random_direction(bases[L], r1, "iso", seed=104729 * k + L)
            nulls[f"nonj1_{k}"] = JS.random_direction(bases[L], r1, "nonj", seed=104729 * k + L + 1)
            nulls[f"nonj4_{k}"] = JS.random_direction(bases[L], r4, "nonj", seed=104729 * k + L + 2)
        spec["nulls"][L] = {k: v.cpu() for k, v in nulls.items()}
    bank = _bank_projections(spec["dirs"])
    for L in READ:
        spec["doses"][L] = {}
        for k in REAL_DIRS:
            dq = C.doses_from_projection(bank[L][k], quantiles=(0.9, 0.99), extra_multiple=None)
            spec["doses"][L][k] = {"q90": dq[0], "q99": dq[1]}
        spec["checks"][f"dose_over_median_norm_L{L}"] = {
            k: {q: v / bank[(L, "norm_median")] for q, v in dd.items()} for k, dd in spec["doses"][L].items()}
        # pilot items' own natural projection vs the bank (is the pilot in-distribution?)
        spec["checks"][f"pilot_proj_L{L}"] = {
            k: [float(x) for x in np.percentile((spec["h"][L] @ v).numpy(), [10, 50, 90])]
            for k, v in spec["dirs"][L].items()}
        spec["checks"][f"bank_proj_L{L}"] = {
            k: [float(x) for x in np.percentile(bank[L][k].numpy(), [10, 50, 90])] for k in REAL_DIRS}
    spec["checks"]["bank_rows"] = bank["n_rows"]
    timing["directions_doses"] = _sec(t0)

    # ---------------- J-Lens: template LO + EOL leak scores, every condition, 40/42/44
    jl = load_jlens(model, tok, layers=list(READ))
    eol = eol_token_ids(tok)
    jtemps = [(fam, a, b) for fam, lst in JLENS_WORDS.items() for a, b in lst]
    jids = [(token_variants(tok, a), token_variants(tok, b)) for _, a, b in jtemps]
    all_ids = sorted({i for ia, ib in jids for i in ia + ib} | set(eol["broad"]))
    pos_of = {t: j for j, t in enumerate(all_ids)}
    col = lambda ids: [pos_of[t] for t in ids]  # noqa: E731
    jout = {"conds": JLENS_CONDS, "templates": jtemps, "eol_sizes": {k: len(v) for k, v in eol.items()},
            "lo": {}, "eol": {}, "probe": {}}
    for L in READ:
        S = build_all(spec, L, JLENS_CONDS)  # [C*N, d]
        lo, en, eb, pr = [], [], [], []
        for s in range(0, S.shape[0], 4096):
            x = S[s:s + 4096]
            sc = jl.scores(x, L, ids=all_ids)  # [n, |ids|] cosine readout, only the needed tokens
            lo.append(torch.stack([sc[:, col(ia)].max(1).values - sc[:, col(ib)].max(1).values
                                   for ia, ib in jids]))
            en.append(sc[:, col(eol["narrow"])].max(1).values)
            eb.append(sc[:, col(eol["broad"])].max(1).values)
            pr.append(x @ spec["probe"][L]["w"] + spec["probe"][L]["b"])
        C_ = len(JLENS_CONDS)
        jout["lo"][L] = torch.cat(lo, 1).reshape(len(jtemps), C_, N)
        jout["eol"][L] = {"narrow": torch.cat(en).reshape(C_, N), "broad": torch.cat(eb).reshape(C_, N)}
        jout["probe"][L] = torch.cat(pr).reshape(C_, N)
    timing["jlens_all"] = _sec(t0)

    # ---------------- natural arm: J-Lens top-20 readout, EOL rank, TextNLA (prompt mode)
    nat = {"jl_top20": jl.readout(spec["h"][42], 42, k=20),
           "eol_rank_narrow": jl.rank_of(spec["h"][42], 42, eol["narrow"]).tolist()}
    fp = flat_pairs(NLA_PAIRS)
    nat["textnla_templates"] = fp
    try:  # descriptive only (natural arm)
        tn = TextNLALocal(model, tok, instruction=nla_instruction())
        nat["textnla_lo"] = tn.paired_logodds_templates([it["text"] for it in items],
                                                        [(a, b) for _, a, b in fp], mode="prompt")
    except Exception as e:  # noqa: BLE001
        nat["textnla_error"] = repr(e)[:2000]
    timing["natural"] = _sec(t0)

    torch.save(spec, f"{OUT}/spec.pt")
    torch.save(jout, f"{OUT}/jlens.pt")
    torch.save(nat, f"{OUT}/natural.pt")
    VOLUMES["/results"].commit()
    return {"timing": timing, "checks": spec["checks"], "doses": spec["doses"], "ranks": ranks,
            "eol_sizes": jout["eol_sizes"], "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}


def _jobs(reader: str) -> list[tuple[int, int]]:
    from pilot_lib import conditions

    layers = (42, 40, 44) if reader == "ol" else (42,)
    return [(L, i) for L in layers for i in range(len(conditions(L)))]


def _score(reader: str, chunk: int, n_chunks: int) -> dict:
    import torch

    from pilot_lib import build, conditions
    from templates import NLA_PAIRS, OL_PAIRS, flat_pairs

    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    jobs = _jobs(reader)[chunk::n_chunks]
    if reader == "ol":
        from mats.readers.olens import load_olens

        ol = load_olens()
        fp = flat_pairs(OL_PAIRS)
        score = lambda A, L, c: ol.logprob_of_strings(A, L, c, batch=100)  # noqa: E731
    else:
        from mats.readers import nla as NL

        nla = NL.load_nla(load_ar=False)
        fp = flat_pairs(NLA_PAIRS)
        score = lambda A, L, c: NL.logprob_of_strings_batch(nla, A, c, batch=100)  # noqa: E731
    t_load = _sec(t0)
    cands = [s for _, a, b in fp for s in (a, b)]
    T = len(fp)
    out = {"jobs": jobs, "templates": fp, "lo": {}}
    t1 = time.time()
    n_acts = 0
    for L in sorted({L for L, _ in jobs}):
        idx = [i for LL, i in jobs if LL == L]
        conds = conditions(L)
        A = torch.cat([build(spec, L, conds[i]) for i in idx])
        lp = score(A, L, cands)  # [len(idx)*N, 2T]
        lo = torch.stack([lp[:, 2 * t] - lp[:, 2 * t + 1] for t in range(T)])
        out["lo"][L] = {"idx": idx, "lo": lo.reshape(T, len(idx), -1)}
        n_acts += A.shape[0]
    t_score = time.time() - t1
    out |= {"t_load": t_load, "t_score": t_score, "s_per_act": t_score / max(n_acts, 1),
            "gpu": torch.cuda.get_device_name(0), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    torch.save(out, f"{OUT}/{reader}_chunk{chunk}.pt")
    VOLUMES["/results"].commit()
    return {k: v for k, v in out.items() if k not in ("lo", "jobs", "templates")} | {
        "n_acts": n_acts, "chunk": chunk, "reader": reader}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def score_ol(chunk: int, n_chunks: int) -> dict:
    return _score("ol", chunk, n_chunks)


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600, memory=131072)
def score_nla(chunk: int, n_chunks: int) -> dict:
    return _score("nla", chunk, n_chunks)


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600, memory=131072)
def fve(part: int, n_parts: int) -> dict:
    import torch

    from mats.readers import nla as NL
    from pilot_lib import build

    torch.manual_seed(1000 + part)
    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    nla = NL.load_nla(load_ar=True)
    t_load = _sec(t0)
    mine = list(range(len(FVE_CONDS)))[part::n_parts]
    res = {}
    for ci in mine:
        A = build(spec, 42, FVE_CONDS[ci])
        o = nla.fve_on(A, k=4, baseline=NLA_BASELINE_00)
        res[ci] = {"cond": FVE_CONDS[ci], "per_example": o["per_example"], "per_sample": o["per_sample"],
                   "explanations": o["explanations"], "n_failed": o["n_failed"]}
    out = {"res": res, "t_load": t_load, "total_s": _sec(t0), "gpu": torch.cuda.get_device_name(0)}
    torch.save(out, f"{OUT}/fve_part{part}.pt")
    VOLUMES["/results"].commit()
    return {"part": part, "conds": mine, "t_load": t_load, "total_s": out["total_s"]}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def gen_ol() -> dict:
    import torch

    from mats.readers.olens import WSBENCH_SAMPLING, load_olens
    from pilot_lib import build

    torch.manual_seed(7)
    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    ol = load_olens()
    A = torch.cat([build(spec, 42, c) for c in GEN_CONDS])
    texts = ol.verbalize(A, 42, k=4, sampling=WSBENCH_SAMPLING, batch=96)
    N = spec["h"][42].shape[0]
    out = {"conds": GEN_CONDS, "texts": [texts[i * N:(i + 1) * N] for i in range(len(GEN_CONDS))],
           "total_s": _sec(t0)}
    torch.save(out, f"{OUT}/gen_ol.pt")
    VOLUMES["/results"].commit()
    return {"total_s": out["total_s"]}


@app.local_entrypoint()
def main(skip_build: bool = False, skip_prep: bool = False, ol_chunks: int = 5, nla_chunks: int = 6,
         fve_parts: int = 4) -> None:
    import subprocess

    here = Path(__file__).resolve().parent
    summary = {}
    if not skip_build:
        items = build_items.remote()
        split = {"status": "DECLARED PILOT SPLIT -- EXCLUDE FROM MAIN ANALYSIS",
                 "experiment": "08_col_pilot", "feature": "col (NJ1)",
                 "generator": "pilot_items.py (PILOT_SEED 8008, 02 cleaner + textwrap, widths 30-70)",
                 "exclusion_rule": "main-run item generation must drop any wikitext paragraph whose "
                                   "sha1(whitespace-normalised text)[:16] or normalised 80-char head "
                                   "matches an entry below",
                 "items": [{"id": it["id"], "passage_sha1_16": it["passage_sha1_16"],
                            "passage_head": it["passage_head"], "width": it["width"], "pos": it["pos"],
                            "col": it["col"]} for it in items]}
        (here / "pilot_split.json").write_text(json.dumps(split, indent=1))
        print(f"built {len(items)} items")
    if not skip_prep:
        summary["prep"] = prep.remote()
        print(json.dumps(summary["prep"], indent=1, default=str)[:6000])
    calls = [score_ol.spawn(i, ol_chunks) for i in range(ol_chunks)]
    calls += [score_nla.spawn(i, nla_chunks) for i in range(nla_chunks)]
    calls += [fve.spawn(i, fve_parts) for i in range(fve_parts)]
    calls += [gen_ol.spawn()]
    summary["jobs"] = []
    for c in calls:
        try:
            summary["jobs"].append(c.get())
        except Exception as e:  # keep the others; report the failure
            summary["jobs"].append({"error": repr(e)[:2000]})
    print(json.dumps(summary["jobs"], indent=1, default=str))
    (here / "results").mkdir(exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "rfm-results", "08_col_pilot", str(here / "results")],
                   check=False)
    (here / "results" / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
