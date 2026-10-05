"""Norm-preserving re-run of the 03 toy (Fable round 3).

Question: is OL/J-Lens anti-tracking on the J-removed direction a norm-asymmetry artefact?

prep (H100): capture L40/42/44 (last token) and L42 last-4 tokens -> directions per fold
  (raw mean-diff, J-removed mean-diff, J-removed INLP-1, J-removed INLP-8 subspace) -> q90/q99
  doses -> K=20 isotropic + K=20 J-complement null directions -> J-Lens LO for EVERY condition at
  40/42/44 (cheap) -> spec.pt + jlens.pt.
score (OL on H100 / NLA on H200, chunked across containers): template-mean paired LO at L42
  for every condition (3 templates' 6 candidates scored in one batch per activation).

    cd experiments/03_pipeline_smoke
    modal run normrerun_modal.py                  # prep, then OL x3 + NLA x4 chunks in parallel
    ../../.venv/bin/python normrerun_analyze.py
"""

import json
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, app_image

app = modal.App("mats-03-normrerun")
OUTN = f"{OUT}/normrerun"
READ = (40, 42, 44)
RSTAR = {40: 384, 42: 384, 44: 256}
INLP_K = 8


def _sec(t0: float) -> float:
    return round(time.time() - t0, 2)


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=2 * 3600, memory=131072)
def prep() -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C
    from mats import jspace as JS
    from mats import probes as P
    from mats.readers.jlens import load_jlens
    from normrerun_lib import K_NULL, build_all, conditions
    from smoke_items import JLENS_WORDS, items

    t0 = time.time()
    VOLUMES["/results"].reload()
    it = items()
    N = len(it)
    y = np.array([x["label"] for x in it])
    pair = np.array([x["pair"] for x in it])
    fold = pair % 2
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    seqs = [tok.encode(x["text"], add_special_tokens=False) for x in it]
    acts = C.capture(model, seqs, list(READ), [[-4, -3, -2, -1]] * N, batch_size=16)
    jl = load_jlens(model, tok, layers=list(READ))
    bases = {L: JS.load_basis_file(f"{OUT}/jspace_basis_L{L}.pt") for L in READ}
    timing = {"load_capture": _sec(t0)}

    spec = {"h": {}, "dirs": {}, "doses": {}, "nulls": {}, "inlp_heldout_auroc": {}, "y": y,
            "pair": pair, "fold": fold}
    for L in READ:
        r = RSTAR[L]
        h = acts[L][:, -1]
        spec["h"][L] = h
        dirs, doses = {}, {}
        for f in (0, 1):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            X = acts[L][tr].reshape(-1, h.shape[-1]).cuda()
            yy = np.repeat(y[tr], 4)
            md = P.mean_diff_direction(X, yy).cpu()
            dirs[("raw", f)] = md
            dirs[("jrem", f)] = JS.j_removed_direction(bases[L], md, r)
            # INLP: k logistic probes, each fitted after projecting out the previous directions
            Xi, W, aucs = X.clone(), [], []
            for _ in range(INLP_K):
                m = P.fit_logreg(Xi, torch.as_tensor(yy, device=Xi.device))
                w = m.direction()
                if float(w @ md.cuda()) < 0:
                    w = -w
                W.append(w.cpu())
                sc = m.decision(acts[L][te, -1].cuda()).cpu().numpy()
                from mats.metrics import auroc
                aucs.append(auroc(sc[y[te] == 1], sc[y[te] == 0]))
                Xi = Xi - (Xi @ w)[:, None] * w[None]
            spec["inlp_heldout_auroc"][(L, f)] = aucs
            dirs[("inlp1_jrem", f)] = JS.j_removed_direction(bases[L], W[0], r)
            Wj = JS.project_out(bases[L], torch.stack(W), r)  # [k, d], each orthogonal to S_r
            Q, _ = torch.linalg.qr(Wj.T.double())
            v = (Q @ (Q.T @ md.double())).float()
            dirs[("inlp8_jrem", f)] = v / v.norm()
            for name in ("raw", "jrem", "inlp1_jrem", "inlp8_jrem"):
                p = acts[L][tr, -1] @ dirs[(name, f)]
                dq = C.doses_from_projection(p, quantiles=(0.9, 0.99), extra_multiple=None)
                doses[(name, f)] = {"q90": dq[0], "q99": dq[1]}
        spec["dirs"][L], spec["doses"][L] = dirs, doses
        spec["nulls"][L] = {f"{kind}{k}": JS.random_direction(bases[L], r, kind, seed=7919 * k + L)
                            for kind in ("nonj", "iso") for k in range(K_NULL)}
    timing["directions"] = _sec(t0)

    conds = conditions()
    jlo = {}
    for L in READ:
        S = build_all(spec, L, conds, fold)
        jlo[L] = jl.paired_logodds(S, L, *JLENS_WORDS).reshape(len(conds), N)
    timing["jlens_all"] = _sec(t0)
    Path(OUTN).mkdir(parents=True, exist_ok=True)
    torch.save(spec, f"{OUTN}/spec.pt")
    torch.save({"lo": jlo, "conds": conds}, f"{OUTN}/jlens.pt")
    VOLUMES["/results"].commit()
    return {"timing": timing, "n_conds": len(conds), "inlp": {str(k): v for k, v in spec["inlp_heldout_auroc"].items()}}


def _chunk(n: int, i: int, k: int) -> list[int]:
    return list(range(i, n, k))


def _score(reader: str, chunk: int, n_chunks: int) -> dict:
    import numpy as np
    import torch

    from normrerun_lib import build, conditions
    from smoke_items import NLA_PAIRS, OL_PAIRS

    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUTN}/spec.pt", weights_only=False)
    conds = conditions()
    mine = _chunk(len(conds), chunk, n_chunks)
    fold = spec["fold"]
    if reader == "ol":
        from mats.readers.olens import load_olens

        ol = load_olens()
        pairs = OL_PAIRS
        score = lambda A, c: ol.logprob_of_strings(A, 42, c, batch=96)  # noqa: E731
    else:
        from mats.readers import nla as NL

        nla = NL.load_nla(load_ar=False)
        pairs = NLA_PAIRS
        score = lambda A, c: NL.logprob_of_strings_batch(nla, A, c, batch=96)  # noqa: E731
    t_load = _sec(t0)
    cands = [s for p in pairs for s in p]
    A = torch.cat([build(spec, 42, conds[i], fold) for i in mine])
    t1 = time.time()
    lp = score(A, cands)  # [n_mine * N, 6]
    t_score = time.time() - t1
    T = len(pairs)
    lo = torch.stack([lp[:, 2 * t] - lp[:, 2 * t + 1] for t in range(T)])  # [T, n_mine*N]
    N = spec["h"][42].shape[0]
    lo = lo.reshape(T, len(mine), N)
    out = {"idx": mine, "lo": lo, "t_load": t_load, "t_score": t_score,
           "s_per_act": t_score / A.shape[0], "gpu": torch.cuda.get_device_name(0),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    torch.save(out, f"{OUTN}/{reader}_chunk{chunk}.pt")
    VOLUMES["/results"].commit()
    return {k: v for k, v in out.items() if k not in ("lo", "idx")} | {"n_acts": int(A.shape[0]),
                                                                      "chunk": chunk}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def score_ol(chunk: int, n_chunks: int) -> dict:
    return _score("ol", chunk, n_chunks)


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600, memory=131072)
def score_nla(chunk: int, n_chunks: int) -> dict:
    return _score("nla", chunk, n_chunks)


@app.local_entrypoint()
def main(skip_prep: bool = False, ol_chunks: int = 3, nla_chunks: int = 4) -> None:
    import subprocess

    summary = {}
    if not skip_prep:
        summary["prep"] = prep.remote()
        print(json.dumps(summary["prep"], indent=1, default=str)[:3000])
    calls = [score_ol.spawn(i, ol_chunks) for i in range(ol_chunks)]
    calls += [score_nla.spawn(i, nla_chunks) for i in range(nla_chunks)]
    summary["score"] = [c.get() for c in calls]
    print(json.dumps(summary["score"], indent=1, default=str))
    here = Path(__file__).resolve().parent / "results" / "normrerun"
    here.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mats-results", "03_pipeline_smoke/normrerun",
                    str(here.parent)], check=False)
    (here / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
