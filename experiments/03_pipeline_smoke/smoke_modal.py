"""03 pipeline smoke: end-to-end plumbing on a toy in-J feature (French vs German, 40 pairs).

Stage A (``stage_base``, H100; subject model + OL LoRA + J-Lens, one 27B copy):
  capture block outputs (38, OL sweep 20..60/4 + 42) at the last 4 tokens -> hook check vs
  hidden_states[L+1] -> J-space bases at 38/40/42/44 (+ overlap with 02's saved L42 basis) ->
  probes (G1 per layer, G2 at r*, 4r*) -> 2-fold cross-fitted directions (raw w and J-removed u)
  -> doses (q90, q99, 2 q99 of the natural projection) -> arm R steered vectors at 40/42/44
  (+ equal-norm random null at q99) -> OL paired LO (primary L42; 40/44 sensitivity; natural
  layer sweep) -> J-Lens LO (leak detector) -> TextNLA (local Qwen, prompt mode) -> mini arm U
  (steer L38, read 40/42/44: probe-shift, J-Lens leak, next-token KL) -> a few OL generations.
Stage B (``stage_nla``, H200; NLA AV + AR): NLA paired LO on every L42 condition, FVE guard
  (natural vs steered, k=4, per-example paired) on 24 items, natural explanations.

    cd experiments/03_pipeline_smoke
    modal run smoke_modal.py            # both stages, then downloads results to ./results/
    python gemini_textnla.py            # local: Gemini TextNLA (OpenRouter)
    python analyze.py                   # local: metrics -> results/metrics.json + tables
"""

import json
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, app_image

app = modal.App("mats-03-pipeline-smoke")
READ = (40, 42, 44)
OL_PRIMARY = 42
STEER_U = 38
POS = [-4, -3, -2, -1]
RSTAR_DEFAULT = {38: 192, 40: 384, 41: 384, 42: 384, 44: 256}  # 02 s8_rstar.py
CONDS = [("zero", 0, 0), ("q90", 1, 0), ("q90", -1, 0), ("q99", 1, 1), ("q99", -1, 1),
         ("2q99", 1, 2), ("2q99", -1, 2), ("null_q99", 1, 1), ("null_q99", -1, 1)]
FVE_ITEMS = 24
NLA_BASELINE_00 = 0.4616  # predict-the-mean baseline on the 00 FineWeb set


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.last = self.t0
        self.sections: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.time()
        self.sections[name] = round(now - self.last, 2)
        print(f"[{now - self.t0:7.1f}s] {name}: {now - self.last:.1f}s", flush=True)
        self.last = now

    def total(self) -> float:
        return round(time.time() - self.t0, 2)


def _rstar() -> dict[int, int]:
    p = Path("/results/02_nonj_features/rstar.json")
    if p.exists():
        lay = json.loads(p.read_text())["layers"]
        return {int(k): int(v["r_star"]) for k, v in lay.items() if v.get("r_star")}
    return dict(RSTAR_DEFAULT)


# ============================================================================ stage A


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def stage_base(n_pairs: int = 40) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C
    from mats import jspace as JS
    from mats import probes as P
    from mats.readers.jlens import load_jlens
    from mats.readers.olens import SWEEP_LAYERS, WSBENCH_SAMPLING, load_olens
    from mats.textnla import TextNLALocal, nla_instruction
    from smoke_items import JLENS_WORDS, NLA_PAIRS, OL_PAIRS, items

    torch.manual_seed(0)
    T = Timer()
    VOLUMES["/results"].reload()
    it = items()[: 2 * n_pairs]
    N = len(it)
    texts = [x["text"] for x in it]
    y = np.array([x["label"] for x in it])
    pair = np.array([x["pair"] for x in it])
    fold = pair % 2
    rstar = _rstar()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    ol = load_olens(model, tok)
    T.lap("load_model_and_olens")

    # ---------------------------------------------------------------- capture
    seqs = [tok.encode(t, add_special_tokens=False) for t in texts]
    layers = sorted({STEER_U, *READ, *SWEEP_LAYERS})
    with ol.base():
        acts = C.capture(ol.model, seqs, layers, [POS] * N, batch_size=16)
        hook_check = {}
        with torch.no_grad():
            for i in (0, 1):
                hs = ol.model(input_ids=torch.tensor([seqs[i]], device="cuda"),
                              output_hidden_states=True).hidden_states
                for L in READ:
                    ref = hs[L + 1][0, -1].float().cpu()
                    hook_check[f"item{i}_L{L}"] = float((acts[L][i, -1] - ref).norm() / ref.norm())
    T.lap("capture")
    print("hook check (rel L2 vs hidden_states[L+1], batched+padded vs single):", hook_check)

    # ---------------------------------------------------------------- J-Lens + J-space
    jl = load_jlens(ol.model, tok, layers=[STEER_U, *READ])
    gain = JS.norm_gain(jl.final_norm)
    bases = {L: JS.jspace_basis(jl.J[L], jl.W_U, gain, r_max=4096, layer=L).to("cpu")
             for L in [STEER_U, *READ]}
    Path(OUT).mkdir(parents=True, exist_ok=True)
    for L, b in bases.items():
        JS.save_basis(b, f"{OUT}/jspace_basis_L{L}.pt")
    overlap_02 = {}
    for L in (38, 42):
        p02 = Path(f"/results/02_nonj_features/jspace_basis_L{L}.pt")
        if p02.exists():
            b02 = JS.load_basis_file(str(p02))
            r = rstar.get(L, 384)
            overlap_02[L] = float((b02.Q(r).T @ bases[L].Q(r)).pow(2).sum() / r)
    T.lap("jspace_bases")
    print("subspace overlap with 02 basis at r*:", overlap_02)

    # ---------------------------------------------------------------- probes (G1/G2)
    yy, gg = np.repeat(y, len(POS)), np.repeat(pair, len(POS))
    probes = {}
    for L in layers:
        X = acts[L].reshape(-1, acts[L].shape[-1]).cuda()
        row = {"G1_auroc": P.cv_auroc(X, yy, gg)["auroc"]}
        if L in bases:
            for mult in (1, 4):
                r = rstar.get(L, 384) * mult
                Q = bases[L].Q(r).cuda()
                rm = lambda Z, Q=Q: JS.project_out_of(Q, Z)  # noqa: E731
                row[f"G2_frozen_r{r}"] = P.cv_auroc(X, yy, gg, transform=rm, retrain=False)["auroc"]
                row[f"G2_retrained_r{r}"] = P.cv_auroc(X, yy, gg, transform=rm, retrain=True)["auroc"]
                Qr = JS.random_subspace(X.shape[1], r, seed=L).cuda()
                rr = lambda Z, Q=Qr: JS.project_out_of(Q, Z)  # noqa: E731
                row[f"G2ctrl_random_frozen_r{r}"] = P.cv_auroc(X, yy, gg, transform=rr,
                                                               retrain=False)["auroc"]
        probes[L] = row
    T.lap("probes")
    print("probes:", json.dumps(probes, indent=0)[:2000])

    # ---------------------------------------------------------------- directions + doses (arm R)
    d = acts[42].shape[-1]
    steered = {L: {k: torch.zeros(len(CONDS), N, d) for k in ("raw", "jrem")} for L in [STEER_U, *READ]}
    dirs, doses, norm_ratio, enrich = {}, {}, {}, {}
    for L in [STEER_U, *READ]:
        r = rstar.get(L, 384)
        for f in (0, 1):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            Xtr = acts[L][tr].reshape(-1, d)
            w = P.mean_diff_direction(Xtr, np.repeat(y[tr], len(POS)))
            enrich[f"L{L}_f{f}"] = float(JS.enrichment(bases[L], w, r))
            for kind in ("raw", "jrem"):
                u = w if kind == "raw" else JS.j_removed_direction(bases[L], w, r)
                null = JS.random_direction(bases[L], r, "j" if kind == "raw" else "nonj",
                                           seed=1000 * L + f)
                ds = C.doses_from_projection(acts[L][tr, -1] @ u)
                dirs[(L, kind, f)] = u
                doses[f"L{L}_{kind}_f{f}"] = ds
                h = acts[L][te, -1]
                hn = float(h.norm(dim=-1).median())
                norm_ratio[f"L{L}_{kind}_f{f}"] = [abs(x) / hn for x in ds]
                for c, (_, sgn, di) in enumerate(CONDS):
                    vec = null if CONDS[c][0].startswith("null") else u
                    steered[L][kind][c, te] = h + sgn * ds[di] * vec
    T.lap("directions_doses_steer")

    # probe score on the steering contrast = projection on the fold's own direction
    probe_lo = {L: {k: torch.zeros(len(CONDS), N) for k in ("raw", "jrem")} for L in READ}
    for L in READ:
        for kind in ("raw", "jrem"):
            for f in (0, 1):
                te = np.flatnonzero(fold == f)
                probe_lo[L][kind][:, te] = steered[L][kind][:, te] @ dirs[(L, kind, f)]

    # ---------------------------------------------------------------- OL (primary 42, sens. 40/44)
    lo_ol = {}
    t_ol = time.time()
    n_ol_pairs = 0
    for L in (OL_PRIMARY, 40, 44):
        lo_ol[L] = {}
        for kind in ("raw", "jrem"):
            S = steered[L][kind].reshape(-1, d)
            lo_ol[L][kind] = ol.paired_logodds_templates(S, L, OL_PAIRS).reshape(len(OL_PAIRS), len(CONDS), N)
            n_ol_pairs += S.shape[0] * len(OL_PAIRS)
    ol_s_per_pair = (time.time() - t_ol) / n_ol_pairs
    T.lap("olens_steered_scoring")
    lo_ol_sweep = ol.layer_sweep_logodds({L: acts[L][:, -1] for L in SWEEP_LAYERS}, OL_PAIRS)
    T.lap("olens_natural_layer_sweep")

    # ---------------------------------------------------------------- J-Lens
    lo_jl, lo_jl_nat, jl_top = {}, {}, {}
    for L in READ:
        lo_jl[L] = {k: jl.paired_logodds(steered[L][k].reshape(-1, d), L, *JLENS_WORDS)
                    .reshape(len(CONDS), N) for k in ("raw", "jrem")}
        lo_jl_nat[L] = jl.paired_logodds(acts[L][:, -1], L, *JLENS_WORDS)
        jl_top[L] = jl.readout(acts[L][:2, -1], L, k=10)
    T.lap("jlens")

    # ---------------------------------------------------------------- TextNLA (local, prompt mode)
    with ol.base():
        tn = TextNLALocal(ol.model, tok, instruction=nla_instruction())
        lo_textnla = tn.paired_logodds_templates(texts, NLA_PAIRS, mode="prompt")
    T.lap("textnla_local")

    # ---------------------------------------------------------------- mini arm U (steer L38)
    armU = {}
    with ol.base():
        clean = C.capture(ol.model, seqs, list(READ), [[-1]] * N, batch_size=16)
        lg_clean = C.logits_at(ol.model, seqs, [[-1]] * N, batch_size=16)[:, 0]
        for kind in ("jrem", "raw"):
            res = {}
            for sgn in (1, -1):
                V = torch.zeros(N, d)
                inj = np.zeros(N)
                for f in (0, 1):
                    te = np.flatnonzero(fold == f)
                    dq = doses[f"L{STEER_U}_{kind}_f{f}"][1]  # q99
                    V[te] = sgn * dq * dirs[(STEER_U, kind, f)]
                    inj[te] = dq
                with C.steer(ol.model, STEER_U, V, [[-1]] * N):
                    cap = C.capture(ol.model, seqs, list(READ), [[-1]] * N, batch_size=16)
                    lg = C.logits_at(ol.model, seqs, [[-1]] * N, batch_size=16)[:, 0]
                res[sgn] = {"cap": cap, "kl": C.next_token_kl(lg_clean, lg)}
            out = {"kl_mean": float(torch.cat([res[1]["kl"], res[-1]["kl"]]).mean()),
                   "injected_q99": inj.tolist()}
            for L in READ:
                shift = torch.zeros(N)
                for f in (0, 1):
                    te = np.flatnonzero(fold == f)
                    u_read = dirs[(L, "raw", f)]
                    shift[te] = ((res[1]["cap"][L][te, 0] - res[-1]["cap"][L][te, 0]) @ u_read) / 2
                out[f"L{L}_probe_shift_frac"] = (shift / torch.tensor(inj, dtype=torch.float32)).tolist()
                out[f"L{L}_jlens_lo_plus"] = jl.paired_logodds(res[1]["cap"][L][:, 0], L, *JLENS_WORDS).tolist()
                out[f"L{L}_jlens_lo_minus"] = jl.paired_logodds(res[-1]["cap"][L][:, 0], L, *JLENS_WORDS).tolist()
                out[f"L{L}_jlens_lo_zero"] = jl.paired_logodds(clean[L][:, 0], L, *JLENS_WORDS).tolist()
            armU[kind] = out
    T.lap("arm_u_L38")

    # ---------------------------------------------------------------- a few OL generations
    t_gen = time.time()
    gens = ol.verbalize(acts[OL_PRIMARY][:16, -1], OL_PRIMARY, k=1, sampling=WSBENCH_SAMPLING, batch=16)
    ol_gen_s = (time.time() - t_gen) / 16
    T.lap("olens_generate_16")

    payload = {
        "texts": texts, "y": y, "pair": pair, "fold": fold, "conds": CONDS, "rstar": rstar,
        "hook_check": hook_check, "overlap_02": overlap_02, "probes": probes, "doses": doses,
        "norm_ratio": norm_ratio, "enrichment_w": enrich,
        "lo_ol": lo_ol, "lo_ol_sweep": lo_ol_sweep, "lo_jl": lo_jl, "lo_jl_nat": lo_jl_nat,
        "probe_lo": probe_lo, "lo_textnla": lo_textnla, "armU": armU, "ol_gens": gens,
        "jl_top": jl_top, "steered42": steered[42], "natural42": acts[42][:, -1],
        "timing": {"sections": T.sections, "total_s": T.total(), "ol_s_per_pair": ol_s_per_pair,
                   "ol_gen_s_per_row_b16": ol_gen_s, "n_ol_pairs": n_ol_pairs},
        "gpu": torch.cuda.get_device_name(0),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    torch.save(payload, f"{OUT}/stage_base.pt")
    VOLUMES["/results"].commit()
    return {"timing": payload["timing"], "peak_mem_gb": payload["peak_mem_gb"], "gpu": payload["gpu"],
            "hook_check": hook_check, "overlap_02": overlap_02}


# ============================================================================ stage B


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600, memory=131072)
def stage_nla() -> dict:
    import numpy as np
    import torch

    from mats.readers import nla as NL
    from smoke_items import NLA_PAIRS

    torch.manual_seed(0)
    T = Timer()
    VOLUMES["/results"].reload()
    base = torch.load(f"{OUT}/stage_base.pt", weights_only=False)
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=True)
    T.lap("load_nla")
    N = len(base["texts"])
    d = base["natural42"].shape[-1]
    lo_nla, n_pairs = {}, 0
    t0 = time.time()
    for kind in ("raw", "jrem"):
        S = base["steered42"][kind].reshape(-1, d)
        lo_nla[kind] = NL.paired_logodds_templates(nla, S, NLA_PAIRS).reshape(
            len(NLA_PAIRS), len(CONDS), N)
        n_pairs += S.shape[0] * len(NLA_PAIRS)
    nla_s_per_pair = (time.time() - t0) / n_pairs
    T.lap("nla_scoring")

    # FVE guard: natural (cond 0) vs the 6 jrem steered conditions, same 24 items, k=4
    sel = np.arange(FVE_ITEMS)
    st = base["steered42"]["jrem"][:7, sel]  # [7, n, d]: zero, q90+-, q99+-, 2q99+-
    t0 = time.time()
    out = nla.fve_on(st.reshape(-1, d), k=4, baseline=NLA_BASELINE_00)
    n_gen = st.shape[0] * st.shape[1] * 4
    gen_s = (time.time() - t0) / n_gen
    per_ex = out["per_example"].reshape(7, len(sel))
    own_base = NL.predict_mean_baseline(st[0])
    T.lap("fve_guard")
    expl = out["explanations"]
    payload = {
        "lo_nla": lo_nla, "fve_per_example": per_ex, "fve_items": sel, "fve_n_failed": out["n_failed"],
        "fve_baseline": NLA_BASELINE_00, "own_baseline_natural": own_base,
        "natural_explanations": [expl[i] for i in range(len(sel))],
        "steered_explanations_2q99_plus": [expl[5 * len(sel) + i] for i in range(len(sel))],
        "timing": {"sections": T.sections, "total_s": T.total(), "nla_s_per_pair": nla_s_per_pair,
                   "nla_gen_plus_recon_s_per_sample": gen_s, "n_gen": n_gen},
        "gpu": torch.cuda.get_device_name(0),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    torch.save(payload, f"{OUT}/stage_nla.pt")
    VOLUMES["/results"].commit()
    return {"timing": payload["timing"], "peak_mem_gb": payload["peak_mem_gb"], "gpu": payload["gpu"]}


@app.local_entrypoint()
def main(stage: str = "all", n_pairs: int = 40) -> None:
    import subprocess

    res = {}
    if stage in ("all", "base"):
        res["base"] = stage_base.remote(n_pairs)
        print(json.dumps(res["base"], indent=1, default=str))
    if stage in ("all", "nla"):
        res["nla"] = stage_nla.remote()
        print(json.dumps(res["nla"], indent=1, default=str))
    here = Path(__file__).resolve().parent / "results"
    here.mkdir(exist_ok=True)
    for f in ("stage_base.pt", "stage_nla.pt"):
        subprocess.run(["modal", "volume", "get", "--force", "mats-results",
                        f"03_pipeline_smoke/{f}", str(here / f)], check=False)
    (here / f"run_summary_{stage}.json").write_text(json.dumps(res, indent=1, default=str))
