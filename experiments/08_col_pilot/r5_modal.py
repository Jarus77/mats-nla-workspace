"""08 round-5 pre-freeze pilot (Fable round 5, D21) -- DECLARED PILOT SPLIT, EXCLUDED FROM MAIN ANALYSIS.

Tasks:
  1 behavioural validation of every col cell (L42 read-layer edit -> newline log-odds at the next position;
    L38 upstream edit -> newline log-odds and wrap column in greedy generation)
  2 class-mean (short <= 20 vs long >= 45) col direction, raw and J-removed 4r*, reader cells at r11 / q90
  3 raw-direction newline-decision cell: OL newline-continuation candidates + 07 frozen break templates
  4 next-digit in-J number cell (numbered lists, next item 5 vs 7): gates G1/G2/J-Lens/G4 + reader cells
  5 OL tense with 07's frozen continuation templates on 120 new items (NLA too)

    modal run r5_modal.py
    ../../.venv/bin/python r5_analyze.py
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, OUT02, OUT07, VOLUMES, app_image, build_image

app = modal.App("mats-08-r5")
L = 42
L_UP = 38
SETS = ("col", "digit", "tense")
RANKS = (384, 1536)
GEN_TOKENS = 48


def _sec(t0: float) -> float:
    return round(time.time() - t0, 1)


@app.function(image=build_image, volumes=VOLUMES, timeout=3600, memory=32768, cpu=8)
def build_r5() -> dict:
    from transformers import AutoTokenizer

    import r5_items as RI

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    col_items = json.loads(Path(f"{OUT}/items.json").read_text())
    calib = json.loads(Path(f"{OUT}/calib_items.json").read_text())
    excl = {it["source_sha1_16"] for it in calib["tense"]} | {it["passage_sha1_16"] for it in col_items}
    tense = RI.build_tense120(tok, OUT02, OUT07, col_items, excl)
    excl |= {it["source_sha1_16"] for it in tense}
    digit = RI.build_digit(tok, OUT02, OUT07, excl)
    items = {"digit": digit, "tense": tense}
    Path(f"{OUT}/r5_items.json").write_text(json.dumps(items))
    VOLUMES["/results"].commit()
    return items


# ----------------------------------------------------------------------------- helpers (GPU)
def _wrap_bank(layer: int, n_items: int | None = None):  # noqa: ANN202
    """02 wrap bank col rows: (X [n, d] cuda fp32 at block `layer` from 02's captures, y)."""
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
    R = rows_for(FEATURES_BY_NAME["col"], items, index, np.random.default_rng(0))
    X = torch.from_numpy(np.load(f"{d}/hs{layer + 1}.npy", mmap_mode="r")[R["rows"]].astype(np.float32)).cuda()
    return X, torch.as_tensor(R["y"]).float().cuda(), items, index, R


def _cov_dir(X, y):  # noqa: ANN001, ANN202
    yc = y - y.mean()
    c = ((X - X.mean(0)) * yc[:, None]).mean(0) / yc.var()
    return (c / c.norm()).cpu()


def _doses(proj) -> dict:  # noqa: ANN001
    from mats import capture as C

    dq = C.doses_from_projection(proj, quantiles=(0.9, 0.99), extra_multiple=None)
    return {"q90": dq[0], "q99": dq[1]}


def _last_logits(model, seqs, batch: int = 16):  # noqa: ANN001, ANN202
    import torch

    with torch.no_grad():
        return _last_logits_impl(model, seqs, batch)


def _last_logits_impl(model, seqs, batch: int = 16):  # noqa: ANN001, ANN202
    """[N, V] fp32 next-token logits at each sequence's last token (edits in scope apply)."""
    import torch

    from mats import capture as C

    out = []
    for s in range(0, len(seqs), batch):
        rows = list(range(s, min(len(seqs), s + batch)))
        ids, mask = C._pad([seqs[r] for r in rows], 0, "cuda")
        with C._batch_context(rows, [len(seqs[r]) for r in rows]), torch.no_grad():
            hs = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        last = hs[torch.arange(len(rows)), mask.sum(1) - 1]
        out.append(model.lm_head(last).float().cpu())
    return torch.cat(out)


def _edit_fn(v, dose, variant: str):  # noqa: ANN001, ANN202
    """dose: float, or ("rel", r) for r * ||h||."""
    def fn(h, _r):  # noqa: ANN001, ANN202
        vv = v.to(h.device).float()
        dd = dose[1] * h.norm(dim=-1, keepdim=True) if isinstance(dose, tuple) else dose
        if variant == "centred":
            return h - (h @ vv)[:, None] * vv[None] + dd * vv[None]
        return h + dd * vv[None]
    return fn


def _readout(lg, ids_a, ids_b=None):  # noqa: ANN001, ANN202
    """log-odds of the token class ids_a vs everything else (or vs ids_b)."""
    import torch

    lp = torch.log_softmax(lg, -1)
    a = torch.logsumexp(lp[:, ids_a], -1)
    if ids_b is None:
        return a - torch.log1p(-a.exp().clamp(max=1 - 1e-6))
    return a - torch.logsumexp(lp[:, ids_b], -1)


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=196608)
def prep_r5() -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C
    from mats import jspace as JS
    from mats import probes as P
    from mats.metrics import auroc, auroc_labels
    from mats.readers.jlens import load_jlens, token_variants
    from pilot_lib import eol_token_ids
    from r5_lib import build, conditions
    from r5_templates import JLENS_WORDS

    t0 = time.time()
    VOLUMES["/results"].reload()
    items = json.loads(Path(f"{OUT}/r5_items.json").read_text())
    colspec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    calspec = torch.load(f"{OUT}/calib_spec.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    b42 = JS.load_basis_file(f"/results/03_pipeline_smoke/jspace_basis_L{L}.pt")
    b38 = JS.load_basis_file(f"/results/03_pipeline_smoke/jspace_basis_L{L_UP}.pt")
    nulls = {k: v for k, v in colspec["nulls"][L].items() if k.startswith(("iso_", "nonj4_"))}
    checks, spec = {}, {}

    # ---------------- col: regression (02) + class-mean directions at L42
    X, y, wrap_items, wrap_index, R = _wrap_bank(L)
    reg = colspec["dirs"][L]["raw"]
    cm = X[y >= 45].mean(0) - X[y <= 20].mean(0)
    cm = (cm / cm.norm()).cpu()
    dirs = {"reg_raw": reg, "reg_jrem4": colspec["dirs"][L]["jrem4"], "cm_raw": cm,
            "cm_jrem4": JS.j_removed_direction(b42, cm, RANKS[1])}
    doses = {k: _doses((X @ v.cuda()).cpu()) for k, v in dirs.items()}
    checks["col_cm"] = {"n_short": int((y <= 20).sum()), "n_long": int((y >= 45).sum()),
                        "cos_cm_reg": float(cm @ reg), "cos_cmj4_regj4": float(dirs["cm_jrem4"] @ dirs["reg_jrem4"]),
                        "enrichment": {k: {f"r{r}": float(JS.enrichment(b42, v, r)) for r in RANKS} for k, v in dirs.items()},
                        "dose_over_median_norm": {k: {q: v / float(X.norm(dim=1).median()) for q, v in dd.items()}
                                                  for k, dd in doses.items()}}
    hcol = colspec["h"][L]
    col_items = colspec["items"]
    ycol = np.array([it["col"] for it in col_items], float)
    checks["col_cm"]["pilot_natural_spearman"] = {
        k: float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr((hcol @ v).numpy(), ycol).statistic)
        for k, v in dirs.items()}
    spec["col"] = {"h": hcol, "dirs": dirs, "doses": doses, "nulls": nulls, "items": col_items, "y": ycol}
    del X

    # ---------------- digit: capture, directions, gates
    dig = items["digit"]
    tr, ev = dig["train"], dig["eval"]
    acts = C.capture(model, [it["ids"] for it in tr + ev], [L], [[-1]] * (len(tr) + len(ev)), batch_size=16)[L][:, 0]
    Htr, Hev = acts[: len(tr)], acts[len(tr):]
    ytr, yev = np.array([it["y"] for it in tr]), np.array([it["y"] for it in ev])
    md = P.mean_diff_direction(Htr, ytr).cpu()
    dd_ = {"raw": md, "jrem4": JS.j_removed_direction(b42, md, RANKS[1])}
    ddoses = {k: _doses((Htr @ v)) for k, v in dd_.items()}
    m = P.fit_logreg(Htr.cuda(), torch.as_tensor(ytr).cuda())
    g = {"G1_meandiff_eval_auroc": auroc_labels((Hev @ md).numpy(), yev),
         "G1_logreg_eval_auroc": auroc_labels(m.decision(Hev.cuda()).cpu().numpy(), yev),
         "G1_jrem4_eval_auroc": auroc_labels((Hev @ dd_["jrem4"]).numpy(), yev),
         "enrichment": {f"r{r}": float(JS.enrichment(b42, md, r)) for r in RANKS},
         "cos_raw_jrem4": float(md @ dd_["jrem4"]),
         "dose_over_median_norm": {k: {q: v / float(Htr.norm(dim=1).median()) for q, v in d.items()} for k, d in ddoses.items()}}
    rng = np.random.default_rng(0)
    shuf = []
    for _ in range(10):
        ys = rng.permutation(ytr)
        shuf.append({f"r{r}": float(JS.enrichment(b42, P.mean_diff_direction(Htr, ys).cpu(), r)) for r in RANKS})
    g["enrichment_shuffled_mean"] = {f"r{r}": float(np.mean([s[f"r{r}"] for s in shuf])) for r in RANKS}
    g["enrichment_shuffled_max"] = {f"r{r}": float(np.max([s[f"r{r}"] for s in shuf])) for r in RANKS}
    checks["digit_gates"] = g
    spec["digit"] = {"h": Hev, "dirs": dd_, "doses": ddoses, "nulls": nulls, "items": ev, "y": yev}

    # ---------------- tense120
    ten = items["tense"]
    Ht = C.capture(model, [it["ids"] for it in ten], [L], [[-1]] * len(ten), batch_size=16)[L][:, 0]
    tdirs = calspec["tense"]["dirs"][L]
    spec["tense"] = {"h": Ht, "dirs": {"raw": tdirs["raw"], "jrem4": tdirs["jrem4"]},
                     "doses": {k: calspec["tense"]["doses"][L][k] for k in ("raw", "jrem4")}, "nulls": nulls,
                     "items": ten, "y": np.array([it["y"] for it in ten])}
    checks["tense120_rawdir_natural_auroc"] = auroc_labels((Ht @ tdirs["raw"]).numpy(), spec["tense"]["y"])
    timing = {"directions": _sec(t0)}

    # ---------------- L38 col directions (02 has no L38 capture: capture 300 wrap items at L38)
    sub = list(range(300))
    seqs38 = [wrap_items[i]["ids"] for i in sub]
    pos_rows = [(int(wrap_index["item"][r]), int(wrap_index["pos"][r]), float(yy))
                for r, yy in zip(R["rows"], R["y"]) if int(wrap_index["item"][r]) < 300]
    blocks = C.decoder_blocks(model)
    feats, ys38 = [], []
    by_item: dict[int, list] = {}
    for it_, p_, yy in pos_rows:
        by_item.setdefault(it_, []).append((p_, yy))
    store = {}
    hk = blocks[L_UP].register_forward_hook(lambda _m, _i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    try:
        for s in range(0, len(sub), 8):
            rows = sub[s:s + 8]
            ids, mask = C._pad([seqs38[r] for r in rows], 0, "cuda")
            with torch.no_grad():
                model.model(input_ids=ids, attention_mask=mask)
            for b, r in enumerate(rows):
                for p_, yy in by_item.get(r, []):
                    feats.append(store["h"][b, p_].float())
                    ys38.append(yy)
    finally:
        hk.remove()
    X38, y38 = torch.stack(feats), torch.tensor(ys38, device="cuda")
    r38 = _cov_dir(X38, y38)
    dirs38 = {"raw": r38, "jrem1": JS.j_removed_direction(b38, r38, RANKS[0]),
              "jrem4": JS.j_removed_direction(b38, r38, RANKS[1])}
    doses38 = {k: _doses((X38 @ v.cuda()).cpu()) for k, v in dirs38.items()}
    checks["L38"] = {"n_rows": len(ys38), "enrichment": {k: {f"r{r}": float(JS.enrichment(b38, v, r)) for r in RANKS}
                                                          for k, v in dirs38.items()},
                     "dose_over_median_norm": {k: {q: v / float(X38.norm(dim=1).median()) for q, v in d.items()}
                                               for k, d in doses38.items()}}
    nulls38 = {}
    for k in range(20):
        nulls38[f"iso_{k}"] = JS.random_direction(b38, RANKS[0], "iso", seed=7777 * k + 1)
        nulls38[f"nonj1_{k}"] = JS.random_direction(b38, RANKS[0], "nonj", seed=7777 * k + 2)
        nulls38[f"nonj4_{k}"] = JS.random_direction(b38, RANKS[1], "nonj", seed=7777 * k + 3)
    spec["L38"] = {"dirs": dirs38, "doses": doses38, "nulls": {k: v.cpu() for k, v in nulls38.items()}}
    del X38
    timing["L38_dirs"] = _sec(t0)

    # ---------------- J-Lens for every reader-cell condition
    jl = load_jlens(model, tok, layers=[L])
    eol = eol_token_ids(tok)["broad"]
    jout = {}
    for S in SETS:
        conds = conditions(S)
        temps = [(fam, a, b) for fam, lst in JLENS_WORDS[S].items() for a, b in lst]
        A = torch.cat([build(spec[S], c) for c in conds])
        lo = torch.stack([jl.paired_logodds(A, L, token_variants(tok, a), token_variants(tok, b)) for _, a, b in temps])
        jout[S] = {"conds": conds, "templates": temps, "lo": lo.reshape(len(temps), len(conds), -1)}
        if S == "col":
            sc = torch.cat([jl.scores(A[i:i + 4096], L, ids=eol).max(1).values for i in range(0, A.shape[0], 4096)])
            jout[S]["eol"] = sc.reshape(len(conds), -1)
    # digit natural J-Lens readability (G3-style)
    d5, d7 = token_variants(tok, ["5", "five"]), token_variants(tok, ["7", "seven"])
    lo_nat = jl.paired_logodds(Hev, L, d7, d5).numpy()
    digits = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    rk = jl.rank_of(Hev, L, digits).numpy()
    checks["digit_gates"]["jlens_natural_auroc_7v5"] = auroc_labels(lo_nat, yev)
    checks["digit_gates"]["jlens_any_digit_top10_rate"] = float((rk < 10).mean())
    checks["digit_gates"]["jlens_any_digit_best_rank_median"] = float(np.median(rk))
    checks["digit_gates"]["jlens_top10_examples"] = [[t for t, _ in r] for r in jl.readout(Hev[:4], L, k=10)]
    timing["jlens"] = _sec(t0)
    torch.save(spec, f"{OUT}/r5_spec.pt")
    torch.save(jout, f"{OUT}/r5_jlens.pt")
    VOLUMES["/results"].commit()
    return {"timing": timing, "checks": checks}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def behave(part: int) -> dict:
    """Behavioural validation (realised effect on the model's own next-token output).
    part 0: col at L42 read-layer edit (pilot directions raw/r*/4r*, q90/q99, centred +-, with nulls;
            regression + class-mean directions at r11 with nulls)
    part 1: col at L38 upstream edit at the read token (centred and additive), with nulls
    part 2: digit G4 at L42 (log P(7) - log P(5), argmax digit), with nulls"""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C
    from pilot_lib import eol_token_ids

    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/r5_spec.pt", weights_only=False)
    colspec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    eol = eol_token_ids(tok)["broad"]
    res: dict = {}

    def run(seqs, layer, v, dose, variant, readout):  # noqa: ANN001, ANN202
        N = len(seqs)
        if v is None:
            lg = _last_logits(model, seqs)
        else:
            with C.edit(model, layer, _edit_fn(v, dose, variant), positions=[[-1]] * N):
                lg = _last_logits(model, seqs)
        return readout(lg)

    if part in (0, 1):
        items = colspec["items"]
        seqs = [it["ids"] for it in items]
        ro = lambda lg: _readout(lg, eol).numpy()  # noqa: E731
        res["zero"] = run(seqs, None, None, 0, "centred", ro)
        if part == 0:
            D, Dz, NUL = colspec["dirs"][L], colspec["doses"][L], colspec["nulls"][L]
            ncls = {"raw": "iso", "jrem1": "nonj1", "jrem4": "nonj4"}
            for d in ("raw", "jrem1", "jrem4"):
                for q in ("q90", "q99"):
                    for s in (1, -1):
                        res[("L42", d, q, s)] = run(seqs, L, D[d], s * Dz[d][q], "centred", ro)
                for k in range(20):
                    for s in (1, -1):
                        res[("L42null", d, ncls[d], k, s)] = run(seqs, L, NUL[f"{ncls[d]}_{k}"], s * Dz[d]["q90"], "centred", ro)
            sp = spec["col"]
            for d in sp["dirs"]:
                for s in (1, -1):
                    res[("L42r11", d, s)] = run(seqs, L, sp["dirs"][d], ("rel", 0.11 * s), "centred", ro)
            for cls in ("iso", "nonj4"):
                for k in range(20):
                    for s in (1, -1):
                        res[("L42r11null", cls, k, s)] = run(seqs, L, sp["nulls"][f"{cls}_{k}"], ("rel", 0.11 * s), "centred", ro)
        else:
            S38 = spec["L38"]
            for variant in ("centred", "add"):
                for d in ("raw", "jrem1", "jrem4"):
                    for q in ("q90", "q99"):
                        for s in (1, -1):
                            res[("L38", variant, d, q, s)] = run(seqs, L_UP, S38["dirs"][d], s * S38["doses"][d][q], variant, ro)
            ncls = {"raw": "iso", "jrem1": "nonj1", "jrem4": "nonj4"}
            for d in ("raw", "jrem1", "jrem4"):
                for k in range(20):
                    for s in (1, -1):
                        res[("L38null", d, ncls[d], k, s)] = run(seqs, L_UP, S38["nulls"][f"{ncls[d]}_{k}"],
                                                                 s * S38["doses"][d]["q90"], "centred", ro)
    else:
        sp = spec["digit"]
        seqs = [it["ids"] for it in sp["items"]]
        i5 = [tok.encode("5", add_special_tokens=False)[0]]
        i7 = [tok.encode("7", add_special_tokens=False)[0]]
        digits = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(10)]

        def ro(lg):  # noqa: ANN001, ANN202
            return {"lo75": _readout(lg, i7, i5).numpy(), "argmax_digit": lg[:, digits].argmax(1).numpy(),
                    "p_digit_top1": (lg.argmax(1)[:, None] == torch.tensor(digits)[None]).any(1).numpy()}

        res["zero"] = run(seqs, None, None, 0, "centred", ro)
        for d in ("raw", "jrem4"):
            for q in ("q90", "r11", "r14"):
                dose = sp["doses"][d]["q90"] if q == "q90" else None
                for s in (1, -1):
                    dd = s * dose if q == "q90" else ("rel", (0.11 if q == "r11" else 0.14) * s)
                    res[("digit", d, q, s)] = run(seqs, L, sp["dirs"][d], dd, "centred", ro)
        for cls in ("iso", "nonj4"):
            for k in range(20):
                for s in (1, -1):
                    res[("digitnull", cls, k, s)] = run(seqs, L, sp["nulls"][f"{cls}_{k}"], ("rel", 0.11 * s), "centred", ro)
    res["_time"] = _sec(t0)
    torch.save(res, f"{OUT}/r5_behave{part}.pt")
    VOLUMES["/results"].commit()
    return {"part": part, "n": len(res), "t": res["_time"]}


GEN_CONDS = [("zero", None, None, 0)] + [("add", d, q, s) for d in ("raw", "jrem1", "jrem4")
                                          for q in ("q90", "q99") for s in (1, -1)]


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def gen_l38(part: int, n_parts: int) -> dict:
    """Greedy continuation (48 tokens, batch 1) with an ADDITIVE L38 steer at the read token and every
    generated token (a constant shift of the perceived column); chars until the first newline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C

    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/r5_spec.pt", weights_only=False)
    colspec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    S38 = spec["L38"]
    blk = C.decoder_blocks(model)[L_UP]
    state = {"v": None}

    def hook(_m, _i, o):  # noqa: ANN001, ANN202
        if state["v"] is None:
            return o
        h = o[0] if isinstance(o, tuple) else o
        h = h.clone()
        h[:, -1] = (h[:, -1].float() + state["v"]).to(h.dtype)
        return (h, *o[1:]) if isinstance(o, tuple) else h

    hd = blk.register_forward_hook(hook)
    out = {}
    try:
        for ci in range(part, len(GEN_CONDS), n_parts):
            kind, d, q, s = GEN_CONDS[ci]
            state["v"] = None if kind == "zero" else (s * S38["doses"][d][q] * S38["dirs"][d]).cuda().float()
            texts = []
            for it in colspec["items"]:
                ids = torch.tensor([it["ids"]], device="cuda")
                with torch.no_grad():
                    g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                       max_new_tokens=GEN_TOKENS, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                texts.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
            out[GEN_CONDS[ci]] = texts
    finally:
        hd.remove()
    torch.save({"texts": out, "t": _sec(t0)}, f"{OUT}/r5_gen{part}.pt")
    VOLUMES["/results"].commit()
    return {"part": part, "conds": len(out), "t": _sec(t0)}


def _jobs(reader: str) -> list[tuple]:
    from r5_lib import conditions

    return [(S, i) for S in SETS for i in range(len(conditions(S)))]


def _score(reader: str, chunk: int, n_chunks: int) -> dict:
    import torch

    from r5_lib import build, conditions
    from r5_templates import NLA_PAIRS, OL_PAIRS, flat

    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/r5_spec.pt", weights_only=False)
    jobs = _jobs(reader)[chunk::n_chunks]
    if reader == "ol":
        from mats.readers.olens import load_olens

        ol = load_olens()
        PAIRS = OL_PAIRS
        score = lambda A, c: ol.logprob_of_strings(A, L, c, batch=112)  # noqa: E731
    else:
        from mats.readers import nla as NL

        nla = NL.load_nla(load_ar=False)
        PAIRS = NLA_PAIRS
        score = lambda A, c: NL.logprob_of_strings_batch(nla, A, c, batch=112)  # noqa: E731
    out = {"lo": {}, "templates": {S: flat(PAIRS[S]) for S in SETS}}
    n_acts = 0
    for S in SETS:
        idx = [i for SS, i in jobs if SS == S]
        if not idx:
            continue
        fp = flat(PAIRS[S])
        cands = [x for _, a, b in fp for x in (a, b)]
        conds = conditions(S)
        A = torch.cat([build(spec[S], conds[i]) for i in idx])
        lp = score(A, cands)
        lo = torch.stack([lp[:, 2 * t] - lp[:, 2 * t + 1] for t in range(len(fp))])
        out["lo"][S] = {"idx": idx, "lo": lo.reshape(len(fp), len(idx), -1)}
        n_acts += A.shape[0]
    out |= {"total_s": _sec(t0), "n_acts": n_acts}
    torch.save(out, f"{OUT}/r5_{reader}_chunk{chunk}.pt")
    VOLUMES["/results"].commit()
    return {"chunk": chunk, "reader": reader, "n_acts": n_acts, "total_s": out["total_s"]}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def score_ol(chunk: int, n_chunks: int) -> dict:
    return _score("ol", chunk, n_chunks)


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600, memory=131072)
def score_nla(chunk: int, n_chunks: int) -> dict:
    return _score("nla", chunk, n_chunks)


@app.local_entrypoint()
def main(skip_build: bool = False, skip_prep: bool = False, ol_chunks: int = 8, nla_chunks: int = 12,
         gen_parts: int = 4) -> None:
    import subprocess

    here = Path(__file__).resolve().parent
    summary = {}
    if not skip_build:
        items = build_r5.remote()
        print({"digit_train": len(items["digit"]["train"]), "digit_eval": len(items["digit"]["eval"]),
               "tense": len(items["tense"])})
        split = json.loads((here / "pilot_split.json").read_text())
        split["round5_items"] = {
            "note": "Round-5 pilot (r5_items.py): digit lists (train = direction/gates, eval = reader cells) and "
                    "120 tense items. ALL EXCLUDED from the main analysis.",
            "digit": [{"id": it["id"], "sent_heads": it["sent_heads"]} for sp in ("train", "eval") for it in items["digit"][sp]],
            "tense": [{k: it[k] for k in ("id", "label", "pos", "source_sha1_16", "source_head")} for it in items["tense"]]}
        (here / "pilot_split.json").write_text(json.dumps(split, indent=1))
    if not skip_prep:
        summary["prep"] = prep_r5.remote()
        print(json.dumps(summary["prep"], indent=1, default=str)[:10000])
    calls = [behave.spawn(p) for p in range(3)] + [gen_l38.spawn(i, gen_parts) for i in range(gen_parts)]
    calls += [score_ol.spawn(i, ol_chunks) for i in range(ol_chunks)]
    calls += [score_nla.spawn(i, nla_chunks) for i in range(nla_chunks)]
    summary["jobs"] = []
    for c in calls:
        try:
            summary["jobs"].append(c.get())
        except Exception as e:  # noqa: BLE001
            summary["jobs"].append({"error": repr(e)[:2000]})
    print(json.dumps(summary["jobs"], indent=1, default=str))
    subprocess.run(["modal", "volume", "get", "--force", "rfm-results", "08_col_pilot", str(here / "results")], check=False)
    (here / "results" / "r5_run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
