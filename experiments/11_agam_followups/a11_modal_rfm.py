"""Experiment 11, Phase B (PILOT / EXPLORATORY): the layer x position sweep, Oracle Lens on its trained
grid, free-text side-by-sides and k=8 samples. Two deployed functions on app "rfm":

  agam_main  A100-80GB  base model + J-Lens + Oracle Lens
      S1 capture 60 col + 120 tense pilot items at LAYERS x POSITIONS (+ next-token punctuation mass)
      S2 per-layer J bases (03's where they exist), class-mean / J-removed directions, cross-validated
         probes and J-Lens scores per (layer, position); J-Lens leak scores for every steering condition
      S3 OL paired log-odds on OL_LAYERS at the read token (+ position -2 at 42/44): zero, +/- jrem,
         +/- raw, K nulls (iso + J-complement) at the jrem dose
      S4 OL free text: k=4 on exp 05 natural/planted vectors (42 and 44) and 120 exp 10 operand vectors;
         k=8 on the col P2 cell (zero, +/- jrem4 q90 from the 08 spec) at 42 and 44; J-Lens top-10 lists
  agam_nla   A100-80GB  NLA (L42 only)
      S5 template LO at positions -1..-4 (zero) and +/- jrem at -1, -2; k=8 explanations on the col P1 cell

    modal deploy a11_modal_rfm.py
    python spawn_driver.py spawn agam_main ; python watch.py &
Every stage checkpoints to rfm-results:/11_agam_followups and is skipped when its file exists.
"""
import json
import time
from pathlib import Path

import modal

from env import ACTS_05, ACTS_10, BASIS_DIR, ITEMS_REMOTE, OUT, PILOT_REMOTE, VOLUMES, app_image

app = modal.App("rfm")
FN_KW = dict(image=app_image, volumes=VOLUMES, gpu="A100-80GB", max_containers=1, single_use_containers=True,
             retries=1, timeout=4 * 3600, memory=131072)
N_OPERAND = 120
SEED = 1111


def _tok():  # noqa: ANN202
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")


def _model():  # noqa: ANN202
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()


def _items() -> dict:
    return json.loads(Path(ITEMS_REMOTE).read_text())


def _labels(items: dict):  # noqa: ANN202
    import numpy as np

    import a11_modal_lib as ML
    return {"col": ML.median_split(np.array([it["col"] for it in items["col"]])),
            "tense": np.array([int(it["y"]) for it in items["tense"]])}


def _basis(layer: int, J, W_U, gain):  # noqa: ANN001, ANN202
    """03's saved basis when it exists, else the SVD basis computed here (saved for reuse)."""
    from mats import jspace as JS
    p03 = Path(f"{BASIS_DIR}/jspace_basis_L{layer}.pt")
    if p03.exists():
        return JS.load_basis_file(str(p03), "cuda"), "03"
    p11 = Path(f"{OUT}/bases/jspace_basis_L{layer}.pt")
    if p11.exists():
        return JS.load_basis_file(str(p11), "cuda"), "11"
    b = JS.jspace_basis(J.cuda(), W_U, gain, r_max=2048, layer=layer)
    p11.parent.mkdir(parents=True, exist_ok=True)
    JS.save_basis(b.to("cpu"), str(p11))
    return b.to("cuda"), "computed"


@app.function(**FN_KW)
def agam_main() -> dict:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import glob

    import numpy as np
    import torch

    import a11_modal_lib as ML
    from mats import capture as C
    from mats import jspace as JS
    from mats import probes as P
    from mats.readers.jlens import load_jlens, token_variants

    t0 = time.time()
    VOLUMES["/results"].reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    items = _items()
    y = _labels(items)
    tok = _tok()
    out: dict = {"stages": {}, "t": {}}
    L_ALL, POS = list(ML.LAYERS), list(ML.POSITIONS)
    model = None

    # ---------------- S1: capture + next-token punctuation mass
    f_acts = f"{OUT}/acts.pt"
    if not Path(f_acts).exists():
        model = _model()
        acts, mass = {}, {}
        pid = ML.punct_token_ids(lambda i: tok.decode([i]), len(tok))
        for feat in ("col", "tense"):
            seqs = [it["ids"] for it in items[feat]]
            H = C.capture(model, seqs, L_ALL, [POS] * len(seqs), batch_size=8)
            acts[feat] = {L: H[L].to(torch.float16) for L in L_ALL}
            lg = C.logits_at(model, seqs, [[-1]] * len(seqs), batch_size=8)[:, 0]
            top = lg.topk(5, dim=-1).indices
            mass[feat] = {"p_punct": ML.softmax_mass(lg, pid["punct"]).tolist(),
                          "p_newline": ML.softmax_mass(lg, pid["newline"]).tolist(),
                          "top5": [[tok.decode([int(t)]) for t in row] for row in top]}
        torch.save({"acts": acts, "mass": mass, "layers": L_ALL, "positions": POS,
                    "ids": {f: [it["id"] for it in items[f]] for f in items}}, f_acts)
        VOLUMES["/results"].commit()
        out["stages"]["S1"] = "ran"
    else:
        out["stages"]["S1"] = "skipped"
    A = torch.load(f_acts, map_location="cpu", weights_only=False)
    out["t"]["S1"] = round(time.time() - t0, 1)

    # ---------------- S2: bases, directions, probes, J-Lens per (layer, position)
    f_map = f"{OUT}/map_raw.pt"
    if not Path(f_map).exists():
        if model is None:
            model = _model()
        from safetensors import safe_open
        snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
        widx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
        ukey = [k for k in widx if k.endswith("lm_head.weight")][0]
        nkey = [k for k in widx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k and "mtp" not in k][0]

        def get(k: str):  # noqa: ANN202
            with safe_open(f"{snap}/{widx[k]}", framework="pt") as f:
                return f.get_tensor(k)
        W_U = get(ukey).cuda().float()
        gain = 1 + get(nkey).float().cuda()
        lens_path = glob.glob("/weights/hf/hub/models--neuronpedia--jacobian-lens/snapshots/*/**/*n1000*.pt", recursive=True)[0]
        Jall = JS.load_jlens_layers(lens_path, L_ALL)
        jl = load_jlens(model, tok, layers=L_ALL)
        eol = [i for i in range(len(tok)) if "\n" in tok.decode([i]) and not tok.decode([i]).strip()]
        tw = [token_variants(tok, ws) for ws in ML.JLENS_TENSE_WORDS]
        rng_seed = SEED
        map_raw: dict = {"cells": {}, "dirs": {}, "doses": {}, "nulls": {}, "basis_source": {}, "jl_scores": {}}
        for L in L_ALL:
            basis, src = _basis(L, Jall[L], W_U, gain)
            map_raw["basis_source"][L] = src
            del Jall[L]
            nulls = {"iso": [JS.random_direction(basis, ML.R_REMOVE, "iso", seed=rng_seed + 7 * k + L).cpu() for k in range(ML.K_NULL)],
                     "jc": [JS.random_direction(basis, ML.R_REMOVE, "nonj", seed=rng_seed + 7 * k + L + 3).cpu() for k in range(ML.K_NULL)]}
            map_raw["nulls"][L] = nulls
            for feat in ("col", "tense"):
                groups = np.arange(len(y[feat]))
                for pi, pos in enumerate(POS):
                    X = A["acts"][feat][L][:, pi].float().cuda()
                    v_raw = ML.class_mean_direction(X, y[feat]).cpu()
                    v_jrem = JS.j_removed_direction(basis, v_raw.cuda(), ML.R_REMOVE).cpu()
                    dirs = {"raw": v_raw, "jrem": v_jrem}
                    doses = {k: ML.q90_dose(X, v.cuda()) for k, v in dirs.items()}
                    map_raw["dirs"][(feat, L, pos)] = dirs
                    map_raw["doses"][(feat, L, pos)] = doses
                    Q = basis.Q(ML.R_REMOVE)
                    cell = {"probe_auroc": P.cv_auroc(X, y[feat], groups, k=5)["auroc"],
                            "probe_auroc_jremoved": P.cv_auroc(X, y[feat], groups, k=5, transform=lambda Z, Q=Q: Z - (Z @ Q) @ Q.T)["auroc"],
                            "enrichment_raw": float(JS.enrichment(basis, v_raw.cuda(), ML.R_REMOVE)),
                            "enrichment_raw_384": float(JS.enrichment(basis, v_raw.cuda(), 384)),
                            "dose_over_median_norm": doses["jrem"] / float(X.norm(dim=1).median())}
                    # J-Lens natural readout: col -> EOL max cosine vs label; tense -> paired word LO vs label
                    if feat == "col":
                        sc = jl.scores(X, L, ids=eol).max(1).values.cpu().numpy()
                    else:
                        sc = jl.paired_logodds(X, L, tw[0], tw[1]).cpu().numpy()
                    from mats.metrics import auroc
                    cell["jlens_natural_auroc"] = auroc(sc[y[feat] == 1], sc[y[feat] == 0])
                    # J-Lens on every steering condition at this cell (leak scores): [C, N]
                    conds = ML.build_conditions(X.cpu(), dirs, doses, nulls, null_ref="jrem")
                    names = [c[0] for c in conds]
                    S = torch.cat([c[1] for c in conds]).cuda()
                    if feat == "col":
                        js = jl.scores(S, L, ids=eol).max(1).values
                    else:
                        js = jl.paired_logodds(S, L, tw[0], tw[1])
                    map_raw["jl_scores"][(feat, L, pos)] = {"names": names, "scores": js.cpu().reshape(len(conds), -1)}
                    map_raw["cells"][(feat, L, pos)] = cell
                    del S, X
            torch.cuda.empty_cache()
        torch.save(map_raw, f_map)
        # J-Lens top-10 lists for the side-by-sides (plain model loaded here; OL's PeftModel is not needed)
        pil = torch.load(PILOT_REMOTE, map_location="cpu", weights_only=False)
        h42, v, dose = pil["h42"].float(), pil["dirs42"]["jrem4"].float(), float(pil["doses42"]["jrem4"]["q90"])
        b05 = torch.load(ACTS_05, map_location="cpu", weights_only=False)
        a10 = torch.load(ACTS_10, map_location="cpu", weights_only=False)
        H10 = (a10["H"] if isinstance(a10, dict) and "H" in a10 else a10["acts"]).float()[:N_OPERAND]
        top = {"col_zero": jl.readout(h42, 42, k=10), "col_jrem4+": jl.readout(ML.centred(h42, v, dose), 42, k=10),
               "col_jrem4-": jl.readout(ML.centred(h42, v, -dose), 42, k=10),
               "planted_natural": jl.readout(b05["acts"][:, 0].float(), 42, k=10),
               "planted_planted": jl.readout(b05["acts"][:, 1].float(), 42, k=10),
               "operand": jl.readout(H10, 42, k=10), "operand_ids": a10.get("ids") if isinstance(a10, dict) else None}
        torch.save(top, f"{OUT}/jl_top10.pt")
        VOLUMES["/results"].commit()
        del W_U, gain, jl, b05, a10, H10
        out["stages"]["S2"] = "ran"
    else:
        out["stages"]["S2"] = "skipped"
    MR = torch.load(f_map, map_location="cpu", weights_only=False)
    out["t"]["S2"] = round(time.time() - t0, 1)
    if model is not None:
        del model
        torch.cuda.empty_cache()

    # ---------------- S3: Oracle Lens on its trained grid (+42)
    from mats.readers.olens import WSBENCH_SAMPLING, load_olens
    ol = load_olens()
    f_ol = f"{OUT}/ol_sweep.pt"
    if not Path(f_ol).exists():
        res: dict = {}
        for L in ML.OL_LAYERS:
            for feat in ("col", "tense"):
                pairs = [(a, b) for _, a, b in ML.TEMPLATES[feat]["ol"]]
                for pos in ([-1, -2] if L in (42, 44) else [-1]):
                    pi = POS.index(pos)
                    X = A["acts"][feat][L][:, pi].float()
                    nulls = MR["nulls"][L] if pos == -1 else {"iso": MR["nulls"][L]["iso"][:4], "jc": MR["nulls"][L]["jc"][:4]}
                    conds = ML.build_conditions(X, MR["dirs"][(feat, L, pos)], MR["doses"][(feat, L, pos)], nulls, null_ref="jrem")
                    names = [c[0] for c in conds]
                    S = torch.cat([c[1] for c in conds]).cuda()
                    lo = ol.paired_logodds_templates(S, L, pairs).cpu()  # [T, C*N]
                    res[(feat, L, pos)] = {"names": names, "templates": ML.TEMPLATES[feat]["ol"],
                                           "lo": lo.reshape(lo.shape[0], len(conds), -1)}
                    out["t"][f"S3_{feat}_{L}_{pos}"] = round(time.time() - t0, 1)
                    torch.save(res, f_ol)  # partial checkpoints
                    VOLUMES["/results"].commit()
        out["stages"]["S3"] = "ran"
    else:
        out["stages"]["S3"] = "skipped"
    out["t"]["S3"] = round(time.time() - t0, 1)

    # ---------------- S4: OL free text (side-by-sides, k=8 cell) + J-Lens top-10 (via the OL base model's W_U)
    f_gen = f"{OUT}/ol_gens.pt"
    if not Path(f_gen).exists():
        gens: dict = {}
        pil = torch.load(PILOT_REMOTE, map_location="cpu", weights_only=False)
        h42, v, dose = pil["h42"].float(), pil["dirs42"]["jrem4"].float(), float(pil["doses42"]["jrem4"]["q90"])
        cell = {"zero": h42, "jrem4+": ML.centred(h42, v, dose), "jrem4-": ML.centred(h42, v, -dose)}
        for L in (42, 44):
            gens[f"col_k8@{L}"] = {k: ol.verbalize(X.cuda(), L, k=8, sampling=WSBENCH_SAMPLING, batch=64) for k, X in cell.items()}
            out["t"][f"S4_k8_{L}"] = round(time.time() - t0, 1)
        b05 = torch.load(ACTS_05, map_location="cpu", weights_only=False)
        nat5, pl5 = b05["acts"][:, 0].float(), b05["acts"][:, 1].float()
        for L in (42, 44):
            gens[f"planted@{L}"] = {"natural": ol.verbalize(nat5.cuda(), L, k=4, sampling=WSBENCH_SAMPLING, batch=64),
                                    "planted": ol.verbalize(pl5.cuda(), L, k=4, sampling=WSBENCH_SAMPLING, batch=64)}
        gens["planted_keep"] = b05["keep"]
        a10 = torch.load(ACTS_10, map_location="cpu", weights_only=False)
        H10 = (a10["H"] if isinstance(a10, dict) and "H" in a10 else a10["acts"]).float()[:N_OPERAND]
        gens["operand@42"] = ol.verbalize(H10.cuda(), 42, k=4, sampling=WSBENCH_SAMPLING, batch=64)
        gens["operand_ids"] = (a10.get("ids") if isinstance(a10, dict) else None)
        torch.save(gens, f_gen)
        VOLUMES["/results"].commit()
        out["stages"]["S4"] = "ran"
    else:
        out["stages"]["S4"] = "skipped"
    out["t"]["total"] = round(time.time() - t0, 1)
    Path(f"{OUT}/summary_main.json").write_text(json.dumps(out, indent=1, default=str))
    VOLUMES["/results"].commit()
    return out


@app.function(**FN_KW)
def agam_ol_bank() -> dict:
    """Point 1 on the P2 channel with the *bank-based* directions: OL nl_cont (and the fixed family) at 40 / 42 / 44
    on exp 08's spec (02's regression direction per layer, J-removed at 4r*(L), bank q90 dose, the 08 null vectors),
    i.e. the pilot's own cell re-scored at the trained neighbours. K = 20 iso + 20 J-complement (nonj4) nulls."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    import a11_modal_lib as ML
    from mats.readers.olens import load_olens

    t0 = time.time()
    VOLUMES["/results"].reload()
    f_out = f"{OUT}/ol_bank.pt"
    if Path(f_out).exists():
        return {"stages": {"bank": "skipped"}}
    sp = torch.load("/results/08_col_pilot/spec.pt", map_location="cpu", weights_only=False)
    ol = load_olens()
    pairs = [(a, b) for _, a, b in ML.TEMPLATES["col"]["ol"]]
    res: dict = {}
    for L in (40, 42, 44):
        H = sp["h"][L].float()
        dirs = {"jrem4": sp["dirs"][L]["jrem4"].float(), "raw": sp["dirs"][L]["raw"].float()}
        doses = {"jrem4": float(sp["doses"][L]["jrem4"]["q90"]), "raw": float(sp["doses"][L]["raw"]["q90"])}
        nulls = {"iso": [sp["nulls"][L][f"iso_{k}"].float() for k in range(20)],
                 "jc": [sp["nulls"][L][f"nonj4_{k}"].float() for k in range(20)]}
        conds = ML.build_conditions(H, dirs, doses, nulls, null_ref="jrem4")
        names = [c[0] for c in conds]
        S = torch.cat([c[1] for c in conds]).cuda()
        lo = ol.paired_logodds_templates(S, L, pairs).cpu()
        res[("col", L, -1)] = {"names": names, "templates": ML.TEMPLATES["col"]["ol"], "lo": lo.reshape(lo.shape[0], len(conds), -1),
                               "source": "08 spec (02 regression direction, bank q90, 08 nulls)"}
        torch.save(res, f_out)
        VOLUMES["/results"].commit()
    out = {"stages": {"bank": "ran"}, "t": round(time.time() - t0, 1)}
    Path(f"{OUT}/summary_bank.json").write_text(json.dumps(out, indent=1))
    VOLUMES["/results"].commit()
    return out


@app.function(**FN_KW)
def agam_nla() -> dict:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    import a11_modal_lib as ML
    from mats.readers import nla as NL

    t0 = time.time()
    VOLUMES["/results"].reload()
    A = torch.load(f"{OUT}/acts.pt", map_location="cpu", weights_only=False)
    MR = torch.load(f"{OUT}/map_raw.pt", map_location="cpu", weights_only=False)
    POS = A["positions"]
    out: dict = {"stages": {}, "t": {}}
    nla = NL.load_nla(load_ar=False)
    f_sw = f"{OUT}/nla_sweep.pt"
    if not Path(f_sw).exists():
        res: dict = {}
        for feat in ("col", "tense"):
            pairs = [(a, b) for _, a, b in ML.TEMPLATES[feat]["nla"]]
            for pos in POS:
                pi = POS.index(pos)
                X = A["acts"][feat][42][:, pi].float()
                if pos in (-1, -2):
                    nulls = {"iso": MR["nulls"][42]["iso"][:4], "jc": MR["nulls"][42]["jc"][:4]} if pos == -2 else MR["nulls"][42]
                    conds = ML.build_conditions(X, MR["dirs"][(feat, 42, pos)], MR["doses"][(feat, 42, pos)], nulls, null_ref="jrem")
                else:
                    conds = [("zero", X)]
                names = [c[0] for c in conds]
                S = torch.cat([c[1] for c in conds]).cuda()
                lo = NL.paired_logodds_templates(nla, S, pairs).cpu()
                res[(feat, 42, pos)] = {"names": names, "templates": ML.TEMPLATES[feat]["nla"], "lo": lo.reshape(lo.shape[0], len(conds), -1)}
                out["t"][f"S5_{feat}_{pos}"] = round(time.time() - t0, 1)
                torch.save(res, f_sw)
                VOLUMES["/results"].commit()
        out["stages"]["S5"] = "ran"
    else:
        out["stages"]["S5"] = "skipped"
    f_gen = f"{OUT}/nla_gens.pt"
    if not Path(f_gen).exists():
        pil = torch.load(PILOT_REMOTE, map_location="cpu", weights_only=False)
        h42, v, dose = pil["h42"].float(), pil["dirs42"]["jrem4"].float(), float(pil["doses42"]["jrem4"]["q90"])
        cell = {"zero": h42, "jrem4+": ML.centred(h42, v, dose), "jrem4-": ML.centred(h42, v, -dose)}
        gens = {k: nla.verbalize(X.cuda(), k=8, batch=48) for k, X in cell.items()}
        torch.save({"col_k8@42": gens}, f_gen)
        VOLUMES["/results"].commit()
        out["stages"]["S6"] = "ran"
    else:
        out["stages"]["S6"] = "skipped"
    out["t"]["total"] = round(time.time() - t0, 1)
    Path(f"{OUT}/summary_nla.json").write_text(json.dumps(out, indent=1, default=str))
    VOLUMES["/results"].commit()
    return out


def _grid_items(items: dict, n_col: int = 12, n_tense: int = 12) -> dict:
    """Deterministic viewer subset: col items evenly spaced over the col value; tense half past / half present."""
    import numpy as np
    col_order = sorted(range(len(items["col"])), key=lambda i: (int(items["col"][i]["col"]), items["col"][i]["id"]))
    col = [col_order[round(j * (len(col_order) - 1) / (n_col - 1))] for j in range(n_col)]
    rng = np.random.default_rng(SEED)
    y = [int(it["y"]) for it in items["tense"]]
    past = sorted(rng.choice([i for i, v in enumerate(y) if v == 1], n_tense // 2, replace=False).tolist())
    pres = sorted(rng.choice([i for i, v in enumerate(y) if v == 0], n_tense // 2, replace=False).tolist())
    return {"col": col, "tense": past + pres}


@app.function(**FN_KW)
def agam_grid() -> dict:
    """Viewer data for Agam (PILOT / EXPLORATORY): unsteered readouts on a small subset of the pilot col and tense
    items at every captured layer x position. J-Lens top-10 at LAYERS x POSITIONS; Oracle Lens k=2 at OL_LAYERS x
    POSITIONS; NLA k=2 at layer 42 x POSITIONS (the NLA reads block 42 only). Checkpoints to grid.pt."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    import a11_modal_lib as ML
    from mats.readers import nla as NL
    from mats.readers.jlens import load_jlens
    from mats.readers.olens import WSBENCH_SAMPLING, load_olens

    t0 = time.time()
    VOLUMES["/results"].reload()
    A = torch.load(f"{OUT}/acts.pt", map_location="cpu", weights_only=False)
    items = _items()
    POS, L_ALL = list(A["positions"]), list(A["layers"])
    sel = _grid_items(items)
    f_out = f"{OUT}/grid.pt"
    res = (torch.load(f_out, map_location="cpu", weights_only=False) if Path(f_out).exists()
           else {"sel": sel, "ids": {f: [items[f][i]["id"] for i in ix] for f, ix in sel.items()},
                 "positions": POS, "layers": L_ALL, "ol_layers": list(ML.OL_LAYERS), "jl": {}, "ol": {}, "nla": {}, "t": {}})

    def save() -> None:
        torch.save(res, f_out)
        VOLUMES["/results"].commit()

    def cells(feat: str, L: int) -> torch.Tensor:
        return A["acts"][feat][L][sel[feat]].float()  # [n, P, d]

    if not res["jl"] or not res["ol"]:
        tok = _tok()
        model = _model()
        if not res["jl"]:
            jl = load_jlens(model, tok, layers=L_ALL)
            for feat in sel:
                for L in L_ALL:
                    X = cells(feat, L)
                    for pi, pos in enumerate(POS):
                        res["jl"][(feat, L, pos)] = jl.readout(X[:, pi], L, k=10)
            del jl
            torch.cuda.empty_cache()
            res["t"]["jl"] = round(time.time() - t0, 1)
            save()
        if not res["ol"]:
            ol = load_olens(model=model, tok=tok)
            for feat in sel:
                for L in ML.OL_LAYERS:
                    X = cells(feat, L)
                    n, P = X.shape[0], X.shape[1]
                    g = ol.verbalize(X.reshape(n * P, -1).cuda(), L, k=2, sampling=WSBENCH_SAMPLING, batch=64)
                    for pi, pos in enumerate(POS):
                        res["ol"][(feat, L, pos)] = [g[i * P + pi] for i in range(n)]
                    res["t"][f"ol_{feat}_{L}"] = round(time.time() - t0, 1)
                    save()
            del ol
        del model
        torch.cuda.empty_cache()
    if not res["nla"]:
        nla = NL.load_nla(load_ar=False)
        for feat in sel:
            X = cells(feat, 42)
            n, P = X.shape[0], X.shape[1]
            g = nla.verbalize(X.reshape(n * P, -1).cuda(), k=2, batch=48)
            for pi, pos in enumerate(POS):
                res["nla"][(feat, 42, pos)] = [g[i * P + pi] for i in range(n)]
        res["t"]["nla"] = round(time.time() - t0, 1)
        save()
    res["t"]["total"] = round(time.time() - t0, 1)
    save()
    return {"t": res["t"], "n_jl": len(res["jl"]), "n_ol": len(res["ol"]), "n_nla": len(res["nla"]), "ids": res["ids"]}
