"""08 matched-dose in-J calibration (K3) -- DECLARED PILOT SPLIT, EXCLUDED FROM MAIN ANALYSIS.

Same pipeline/cells as the col pilot (L42, centred, q90 primary / q99 secondary, K=20 iso +
J-complement(4r*) null bands, fraction of probe; NLA@42, OL@42 (+40/44 real-only), J-Lens 40/42/44).
Features: lang (02 direction Russian - Polish), tense (Past - Pres), and col (dose-matched arm only).
Directions: raw and J-removed at 4r*. Doses: q90/q99 of the 02 bank projection, plus per-item
relative-norm doses 0.11 / 0.14 * ||h|| (dose-matched across features).

    modal run calib_modal.py
    ../../.venv/bin/python calib_analyze.py
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, OUT02, OUT07, VOLUMES, app_image, build_image

app = modal.App("mats-08-calib")
READ = (40, 42, 44)
FEATS = ("lang", "tense", "col")
BANK = {"lang": ("lang", "lang"), "tense": ("prose", "tense")}  # feature -> (02 dataset, 02 feature)


def _sec(t0: float) -> float:
    return round(time.time() - t0, 1)


@app.function(image=build_image, volumes=VOLUMES, timeout=3600, memory=32768, cpu=8)
def build_calib() -> dict:
    from transformers import AutoTokenizer

    import calib_items as CI

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    col_items = json.loads(Path(f"{OUT}/items.json").read_text())
    items = {"lang": CI.build_lang(tok), "tense": CI.build_tense(tok, OUT02, OUT07, col_items)}
    Path(f"{OUT}/calib_items.json").write_text(json.dumps(items))
    VOLUMES["/results"].commit()
    return items


def _bank(feature: str, dirs: dict) -> dict:
    import pickle
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root/ext02")
    from features import FEATURES_BY_NAME, rows_for

    ds, fn = BANK[feature]
    d = f"{OUT02}/acts/{ds}"
    with open(f"{OUT02}/data/{ds}.pkl", "rb") as f:
        items = pickle.load(f)
    index = dict(np.load(f"{d}/index.npz"))
    R = rows_for(FEATURES_BY_NAME[fn], items, index, np.random.default_rng(0))
    rows = R["rows"]
    out = {"n_rows": int(len(rows))}
    for L, dd in dirs.items():
        X = torch.from_numpy(np.load(f"{d}/hs{L + 1}.npy", mmap_mode="r")[rows].astype(np.float32)).cuda()
        out[L] = {k: (X @ v.cuda().float()).cpu() for k, v in dd.items()}
        out[(L, "norm_median")] = float(X.norm(dim=1).median())
        del X
    return out


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=2 * 3600, memory=131072)
def prep_calib() -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calib_lib import build, jlens_conditions
    from calib_templates import JLENS_WORDS
    from mats import capture as C
    from mats import jspace as JS
    from mats.readers.jlens import load_jlens, token_variants

    t0 = time.time()
    VOLUMES["/results"].reload()
    items = json.loads(Path(f"{OUT}/calib_items.json").read_text())
    colspec = torch.load(f"{OUT}/spec.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    ranks = colspec["ranks"]
    bases = {L: JS.load_basis_file(f"/results/03_pipeline_smoke/jspace_basis_L{L}.pt") for L in READ}
    nulls = {L: {k: v for k, v in colspec["nulls"][L].items() if k.startswith(("iso_", "nonj4_"))} for L in READ}
    spec, checks = {}, {}
    for F in FEATS:
        sp = {"h": {}, "dirs": {}, "doses": {}, "nulls": nulls, "probe": {}}
        if F == "col":
            sp["items"] = colspec["items"]
            sp["y"] = np.array([it["col"] for it in colspec["items"]], float)
            for L in READ:
                sp["h"][L] = colspec["h"][L]
                sp["dirs"][L] = {"raw": colspec["dirs"][L]["raw"], "jrem4": colspec["dirs"][L]["jrem4"]}
                sp["doses"][L] = {k: colspec["doses"][L][k2] for k, k2 in (("raw", "raw"), ("jrem4", "jrem4"))}
                sp["probe"][L] = colspec["probe"][L]
            spec[F] = sp
            continue
        its = items[F]
        sp["items"], sp["y"] = its, np.array([it["y"] for it in its])
        acts = C.capture(model, [it["ids"] for it in its], list(READ), [[-1]] * len(its), batch_size=16)
        for L in READ:
            P = torch.load(f"{OUT02}/probes/{F}_hs{L + 1}.pt", weights_only=False)
            raw = P["steer_dir_unit"].float()
            j4 = (P["steer_dir_unit_jrem_r1536"].float() if L == 42 and "steer_dir_unit_jrem_r1536" in P
                  else JS.j_removed_direction(bases[L], raw, ranks[L][1]))
            j4 = j4 / j4.norm()
            sp["dirs"][L] = {"raw": raw, "jrem4": j4}
            sp["h"][L] = acts[L][:, 0].float()
            cls = P["classes"]
            a, b = (cls.index("Russian"), cls.index("Polish")) if F == "lang" else (cls.index("Past"), cls.index("Pres"))
            W = P["probe_w_raw"].float()
            bb = P["probe_b_raw"].float().reshape(-1)  # stored as [1, K]
            sp["probe"][L] = {"w": W[:, a] - W[:, b], "b": float(bb[a] - bb[b])}
            checks[f"{F}_L{L}_enrichment"] = {k: {f"r{r}": float(JS.enrichment(bases[L], v, r)) for r in ranks[L]}
                                              for k, v in sp["dirs"][L].items()}
            checks[f"{F}_L{L}_cos_raw_jrem4"] = float(raw @ j4)
        bank = _bank(F, sp["dirs"])
        for L in READ:
            sp["doses"][L] = {}
            for k in ("raw", "jrem4"):
                dq = C.doses_from_projection(bank[L][k], quantiles=(0.9, 0.99), extra_multiple=None)
                sp["doses"][L][k] = {"q90": dq[0], "q99": dq[1]}
            checks[f"{F}_L{L}_dose_over_median_norm"] = {
                k: {q: v / bank[(L, "norm_median")] for q, v in dd.items()} for k, dd in sp["doses"][L].items()}
            checks[f"{F}_L{L}_pilot_norm_median"] = float(sp["h"][L].norm(dim=1).median())
            checks[f"{F}_L{L}_bank_norm_median"] = bank[(L, "norm_median")]
            # natural probe AUROC on the calibration items (sanity: items carry the feature)
            from mats.metrics import auroc_labels
            sc = (sp["h"][L] @ sp["probe"][L]["w"]).numpy()
            checks[f"{F}_L{L}_probe_natural_auroc"] = auroc_labels(sc, sp["y"])
            sc2 = (sp["h"][L] @ sp["dirs"][L]["raw"]).numpy()
            checks[f"{F}_L{L}_rawdir_natural_auroc"] = auroc_labels(sc2, sp["y"])
        checks[f"{F}_bank_rows"] = bank["n_rows"]
        spec[F] = sp
    for F in FEATS:  # rel-norm dose in bank-percentile terms (for the write-up)
        for L in (42,):
            nm = float(spec[F]["h"][L].norm(dim=1).median())
            checks[f"{F}_L{L}_q90_over_item_norm"] = {k: spec[F]["doses"][L][k]["q90"] / nm for k in ("raw", "jrem4")}
    timing = {"prep": _sec(t0)}

    jl = load_jlens(model, tok, layers=list(READ))
    jout = {}
    for F in FEATS:
        temps = [(fam, a, b) for fam, lst in JLENS_WORDS[F].items() for a, b in lst]
        jids = [(token_variants(tok, a), token_variants(tok, b)) for _, a, b in temps]
        conds = jlens_conditions(F)
        jout[F] = {"templates": temps, "conds": conds, "lo": {}, "probe": {}, "natural_lo": {}}
        for L in READ:
            S = torch.cat([build(spec[F], L, c) for c in conds])
            lo = torch.stack([jl.paired_logodds(S, L, ia, ib) for ia, ib in jids])
            jout[F]["lo"][L] = lo.reshape(len(temps), len(conds), -1)
            jout[F]["probe"][L] = (S @ spec[F]["probe"][L]["w"] + spec[F]["probe"][L]["b"]).reshape(len(conds), -1)
    timing["jlens"] = _sec(t0)
    torch.save(spec, f"{OUT}/calib_spec.pt")
    torch.save(jout, f"{OUT}/calib_jlens.pt")
    VOLUMES["/results"].commit()
    return {"timing": timing, "checks": checks,
            "doses": {F: {str(L): spec[F]["doses"][L] for L in READ} for F in FEATS}}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3600, memory=131072)
def jlens_content() -> dict:
    """Content leak sets for the in-J features (added after the label-token leak check; the label words
    'Russian'/'past' are not what the J-space carries for these features). Unpaired max cosine score
    over token classes, every calibration condition, 40/42/44:
      lang : Cyrillic-containing tokens (+) vs tokens containing Polish diacritics (-)
      tense: past auxiliaries/verbs (+) vs present (-) (fixed lists below)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calib_lib import build, jlens_conditions
    from mats.readers.jlens import load_jlens, token_variants

    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/calib_spec.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda")
    jl = load_jlens(model, tok, layers=list(READ))
    voc = [tok.decode([i]) for i in range(len(tok))]
    cyr = [i for i, t in enumerate(voc) if any("\u0400" <= c <= "\u04ff" for c in t)]
    pol = [i for i, t in enumerate(voc) if any(c in "ąęłńśźżĄĘŁŃŚŹŻ" for c in t)]
    past = token_variants(tok, ["was", "were", "had", "did", "said", "became", "went", "made", "took", "came"])
    pres = token_variants(tok, ["is", "are", "has", "does", "says", "becomes", "goes", "makes", "takes", "comes"])
    sets = {"lang": (cyr, pol), "tense": (past, pres)}
    out = {"sizes": {F: (len(a), len(b)) for F, (a, b) in sets.items()}, "lo": {}}
    for F, (A_, B_) in sets.items():
        conds = jlens_conditions(F)
        ids = A_ + B_
        for L in READ:
            S = torch.cat([build(spec[F], L, c) for c in conds])
            res = []
            for s0 in range(0, S.shape[0], 4096):
                sc = jl.scores(S[s0:s0 + 4096], L, ids=ids)
                res.append(sc[:, :len(A_)].max(1).values - sc[:, len(A_):].max(1).values)
            out["lo"][(F, L)] = torch.cat(res).reshape(len(conds), -1)
    torch.save(out, f"{OUT}/calib_jlens_content.pt")
    VOLUMES["/results"].commit()
    return out["sizes"]


def _jobs(reader: str) -> list[tuple]:
    from calib_lib import conditions

    layers = (42, 40, 44) if reader == "ol" else (42,)
    return [(F, L, i) for F in FEATS for L in layers for i in range(len(conditions(F, L)))]


def _score(reader: str, chunk: int, n_chunks: int) -> dict:
    import torch

    from calib_lib import build, conditions
    from calib_templates import NLA_PAIRS, OL_PAIRS, flat

    t0 = time.time()
    VOLUMES["/results"].reload()
    spec = torch.load(f"{OUT}/calib_spec.pt", weights_only=False)
    jobs = _jobs(reader)[chunk::n_chunks]
    if reader == "ol":
        from mats.readers.olens import load_olens

        ol = load_olens()
        PAIRS = OL_PAIRS
        score = lambda A, L, c: ol.logprob_of_strings(A, L, c, batch=100)  # noqa: E731
    else:
        from mats.readers import nla as NL

        nla = NL.load_nla(load_ar=False)
        PAIRS = NLA_PAIRS
        score = lambda A, L, c: NL.logprob_of_strings_batch(nla, A, c, batch=100)  # noqa: E731
    t_load = _sec(t0)
    out = {"lo": {}, "templates": {F: flat(PAIRS[F]) for F in FEATS}}
    n_acts = 0
    for F in FEATS:
        fp = flat(PAIRS[F])
        cands = [s for _, a, b in fp for s in (a, b)]
        for L in READ:
            idx = [i for FF, LL, i in jobs if FF == F and LL == L]
            if not idx:
                continue
            conds = conditions(F, L)
            A = torch.cat([build(spec[F], L, conds[i]) for i in idx])
            lp = score(A, L, cands)
            lo = torch.stack([lp[:, 2 * t] - lp[:, 2 * t + 1] for t in range(len(fp))])
            out["lo"][(F, L)] = {"idx": idx, "lo": lo.reshape(len(fp), len(idx), -1)}
            n_acts += A.shape[0]
    out |= {"t_load": t_load, "total_s": _sec(t0), "n_acts": n_acts, "gpu": torch.cuda.get_device_name(0)}
    torch.save(out, f"{OUT}/calib_{reader}_chunk{chunk}.pt")
    VOLUMES["/results"].commit()
    return {k: v for k, v in out.items() if k not in ("lo", "templates")} | {"chunk": chunk, "reader": reader}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def score_ol(chunk: int, n_chunks: int) -> dict:
    return _score("ol", chunk, n_chunks)


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600, memory=131072)
def score_nla(chunk: int, n_chunks: int) -> dict:
    return _score("nla", chunk, n_chunks)


@app.local_entrypoint()
def main(skip_build: bool = False, skip_prep: bool = False, ol_chunks: int = 7, nla_chunks: int = 10) -> None:
    import subprocess

    here = Path(__file__).resolve().parent
    summary = {}
    if not skip_build:
        items = build_calib.remote()
        split = json.loads((here / "pilot_split.json").read_text())
        split["calibration_items"] = {
            F: [{k: it[k] for k in ("id", "label", "pos", "source", "source_sha1_16", "source_head")}
                for it in lst] for F, lst in items.items()}
        split["calibration_note"] = ("Matched-dose in-J calibration (calib_items.py, CALIB_SEED 9009): "
                                     "these lang (papluca test) and tense (wikitext) sources are also "
                                     "EXCLUDED from the main analysis.")
        (here / "pilot_split.json").write_text(json.dumps(split, indent=1))
        print({F: len(v) for F, v in items.items()})
    if not skip_prep:
        summary["prep"] = prep_calib.remote()
        print(json.dumps(summary["prep"], indent=1, default=str)[:8000])
    calls = [score_ol.spawn(i, ol_chunks) for i in range(ol_chunks)]
    calls += [score_nla.spawn(i, nla_chunks) for i in range(nla_chunks)]
    summary["jobs"] = []
    for c in calls:
        try:
            summary["jobs"].append(c.get())
        except Exception as e:  # noqa: BLE001
            summary["jobs"].append({"error": repr(e)[:2000]})
    print(json.dumps(summary["jobs"], indent=1, default=str))
    subprocess.run(["modal", "volume", "get", "--force", "rfm-results", "08_col_pilot", str(here / "results")],
                   check=False)
    (here / "results" / "calib_run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
