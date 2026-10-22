"""Phase B local analysis: the layer x position map, the Oracle Lens trained-grid sweep with null bands,
the J-Lens leak check per layer, the NLA position check, the punctuation mass per item, and the
side-by-side text from the Modal job (results/phaseB/*.pt, downloaded by watch.py).

    ../../.venv/bin/python a11_phaseB_analyze.py   -> results/phaseB.json + results/phaseB.md (+ sidebyside_B.md)

PILOT / EXPLORATORY.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import a11_lib as L  # noqa: E402
import a11_modal_lib as ML  # noqa: E402
from mats import metrics as M  # noqa: E402

PB = HERE / "results" / "phaseB"
OUT = HERE / "results"


def _pm(blk: dict, name: str) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """(plus, minus, nulls) for a real direction from a {names, lo [T,C,N] or scores [C,N]} block (family mean)."""
    names = blk["names"]
    X = blk["lo"].numpy().mean(0) if "lo" in blk else blk["scores"].numpy()
    ci = {n: i for i, n in enumerate(names)}
    p, m = X[ci[f"{name}+"]], X[ci[f"{name}-"]]
    nulls = []
    for fam in ("iso", "jc"):
        k = 0
        while f"{fam}{k}+" in ci:
            nulls.append((X[ci[f"{fam}{k}+"]], X[ci[f"{fam}{k}-"]]))
            k += 1
    return p, m, nulls


def _fam_block(blk: dict, fam: str) -> dict:
    ix = [i for i, t in enumerate(blk["templates"]) if t[0] == fam]
    return {"names": blk["names"], "lo": blk["lo"][ix]}


def cell_stats(blk: dict, name: str) -> dict[str, Any]:
    p, m, nulls = _pm(blk, name)
    st = M.paired_stats(p, m) | {"n": int(len(p))}
    if nulls:
        b = M.null_band_contrast((p, m), nulls)
        st |= {"z_shift": b["mean_shift"]["z"], "z_auroc": b["auroc"]["z"], "k_null": len(nulls),
               "out": bool(b["mean_shift"]["outside_two_sided"])}
    return st


def main() -> None:
    A = torch.load(PB / "acts.pt", weights_only=False)
    MR = torch.load(PB / "map_raw.pt", weights_only=False)
    OLS = torch.load(PB / "ol_sweep.pt", weights_only=False) if (PB / "ol_sweep.pt").exists() else {}
    NLS = torch.load(PB / "nla_sweep.pt", weights_only=False) if (PB / "nla_sweep.pt").exists() else {}
    res: dict[str, Any] = {"note": "pilot items (60 col, 120 tense); directions = class-mean per (layer, position); "
                                   "dose q90 of the items' own projection; K nulls per band at the J-removed dose"}
    # ---- map cells: probe (natural, CV) + J-Lens natural + J-Lens leak + OL + NLA
    cells = []
    for (feat, Lr, pos), c in MR["cells"].items():
        row = {"feature": feat, "layer": Lr, "pos": pos, **c, "basis": MR["basis_source"][Lr]}
        js = MR["jl_scores"][(feat, Lr, pos)]
        jl = cell_stats({"names": js["names"], "scores": js["scores"]}, "jrem")
        row["jlens_leak_z"] = jl.get("z_shift")
        row["jlens_leak_out"] = jl.get("out")
        row["jlens_raw_shift"] = cell_stats({"names": js["names"], "scores": js["scores"]}, "raw")["mean_shift"]
        if (feat, Lr, pos) in OLS:
            blk = OLS[(feat, Lr, pos)]
            fam = "nl_cont" if feat == "col" else "f07"
            fb = _fam_block(blk, fam)
            s = cell_stats(fb, "jrem")
            row.update({"OL_fam": fam, "OL_jrem_auroc": s["auroc"], "OL_jrem_shift": s["mean_shift"], "OL_jrem_z": s.get("z_shift"),
                        "OL_raw_auroc": cell_stats(fb, "raw")["auroc"], "OL_raw_shift": cell_stats(fb, "raw")["mean_shift"]})
            if feat == "col":
                fx = _fam_block(blk, "fixed")
                row["OL_fixed_jrem_auroc"] = cell_stats(fx, "jrem")["auroc"]
        if (feat, Lr, pos) in NLS and "jrem+" in NLS[(feat, Lr, pos)]["names"]:
            blk = NLS[(feat, Lr, pos)]
            fam = "m2" if feat == "col" else "f07"
            fb = _fam_block(blk, fam)
            s = cell_stats(fb, "jrem")
            row.update({"NLA_fam": fam, "NLA_jrem_auroc": s["auroc"], "NLA_jrem_shift": s["mean_shift"], "NLA_jrem_z": s.get("z_shift"),
                        "NLA_raw_auroc": cell_stats(fb, "raw")["auroc"]})
        cells.append(row)
    res["cells"] = cells
    # consistency: rank correlation of natural probe AUROC with each reader's z across cells
    for key in ("jlens_natural_auroc", "OL_jrem_z", "OL_raw_auroc", "NLA_jrem_z"):
        xs = [c["probe_auroc"] for c in cells if key in c and c[key] is not None]
        ys = [c[key] for c in cells if key in c and c[key] is not None]
        res[f"rank_corr_probe_vs_{key}"] = L.rank_corr(xs, ys)
    # ---- NLA position check: zero-dose template LO at -1..-4 vs label (does the NLA read the same thing one token earlier?)
    nla_pos = {}
    for (feat, Lr, pos), blk in NLS.items():
        zi = blk["names"].index("zero")
        fam = "m2" if feat == "col" else "f07"
        fb = _fam_block(blk, fam)
        S = fb["lo"].numpy().mean(0)[zi]
        y = ML.median_split(np.array([it["col"] for it in json.loads((HERE / "data" / "items.json").read_text())["col"]])) if feat == "col" \
            else np.array([it["y"] for it in json.loads((HERE / "data" / "items.json").read_text())["tense"]])
        nla_pos[f"{feat}|pos{pos}"] = {"natural_auroc_vs_label": M.auroc(S[y == 1], S[y == 0])}
    res["nla_natural_by_position"] = nla_pos
    # ---- punctuation mass (A3 follow-up): next-token punctuation / newline mass per item, by stratum
    items = json.loads((HERE / "data" / "items.json").read_text())
    pm = {}
    for feat in ("col", "tense"):
        mass = A["mass"][feat]
        ctx = [L.punct_context(it["text"]) for it in items[feat]]
        strata = np.array([c["ends"] for c in ctx])
        pm[feat] = {"p_punct_mean": float(np.mean(mass["p_punct"])), "p_newline_mean": float(np.mean(mass["p_newline"])),
                    "by_ends": {s: {"n": int((strata == s).sum()), "p_punct": float(np.mean(np.array(mass["p_punct"])[strata == s]))}
                                for s in sorted(set(strata.tolist()))},
                    "top5_examples": mass["top5"][:5]}
        if feat == "tense":
            hi = np.array(mass["p_punct"]) > 0.5
            pm[feat]["n_items_next_token_mostly_punct"] = int(hi.sum())
            # OL tense shift at 42 / -1 split by whether the model's next token is mostly punctuation
            blk = OLS.get(("tense", 42, -1))
            if blk is not None:
                p, m, _ = _pm(_fam_block(blk, "f07"), "jrem")
                pm[feat]["OL_f07_jrem_shift_by_next_punct"] = L.strata_stats(p, m, np.where(hi, "punct_next", "word_next").tolist())
    res["punctuation_mass"] = pm
    # ---- OL P2 channel with the BANK directions (exp 08 spec: 02 regression direction, bank q90, 08 nulls) at 40/42/44
    bank = {}
    if (PB / "ol_bank.pt").exists():
        OB = torch.load(PB / "ol_bank.pt", weights_only=False)
        for (feat, Lr, pos), blk in OB.items():
            for fam in ("nl_cont", "fixed"):
                fb = _fam_block(blk, fam)
                bank[f"OL@{Lr}|{fam}"] = {"jrem4": cell_stats(fb, "jrem4"), "raw": cell_stats(fb, "raw")}
    res["ol_bank_40_42_44"] = bank
    # ---- direction agreement: sweep (item class-mean) vs bank directions at 40/42/44
    cos = {}
    try:
        sp = torch.load(HERE.parents[0] / "08_col_pilot" / "results" / "08_col_pilot" / "spec.pt", weights_only=False)
        for Lr in (40, 42, 44):
            v = MR["dirs"][("col", Lr, -1)]
            cos[f"col@{Lr}"] = {"raw": float(v["raw"] @ sp["dirs"][Lr]["raw"]), "jrem": float(v["jrem"] @ sp["dirs"][Lr]["jrem4"])}
    except FileNotFoundError:
        pass
    res["sweep_vs_bank_direction_cosine"] = cos
    (OUT / "phaseB.json").write_text(json.dumps(res, indent=1, default=float))
    # ---- markdown
    lines = ["# Phase B — layer x position map, OL trained grid, NLA position check (PILOT / EXPLORATORY)\n",
             "| feature | layer | pos | basis | probe AUROC (CV) | probe J-removed | enrich 384/1536 | J-Lens natural | J-Lens leak z | OL jrem AUROC (z) | OL raw AUROC | NLA jrem AUROC (z) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["feature"], c["layer"], -c["pos"])):
        f = lambda k, nd=2: ("-" if c.get(k) is None else f"{c[k]:.{nd}f}")  # noqa: E731
        olz = f"{f('OL_jrem_auroc')} ({f('OL_jrem_z', 1)})" if "OL_jrem_auroc" in c else "-"
        nlz = f"{f('NLA_jrem_auroc')} ({f('NLA_jrem_z', 1)})" if "NLA_jrem_auroc" in c else "-"
        lines.append(f"| {c['feature']} | {c['layer']} | {c['pos']} | {c['basis']} | {f('probe_auroc')} | {f('probe_auroc_jremoved')} | "
                     f"{f('enrichment_raw_384')}/{f('enrichment_raw')} | {f('jlens_natural_auroc')} | {f('jlens_leak_z', 1)} | {olz} | {f('OL_raw_auroc')} | {nlz} |")
    lines += ["", "rank correlations (probe AUROC vs reader): " + json.dumps({k: v for k, v in res.items() if k.startswith("rank_corr")}),
              "", "NLA natural AUROC by position: " + json.dumps(nla_pos), "",
              "punctuation mass: " + json.dumps(pm, default=float)[:2500]]
    (OUT / "phaseB.md").write_text("\n".join(lines))
    print("\n".join(lines)[:6000])


if __name__ == "__main__":
    main()
