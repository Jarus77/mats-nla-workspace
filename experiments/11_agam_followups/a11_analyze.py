"""Phase A of experiment 11 (mentor follow-ups): analyses on EXISTING data, no GPU.

PILOT / EXPLORATORY. Everything here re-reads the declared pilot split of exp 08/09 (excluded from the
main analysis) and the WorkspaceBench readouts of exp 01/06.

    ../../.venv/bin/python a11_analyze.py      -> results/phaseA.json + results/phaseA.md

Sections (Agam's points):
  A1  Oracle Lens at layers 40/42/44 with null bands (point 1)
  A2  consensus over k samples on the existing free readouts (point 2)
  A3  punctuation context at the read token, per-stratum shifts (point 3)
  A5  probe accuracy vs reader detection at 40/42/44 (point 5)
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parents[0] / "src"))
sys.path.insert(0, str(EXP / "08_col_pilot"))
sys.path.insert(0, str(EXP / "09_ol_mechanism"))

import a11_lib as L  # noqa: E402
import o9_lib  # noqa: E402
import pilot_lib  # noqa: E402
import r5_lib  # noqa: E402
from mats import metrics as M  # noqa: E402
from templates import MENTION_RE, MENTION_RE_STRICT  # noqa: E402

D08 = EXP / "08_col_pilot" / "results" / "08_col_pilot"
D09 = EXP / "09_ol_mechanism" / "results" / "09_ol_mechanism"
D07 = EXP / "07_template_mining" / "results"
D06 = EXP / "06_resolving_decomposition" / "data"
D01 = EXP / "01_wsbench_replication"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

RX_STRICT = re.compile(MENTION_RE_STRICT, re.I)
RX_LOOSE = re.compile(MENTION_RE, re.I)
K_NULL = pilot_lib.K_NULL


# ----------------------------------------------------------------------------- loaders
def load_pilot_decoder(name: str) -> dict[int, tuple[np.ndarray, list]]:
    """{layer: (lo [T, C, N], templates)} from exp 08 pilot chunks (conditions per pilot_lib)."""
    out: dict[int, list] = {}
    for f in sorted(glob.glob(str(D08 / f"{name}_chunk*.pt"))):
        d = torch.load(f, weights_only=False)
        for layer, blk in d["lo"].items():
            C = len(pilot_lib.conditions(layer))
            T, _, N = blk["lo"].shape
            if layer not in out:
                out[layer] = [np.full((T, C, N), np.nan), d["templates"]]
            out[layer][0][:, blk["idx"], :] = blk["lo"].numpy()
    return {k: (v[0], v[1]) for k, v in out.items()}


def load_r5(reader: str) -> dict[str, tuple[np.ndarray, list]]:
    out: dict[str, list] = {}
    for f in sorted(glob.glob(str(D08 / f"r5_{reader}_chunk*.pt"))):
        d = torch.load(f, weights_only=False)
        for S, blk in d["lo"].items():
            if S not in out:
                out[S] = [np.full((blk["lo"].shape[0], len(r5_lib.conditions(S)), blk["lo"].shape[2]), np.nan), d["templates"][S]]
            out[S][0][:, blk["idx"], :] = blk["lo"].numpy()
    return {k: (v[0], v[1]) for k, v in out.items()}


def load_o9(S: str, adapter: str = "on") -> tuple[np.ndarray, list]:
    C = len(o9_lib.conditions(S))
    X, temps = None, None
    for f in sorted(glob.glob(str(D09 / f"ol_{adapter}_{S}_chunk*.pt"))):
        d = torch.load(f, weights_only=False)
        lo = d["lo"].numpy()
        if X is None:
            X, temps = np.full((lo.shape[0], C, lo.shape[2]), np.nan), d["templates"]
        X[:, d["idx"], :] = lo
    return X, temps


def fam_mean(lo: np.ndarray, temps: list, fam: str) -> np.ndarray:
    ix = [i for i, t in enumerate(temps) if t[0] == fam]
    return lo[ix].mean(0)


def band_summary(p: np.ndarray, m: np.ndarray, nulls: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    b = M.null_band_contrast((p, m), nulls)
    return {"z_shift": b["mean_shift"]["z"], "z_auroc": b["auroc"]["z"],
            "out_shift": bool(b["mean_shift"]["outside_two_sided"]), "k": len(nulls)}


# ----------------------------------------------------------------------------- A1: OL layers
def a1_ol_layers() -> dict[str, Any]:
    """OL@40/42/44 on the exp 08 col pilot cells. At 40/44 the only null blocks are the jrem1 sets
    (iso, nonj1) at q90/q99; jrem4 and raw cells there get a PROXY band from those blocks (larger or
    equal norm -> conservative z). At 42 every cell has its matched band."""
    dec = load_pilot_decoder("ol")
    out: dict[str, Any] = {"note": "pilot split, excluded from main analysis; 60 items; K=20 nulls per band",
                           "cells": {}}
    for layer, (lo, temps) in sorted(dec.items()):
        conds = pilot_lib.conditions(layer)
        ci = {c: i for i, c in enumerate(conds)}
        fams = list(dict.fromkeys(t[0] for t in temps))
        for fam in fams:
            S = fam_mean(lo, temps, fam)
            for d in pilot_lib.REAL_DIRS:
                for q in pilot_lib.DOSES:
                    p, m = S[ci[("real", d, "centred", q, 1, d)]], S[ci[("real", d, "centred", q, -1, d)]]
                    if np.isnan(p).any() or np.isnan(m).any():
                        continue
                    cell = M.paired_stats(p, m) | {"n": int(len(p))}
                    refs = [r for r in ("jrem1",) if layer != 42] or ([d] if d != "raw" else ["raw"])
                    for ref in refs:
                        for cls in pilot_lib.NULL_SETS[ref]:
                            keys = [(("null", f"{cls}_{k}", "centred", q, s, ref)) for k in range(K_NULL) for s in (1, -1)]
                            if not all(k in ci for k in keys):
                                continue
                            nulls = [(S[ci[("null", f"{cls}_{k}", "centred", q, 1, ref)]],
                                      S[ci[("null", f"{cls}_{k}", "centred", q, -1, ref)]]) for k in range(K_NULL)]
                            if any(np.isnan(a).any() or np.isnan(b).any() for a, b in nulls):
                                continue
                            tag = cls if ref == d or (d == "raw" and ref == "raw") else f"{cls}~proxy({ref})"
                            cell[f"band:{tag}"] = band_summary(p, m, nulls)
                    out["cells"][f"OL@{layer}|{fam}|{d}|{q}"] = cell
    # calibration (lang / tense raw at r11-equivalent q90) from calib_metrics.json
    cm = json.loads((EXP / "08_col_pilot" / "results" / "calib_metrics.json").read_text())["cells"]
    out["calib"] = {k: {kk: v[kk] for kk in ("auroc", "paired_win", "mean_shift")} for k, v in cm.items()
                    if "|OL@" in k}
    # WorkspaceBench per-layer OL (exp 06 tables; exp 01 grid)
    t06 = json.loads((D06 / "tables.json").read_text())["pass"]
    out["wsbench_pass"] = {k: v for k, v in t06.items() if k.startswith("OL") or k in ("NLA@42", "JLens@42")}
    grid = {}
    for fam_dir in sorted((D01 / "judged" / "olens_grid").glob("*")):
        rj = fam_dir / "results.json"
        if not rj.exists():
            continue
        rows = json.loads(rj.read_text())["rows"]
        per_layer: dict[str, list[float]] = {}
        for r in rows:
            for layer, blk in r["layers"].items():
                hit = any(blk["hits"].get(role) for role in r["roles"]) if blk["hits"] else False
                per_layer.setdefault(layer, []).append(float(hit))
        grid[fam_dir.name] = {layer: float(np.mean(v)) for layer, v in sorted(per_layer.items(), key=lambda kv: int(kv[0]))}
    out["wsbench_ol_grid_pass_by_layer"] = grid
    return out


# ----------------------------------------------------------------------------- A2: consensus
def _cond_key(c: tuple) -> str:
    return "natural" if c[0] == "zero" else f"{c[0]}|{c[1]}|{c[3]}|{'+' if c[4] > 0 else '-'}"


def a2_consensus() -> dict[str, Any]:
    out: dict[str, Any] = {"note": "k=4 samples per item per condition (exp 08); regex = MENTION_RE_STRICT (loose in parallel)",
                           "NLA": {}, "OL": {}, "exp07_self_consistency": {}}
    fres: dict = {}
    for f in sorted(glob.glob(str(D08 / "fve_part*.pt"))):
        fres.update(torch.load(f, weights_only=False)["res"])
    for _, r in sorted(fres.items()):
        rows = r["explanations"]
        out["NLA"][_cond_key(r["cond"])] = {
            "strict": L.consensus_rate(rows, lambda t: bool(RX_STRICT.search(t)), 2),
            "loose": L.consensus_rate(rows, lambda t: bool(RX_LOOSE.search(t)), 2)}
    g = torch.load(D08 / "gen_ol.pt", weights_only=False)
    for c, rows in zip(g["conds"], g["texts"]):
        out["OL"][_cond_key(c)] = {
            "strict": L.consensus_rate(rows, lambda t: bool(RX_STRICT.search(t)), 2),
            "loose": L.consensus_rate(rows, lambda t: bool(RX_LOOSE.search(t)), 2)}
    # exp 07: k=8 unsteered generations; self-consistency of the col-mention property per reader/layer
    for feat in ("nj1_col", "nj1_col_break"):
        for reader in ("nla", "ol"):
            p = D07 / f"gen_{reader}_{feat}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            gens = d.get("nla_gens") or d.get("ol_gens")
            if reader == "nla":
                rows = list(gens.values())
                out["exp07_self_consistency"][f"{feat}|NLA@42"] = {
                    "strict_sc": L.self_consistency(rows, lambda t: bool(RX_STRICT.search(t)))["mean"],
                    "strict": L.consensus_rate(rows, lambda t: bool(RX_STRICT.search(t)), 2)}
            else:
                for layer, by_item in gens.items():
                    rows = list(by_item.values()) if isinstance(by_item, dict) else by_item
                    out["exp07_self_consistency"][f"{feat}|OL@{layer}"] = {
                        "strict_sc": L.self_consistency(rows, lambda t: bool(RX_STRICT.search(t)))["mean"],
                        "strict": L.consensus_rate(rows, lambda t: bool(RX_STRICT.search(t)), 2)}
    for blk in ("NLA", "OL"):
        for k, v in out[blk].items():
            for kind in ("strict", "loose"):
                v[kind] = {kk: vv for kk, vv in v[kind].items() if kk != "support"}
    for k, v in out["exp07_self_consistency"].items():
        v["strict"] = {kk: vv for kk, vv in v["strict"].items() if kk != "support"}
    return out


# ----------------------------------------------------------------------------- A3: punctuation strata
def _strata_for_items(items: list[dict]) -> dict[str, list[str]]:
    ctx = [L.punct_context(it["text"]) for it in items]
    strata = {"ends": [c["ends"] for c in ctx],
              "sentence_start": [str(c["sentence_start"]) for c in ctx]}
    if "chars_remaining" in items[0]:
        rem = np.array([it["chars_remaining"] for it in items])
        strata["chars_remaining"] = ["<=5" if r <= 5 else ("6-20" if r <= 20 else ">20") for r in rem]
        strata["next_is_nl"] = [str(bool(it["next_is_nl"])) for it in items]
    return strata


def a3_punctuation() -> dict[str, Any]:
    out: dict[str, Any] = {"note": "per-stratum paired shift (+ minus -), pilot split; strata from the text ending at the read token"}
    # --- col, round 5 (L42): OL nl_cont (P2 channel), NLA m2 (P1 template), fixed families; direction reg_jrem4, q90
    sp5 = torch.load(D08 / "r5_spec.pt", weights_only=False)
    items = sp5["col"]["items"]
    strata = _strata_for_items(items)
    out["col_strata_counts"] = {k: {s: int(n) for s, n in zip(*np.unique(v, return_counts=True))} for k, v in strata.items()}
    conds = r5_lib.conditions("col")
    ci = {c: i for i, c in enumerate(conds)}
    col_cells: dict[str, Any] = {}
    for reader in ("ol", "nla"):
        lo, temps = load_r5(reader)["col"]
        fams = list(dict.fromkeys(t[0] for t in temps))
        out[f"col_templates_{reader}"] = [list(t) for t in temps]
        for fam in fams:
            S = fam_mean(lo, temps, fam)
            for d in ("reg_jrem4", "reg_raw"):
                p, m = S[ci[("real", d, "q90", 1, d)]], S[ci[("real", d, "q90", -1, d)]]
                if np.isnan(p).any() or np.isnan(m).any():
                    continue
                col_cells[f"{reader.upper()}@42|{fam}|{d}|q90"] = {
                    "all": M.paired_stats(p, m) | {"n": int(len(p))},
                    **{sk: L.strata_stats(p, m, sv) for sk, sv in strata.items()}}
    out["col"] = col_cells
    # --- tense, exp 09 (L42): OL tense_tok / f07_tense vs the model's own tense readout; direction jrem4 and raw, q90
    it9 = json.loads((D09 / "items.json").read_text())["tense"]
    strata_t = _strata_for_items(it9)
    out["tense_strata_counts"] = {k: {s: int(n) for s, n in zip(*np.unique(v, return_counts=True))} for k, v in strata_t.items()}
    X, temps = load_o9("tense", "on")
    conds9 = o9_lib.conditions("tense")
    ci9 = {c: i for i, c in enumerate(conds9)}
    beh = torch.load(D09 / "behave_tense.pt", weights_only=False)
    assert beh["conds"] == conds9
    B = {r: np.stack([v[r] for v in beh["vals"]]) for r in beh["readouts"]}
    tense_cells: dict[str, Any] = {}
    for d in ("jrem4", "raw"):
        for q in ("q90", "r108"):
            ip, im = ci9[("real", d, q, 1, d)], ci9[("real", d, q, -1, d)]
            for fam in ("tense_tok", "f07_tense"):
                S = fam_mean(X, temps, fam)
                if np.isnan(S[ip]).any():
                    continue
                tense_cells[f"OL@42|{fam}|{d}|{q}"] = {"all": M.paired_stats(S[ip], S[im]) | {"n": int(S.shape[1])},
                                                     **{sk: L.strata_stats(S[ip], S[im], sv) for sk, sv in strata_t.items()}}
            for r in ("tense_tok", "tense_broad"):
                tense_cells[f"model|{r}|{d}|{q}"] = {"all": M.paired_stats(B[r][ip], B[r][im]) | {"n": int(B[r].shape[1])},
                                                   **{sk: L.strata_stats(B[r][ip], B[r][im], sv) for sk, sv in strata_t.items()}}
    out["tense"] = tense_cells
    return out


# ----------------------------------------------------------------------------- A5: probe vs readers by layer
def a5_probe_vs_readers() -> dict[str, Any]:
    """Exp 08 col pilot: at L40/42/44 the ridge probe, the J-Lens (EOL leak sets + word templates) and OL
    were scored on the same steered vectors; NLA at 42 only. Per layer and cell: probe AUROC, reader AUROC,
    fraction of probe, z vs the matched band where it exists."""
    jl = torch.load(D08 / "jlens.pt", weights_only=False)
    conds = pilot_lib.JLENS_CONDS
    ci = {c: i for i, c in enumerate(conds)}
    dec = {"OL": load_pilot_decoder("ol"), "NLA": load_pilot_decoder("nla")}
    out: dict[str, Any] = {"note": "exp 08 col pilot; same 60 items and steered vectors per layer; probe = ridge on raw h",
                           "cells": []}
    for layer in (40, 42, 44):
        probe = jl["probe"][layer].numpy()
        eol = jl["eol"][layer]["narrow"].numpy()
        for d in pilot_lib.REAL_DIRS:
            for q in pilot_lib.DOSES:
                ip, im = ci[("real", d, "centred", q, 1, d)], ci[("real", d, "centred", q, -1, d)]
                cell: dict[str, Any] = {"layer": layer, "dir": d, "dose": q,
                                        "probe_auroc": M.auroc(probe[ip], probe[im]),
                                        "jlens_eol_auroc": M.auroc(eol[ip], eol[im])}
                for rname, by_layer in dec.items():
                    if layer not in by_layer:
                        continue
                    lo, temps = by_layer[layer]
                    rc = {c: i for i, c in enumerate(pilot_lib.conditions(layer))}
                    for fam in dict.fromkeys(t[0] for t in temps):
                        S = fam_mean(lo, temps, fam)
                        p, m = S[rc[("real", d, "centred", q, 1, d)]], S[rc[("real", d, "centred", q, -1, d)]]
                        if np.isnan(p).any():
                            continue
                        a = M.auroc(p, m)
                        cell[f"{rname}|{fam}|auroc"] = a
                        cell[f"{rname}|{fam}|frac_of_probe"] = M.fraction_of_probe(a, cell["probe_auroc"]) if cell["probe_auroc"] > 0.5 else float("nan")
                out["cells"].append(cell)
    # calibration features (lang, tense): J-Lens label template + probe at 40/42/44; OL raw at 40/44 from calib_metrics
    cj = torch.load(D08 / "calib_jlens.pt", weights_only=False)
    cm = json.loads((EXP / "08_col_pilot" / "results" / "calib_metrics.json").read_text())["cells"]
    for feat in ("lang", "tense", "col"):
        cc = cj[feat]["conds"]
        cci = {tuple(c): i for i, c in enumerate(cc)}
        for layer in (40, 42, 44):
            probe = cj[feat]["probe"][layer].numpy()
            lo = cj[feat]["lo"][layer].numpy().mean(0)
            for d in ("raw", "jrem4"):
                key_p = next((k for k in cci if k[0] == "real" and k[1] == d and k[2] == "q90" and k[3] == 1), None)
                key_m = next((k for k in cci if k[0] == "real" and k[1] == d and k[2] == "q90" and k[3] == -1), None)
                if key_p is None or key_m is None:
                    continue
                ip, im = cci[key_p], cci[key_m]
                cell = {"feature": feat, "layer": layer, "dir": d, "dose": "q90",
                        "probe_auroc": M.auroc(probe[ip], probe[im]), "jlens_label_auroc": M.auroc(lo[ip], lo[im])}
                k_ol = f"{feat}|OL@{layer}|label|{d}|q90"
                if k_ol in cm:
                    cell["OL|label|auroc"] = cm[k_ol]["auroc"]
                out["cells"].append(cell)
    # natural decodability by layer (probe on UNSTEERED items, the number Agam means by "probe accuracy"):
    # exp 02 G1 (R2 for counts, accuracy for classes) at L40/42/44 and exp 04 G1_by_layer (AUROC) at 38-44.
    g02 = json.loads((EXP / "02_nonj_features" / "gates.json").read_text())
    g04 = json.loads((EXP / "04_backup_features" / "results" / "gates.json").read_text())
    nat: dict[str, dict[str, float]] = {}
    for feat, g in g02.items():
        if feat.startswith("_"):
            continue
        g1 = g["G1"]
        nat[f"02:{feat}"] = {"40": g1["L40"], "42": g1.get("r2", g1.get("acc")), "44": g1["L44"], "metric": "r2" if "r2" in g1 else "acc"}
    for feat, g in g04.items():
        if isinstance(g, dict) and "G1_by_layer" in g:
            nat[f"04:{feat}"] = {**{k: v for k, v in g["G1_by_layer"].items()}, "metric": "auroc"}
    out["natural_decodability_by_layer"] = nat
    out["caveat"] = ("probe_auroc on steered pairs is 1.00 for col by construction (the steering direction is the probe's own "
                     "covariance direction); use natural_decodability_by_layer for 'probe accuracy', and B1 for a per-layer "
                     "cross-validated probe on natural items at the readers' sites.")
    # rank correlation between probe AUROC and each reader's AUROC across cells
    rc: dict[str, Any] = {}
    keys = sorted({k for c in out["cells"] for k in c if k.endswith("|auroc") or k in ("jlens_eol_auroc", "jlens_label_auroc")})
    for k in keys:
        xs = [c["probe_auroc"] for c in out["cells"] if k in c]
        ys = [c[k] for c in out["cells"] if k in c]
        rc[k] = L.rank_corr(xs, ys)
    out["rank_corr_probe_vs_reader"] = rc
    return out


# ----------------------------------------------------------------------------- report
def _f(x: Any, nd: int = 2) -> str:
    try:
        return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def report(res: dict[str, Any]) -> str:
    lines = ["# Phase A (existing data) — PILOT / EXPLORATORY\n"]
    a1 = res["A1"]
    lines += ["## A1 Oracle Lens by layer (exp 08 col pilot, centred, 60 items)\n",
              "| cell | AUROC | win | shift | z_shift (band) |", "|---|---|---|---|---|"]
    for k, c in a1["cells"].items():
        bands = ", ".join(f"{_f(v['z_shift'], 1)} ({kk.split(':', 1)[1]})" for kk, v in c.items() if kk.startswith("band:"))
        lines.append(f"| {k} | {_f(c['auroc'])} | {_f(c['paired_win'])} | {_f(c['mean_shift'])} | {bands} |")
    lines += ["", "WorkspaceBench pass (exp 06): " + json.dumps(a1["wsbench_pass"]), "",
              "OL grid pass by layer (exp 01): " + json.dumps(a1["wsbench_ol_grid_pass_by_layer"]), ""]
    a2 = res["A2"]
    lines += ["## A2 consensus (>=2 of 4 samples) vs single-sample mention rates (strict regex)\n",
              "| reader | condition | single | items any | consensus |", "|---|---|---|---|---|"]
    for rd in ("NLA", "OL"):
        for k, v in a2[rd].items():
            s = v["strict"]
            lines.append(f"| {rd} | {k} | {_f(s['single'], 3)} | {_f(s['items_any'], 3)} | {_f(s['consensus'], 3)} |")
    lines += ["", "exp 07 (k=8, unsteered) strict self-consistency: " + json.dumps(
        {k: {"sc": _f(v["strict_sc"]), "single": _f(v["strict"]["single"], 3), "consensus": _f(v["strict"]["consensus"], 3)}
         for k, v in a2["exp07_self_consistency"].items()}), ""]
    a3 = res["A3"]
    lines += ["## A3 punctuation strata (shift = mean LO(+) − LO(−); n per stratum)\n",
              "col strata counts: " + json.dumps(a3["col_strata_counts"]), "",
              "| cell | all | ends=stop | ends=punct | ends=word | sent_start=True | sent_start=False |",
              "|---|---|---|---|---|---|---|"]
    for k, c in a3["col"].items():
        e, s = c["ends"], c["sentence_start"]
        lines.append(f"| {k} | {_f(c['all']['mean_shift'])} (n{c['all']['n']}) | "
                     + " | ".join(f"{_f(e[x]['mean_shift'])} (n{e[x]['n']})" if x in e else "-" for x in ("stop", "punct", "word"))
                     + " | " + " | ".join(f"{_f(s[x]['mean_shift'])} (n{s[x]['n']})" if x in s else "-" for x in ("True", "False")) + " |")
    lines += ["", "tense strata counts: " + json.dumps(a3["tense_strata_counts"]), "",
              "| cell | all | ends=stop | ends=punct | ends=word | sent_start=True | sent_start=False |",
              "|---|---|---|---|---|---|---|"]
    for k, c in a3["tense"].items():
        e, s = c["ends"], c["sentence_start"]
        lines.append(f"| {k} | {_f(c['all']['mean_shift'])} (n{c['all']['n']}) | "
                     + " | ".join(f"{_f(e[x]['mean_shift'])} (n{e[x]['n']})" if x in e else "-" for x in ("stop", "punct", "word"))
                     + " | " + " | ".join(f"{_f(s[x]['mean_shift'])} (n{s[x]['n']})" if x in s else "-" for x in ("True", "False")) + " |")
    a5 = res["A5"]
    lines += ["", "## A5 probe vs readers by layer (AUROC on the same steered pairs)\n",
              "| feature | layer | dir | dose | probe | J-Lens | OL | NLA |", "|---|---|---|---|---|---|---|---|"]
    for c in a5["cells"]:
        jl = c.get("jlens_eol_auroc", c.get("jlens_label_auroc"))
        ol = next((v for k, v in c.items() if k.startswith("OL|") and k.endswith("|auroc") and ("count_qual" in k or "label" in k)), None)
        nla = next((v for k, v in c.items() if k.startswith("NLA|") and k.endswith("|auroc") and "count_qual" in k), None)
        lines.append(f"| {c.get('feature', 'col')} | {c['layer']} | {c['dir']} | {c['dose']} | {_f(c['probe_auroc'])} | {_f(jl)} | {_f(ol)} | {_f(nla)} |")
    lines += ["", "rank correlation (probe AUROC vs reader AUROC across cells): " + json.dumps(
        {k: {"rho": _f(v["rho"]), "n": v["n"]} for k, v in a5["rank_corr_probe_vs_reader"].items()}), ""]
    return "\n".join(lines)


def main() -> None:
    res = {"A1": a1_ol_layers(), "A2": a2_consensus(), "A3": a3_punctuation(), "A5": a5_probe_vs_readers()}
    (OUT / "phaseA.json").write_text(json.dumps(res, indent=1, default=float))
    md = report(res)
    (OUT / "phaseA.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
