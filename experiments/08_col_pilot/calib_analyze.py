"""Analysis of the matched-dose in-J calibration (K3). PILOT -- EXCLUDED FROM MAIN ANALYSIS.

    ../../.venv/bin/python calib_analyze.py  -> results/calib_metrics.json, results/calib_stdout.txt
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from calib_lib import K_NULL, ZERO, conditions, dose_keys  # noqa: E402
from calib_templates import PRIMARY  # noqa: E402
from mats import metrics as M  # noqa: E402

RES = HERE / "results"
DATA = RES / "08_col_pilot"
FEATS = ("lang", "tense", "col")
NB = 1000


def load(reader: str) -> dict:
    out: dict = {}
    for f in sorted(glob.glob(str(DATA / f"calib_{reader}_chunk*.pt"))):
        d = torch.load(f, weights_only=False)
        for (F, L), blk in d["lo"].items():
            if (F, L) not in out:
                T = blk["lo"].shape[0]
                out[(F, L)] = [np.full((T, len(conditions(F, L)), blk["lo"].shape[2]), np.nan), d["templates"][F]]
            out[(F, L)][0][:, blk["idx"], :] = blk["lo"].numpy()
    return out


def scores(lo: np.ndarray, temps: list) -> dict[str, np.ndarray]:
    out = {}
    for fam in dict.fromkeys(t[0] for t in temps):
        out[fam] = lo[[i for i, t in enumerate(temps) if t[0] == fam]].mean(0)
    for i in range(len(temps)):
        out[f"t{i}"] = lo[i]
    return out


def cell(S: np.ndarray, conds: list, d: str, q: str) -> dict:
    ci = {c: i for i, c in enumerate(conds)}
    kp, km = ("real", d, q, 1, d), ("real", d, q, -1, d)
    if kp not in ci:
        return {"missing": True}
    p, m = S[ci[kp]], S[ci[km]]
    if np.isnan(p).any():
        return {"missing": True}
    r = M.follow_feature(p, m, n_boot=NB)
    z = S[ci[ZERO]]
    out = {"auroc": r["auroc"], "lo": r["lo"], "hi": r["hi"], "paired_win": r["paired_win"],
           "mean_shift": r["mean_shift"], "shift_sd": float(np.std(p - m, ddof=1)),
           "auroc_zero_sub": M.auroc(p - z, m - z), "fraction_of_probe": 2 * (r["auroc"] - 0.5), "bands": {}}
    for cls in ("iso", "nonj4"):
        ks = [(("null", f"{cls}_{k}", q, 1, d), ("null", f"{cls}_{k}", q, -1, d)) for k in range(K_NULL)]
        if all(a in ci and b in ci for a, b in ks):
            nulls = [(S[ci[a]], S[ci[b]]) for a, b in ks]
            if not any(np.isnan(x).any() or np.isnan(y).any() for x, y in nulls):
                b = M.null_band_contrast((p, m), nulls)
                sh = np.array([x - y for x, y in nulls])
                b["null_shift_sd_between_dirs"] = float(sh.mean(1).std(ddof=1))
                b["null_item_se"] = float(np.sqrt((sh.var(1, ddof=1) / sh.shape[1]).mean()))
                b["null_auroc_sd"] = float(np.std([M.auroc(x, y) for x, y in nulls], ddof=1))
                b["shift_over_item_sd"] = float(np.mean(p - m) / np.std(p - m, ddof=1))
                out["bands"][cls] = b
    return out


def main() -> None:
    spec = torch.load(DATA / "calib_spec.pt", weights_only=False)
    jl = torch.load(DATA / "calib_jlens.pt", weights_only=False)
    dec = {"OL": load("ol"), "NLA": load("nla")}
    out: dict = {"cells": {}, "doses": {F: {str(L): spec[F]["doses"][L] for L in (40, 42, 44)} for F in FEATS}}
    for F in FEATS:
        tables = {}
        for rname, by in dec.items():
            for L in (40, 42, 44):
                if (F, L) in by:
                    lo, temps = by[(F, L)]
                    tables[f"{rname}@{L}"] = (scores(lo, temps), conditions(F, L))
        cpath = DATA / "calib_jlens_content.pt"
        content = torch.load(cpath, weights_only=False) if cpath.exists() else None
        for L in (40, 42, 44):
            temps = jl[F]["templates"]
            S = scores(jl[F]["lo"][L].numpy(), temps)
            if content is not None and (F, L) in content["lo"]:
                S["content"] = content["lo"][(F, L)].numpy()
            tables[f"JLens@{L}"] = (S, jl[F]["conds"])
            tables[f"probe@{L}"] = ({"ridge": jl[F]["probe"][L].numpy()}, jl[F]["conds"])
        for name, (S, conds) in tables.items():
            for key, X in S.items():
                if key.startswith("t") and not name.startswith(("NLA", "OL")):
                    continue
                for d in ("raw", "jrem4"):
                    for q in dose_keys(F, d):
                        out["cells"][f"{F}|{name}|{key}|{d}|{q}"] = cell(X, conds, d, q)
    (RES / "calib_metrics.json").write_text(json.dumps(out, indent=1, default=float))
    report(out)
    print("\n## leak test, 4r* direction (J-Lens, two-sided vs J-complement band): label / content\n")
    for F in ("lang", "tense"):
        for L in (40, 42, 44):
            for q in ("q90", "q99"):
                row = []
                for key in ("label", "content"):
                    c = out["cells"].get(f"{F}|JLens@{L}|{key}|jrem4|{q}")
                    if c and not c.get("missing"):
                        b = c["bands"]["nonj4"]
                        out_ = b["mean_shift"]["outside_two_sided"] or b["auroc"]["outside_two_sided"]
                        row.append(f"{key}: AUROC {c['auroc']:.2f} z_shift {b['mean_shift']['z']:+.1f} "
                                   f"z_AUROC {b['auroc']['z']:+.1f} {'LEAK' if out_ else 'clean'}")
                print(f"{F} L{L} {q}: " + " | ".join(row))
    print("\n## J-Lens content score on the RAW direction (does the in-J feature move J-Lens content?)\n")
    for F in ("lang", "tense"):
        c = out["cells"].get(f"{F}|JLens@42|content|raw|q90")
        if c and not c.get("missing"):
            print(F, f"AUROC {c['auroc']:.2f} win {c['paired_win']:.2f} z_iso {c['bands']['iso']['mean_shift']['z']:+.1f}")


def _z(c: dict, cls: str, k: str) -> str:
    b = c.get("bands", {}).get(cls)
    return f"{b[k]['z']:+.1f}{'*' if b[k]['outside_two_sided'] else ''}" if b else "–"


def report(o: dict) -> None:
    print("# matched-dose in-J calibration (K3) -- PILOT, EXCLUDED FROM MAIN ANALYSIS\n")
    print("doses:", json.dumps(o["doses"], default=float)[:2000])
    for F in FEATS:
        fam = PRIMARY[F]
        print(f"\n## {F} (family {fam}); centred; AUROC [CI] / win / shift±SD / z shift iso,Jc / z AUROC iso,Jc / frac\n")
        print("| reader | dir | dose | AUROC [CI] | win | shift±SD | z shift iso / Jc | z AUROC iso / Jc | frac |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k, c in o["cells"].items():
            f, name, key, d, q = k.split("|")
            if f != F or key not in (fam, "ridge") or c.get("missing"):
                continue
            print(f"| {name} | {d} | {q} | {c['auroc']:.2f} [{c['lo']:.2f},{c['hi']:.2f}] | {c['paired_win']:.2f} | "
                  f"{c['mean_shift']:+.3f}±{c['shift_sd']:.3f} | {_z(c, 'iso', 'mean_shift')} / {_z(c, 'nonj4', 'mean_shift')} | "
                  f"{_z(c, 'iso', 'auroc')} / {_z(c, 'nonj4', 'auroc')} | {c['fraction_of_probe']:+.2f} |")
    print("\n## null-band width (L42, J-complement class): null shift SD between directions / item SE / null AUROC SD; real shift/item-SD\n")
    for k, c in o["cells"].items():
        f, name, key, d, q = k.split("|")
        if name in ("NLA@42", "OL@42") and key == PRIMARY[f] and not c.get("missing") and "nonj4" in c["bands"]:
            b = c["bands"]["nonj4"]
            print(f"{f:5s} {name} {d:5s} {q}: dirSD {b['null_shift_sd_between_dirs']:.3f}  itemSE {b['null_item_se']:.3f}  "
                  f"AUROC-SD {b['null_auroc_sd']:.3f}  real shift {c['mean_shift']:+.3f} (d_z {b['shift_over_item_sd']:+.2f})")
    print("\n## per-template AUROC (centred, raw q90 / jrem4 q90)\n")
    for k, c in o["cells"].items():
        f, name, key, d, q = k.split("|")
        if key.startswith("t") and d == "raw" and q == "q90" and not c.get("missing"):
            c2 = o["cells"].get(f"{f}|{name}|{key}|jrem4|q90", {})
            print(f"{f} {name} {key}: {c['auroc']:.2f} / {c2.get('auroc', float('nan')):.2f}")


if __name__ == "__main__":
    main()
