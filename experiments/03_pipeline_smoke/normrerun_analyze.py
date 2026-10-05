"""Local analysis of the norm-preserving re-run -> results/normrerun/metrics.json + tables.

    ../../.venv/bin/python normrerun_analyze.py
"""
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mats import metrics as M  # noqa: E402
from normrerun_lib import REAL_DIRS, VARIANTS, conditions  # noqa: E402

RES = HERE / "results" / "normrerun"
NB = 2000


def load_reader(name: str, n_conds: int) -> np.ndarray | None:
    files = sorted(glob.glob(str(RES / f"{name}_chunk*.pt")))
    if not files:
        return None
    lo = None
    for f in files:
        d = torch.load(f, weights_only=False)
        T, _, N = d["lo"].shape
        if lo is None:
            lo = np.full((T, n_conds, N), np.nan)
        lo[:, d["idx"], :] = d["lo"].numpy()
    return lo.mean(0)  # template mean [C, N]


def main() -> None:
    spec = torch.load(RES / "spec.pt", weights_only=False)
    jl = torch.load(RES / "jlens.pt", weights_only=False)
    conds = [tuple(c) for c in jl["conds"]]
    assert conds == conditions(), "condition list changed since the run"
    ci = {c: i for i, c in enumerate(conds)}
    pair = np.asarray(spec["pair"])
    readers = {"OL@42": load_reader("ol", len(conds)), "NLA@42": load_reader("nla", len(conds))}
    for L in (40, 42, 44):
        readers[f"JLens@{L}"] = jl["lo"][L].numpy()
    readers = {k: v for k, v in readers.items() if v is not None}
    zero = ci[("zero", None, "add", None, 0, None)]

    out = {"stats": {}, "null_band": {}, "identity_add_vs_preserve": {}, "asymmetry": {},
           "doses": {f"L{L}|{k[0]}|f{k[1]}": v for L, d in spec["doses"].items() for k, v in d.items()},
           "inlp_heldout_auroc": {f"L{k[0]}|f{k[1]}": v for k, v in spec["inlp_heldout_auroc"].items()}}
    for rname, lo in readers.items():
        for d in REAL_DIRS:
            for v in VARIANTS:
                for q in ("q90", "q99"):
                    p, m = lo[ci[("real", d, v, q, 1, d)]], lo[ci[("real", d, v, q, -1, d)]]
                    r = M.follow_feature(p, m, clusters=pair, n_boot=NB)
                    rz = M.follow_feature(p, m, clusters=pair, lo_zero=lo[zero], n_boot=NB)
                    out["stats"][f"{rname}|{d}|{v}|{q}"] = {
                        "auroc": r["auroc"], "lo": r["lo"], "hi": r["hi"], "paired_win": r["paired_win"],
                        "mean_shift": r["mean_shift"], "shift_sd": float(np.std(p - m, ddof=1)),
                        "auroc_zero_sub": rz["auroc"]}
                # null bands at q99 (nulls share the ± construction and the q99 dose of the
                # reference direction: raw for raw, jrem for the J-removed directions)
                ref = "raw" if d == "raw" else "jrem"
                nv = "centred" if v == "centred" else "add"
                if nv == "centred" and ref == "raw":
                    continue
                real = (lo[ci[("real", d, v, "q99", 1, d)]], lo[ci[("real", d, v, "q99", -1, d)]])
                for kind in ("nonj", "iso"):
                    nulls = [(lo[ci[("null", f"{kind}{k}", nv, "q99", 1, ref)]],
                              lo[ci[("null", f"{kind}{k}", nv, "q99", -1, ref)]]) for k in range(20)]
                    out["null_band"][f"{rname}|{d}|{v}|{kind}"] = M.null_band_contrast(real, nulls)
        for d in REAL_DIRS:
            a = lo[[ci[("real", d, "add", q, s, d)] for q in ("q90", "q99") for s in (1, -1)]]
            b = lo[[ci[("real", d, "preserve", q, s, d)] for q in ("q90", "q99") for s in (1, -1)]]
            out["identity_add_vs_preserve"][f"{rname}|{d}"] = float(np.max(np.abs(a - b)))

    fold = np.asarray(spec["fold"])
    h = spec["h"][42].float()
    for d in REAL_DIRS:
        ratio, proj = np.zeros(len(h)), np.zeros(len(h))
        for f in (0, 1):
            te = np.flatnonzero(fold == f)
            u = spec["dirs"][42][(d, f)]
            D = spec["doses"][42][(d, f)]["q99"]
            ratio[te] = ((h[te] + D * u).norm(dim=-1) / (h[te] - D * u).norm(dim=-1)).numpy()
            proj[te] = (h[te] @ u).numpy() / D
        out["asymmetry"][d] = {"norm_ratio_plus_over_minus_median": float(np.median(ratio)),
                               "norm_ratio_range": [float(ratio.min()), float(ratio.max())],
                               "proj_over_dose_median": float(np.median(proj))}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    report(out, readers)


def report(o: dict, readers: dict) -> None:
    print("\n### q99 FollowFeature: cross-item AUROC [CI] / paired win / mean shift ± SD / 0-sub AUROC\n")
    print("| reader | direction | additive (original) | norm-preserving | centred (equal-norm) |")
    print("|---|---|---|---|---|")
    for rname in readers:
        for d in REAL_DIRS:
            cells = []
            for v in VARIANTS:
                s = o["stats"][f"{rname}|{d}|{v}|q99"]
                cells.append(f"{s['auroc']:.2f} [{s['lo']:.2f},{s['hi']:.2f}] / {s['paired_win']:.2f} / "
                             f"{s['mean_shift']:+.2f}±{s['shift_sd']:.2f} / {s['auroc_zero_sub']:.2f}")
            print(f"| {rname} | {d} | " + " | ".join(cells) + " |")
    print("\n### q90 cross-item AUROC (add / preserve / centred)\n")
    for rname in readers:
        print(rname, {d: [round(o["stats"][f"{rname}|{d}|{v}|q90"]["auroc"], 2) for v in VARIANTS]
                      for d in REAL_DIRS})
    print("\n### null band (q99): percentile / z(null SD) / outside 95% band, for paired win | mean shift | AUROC\n")
    print("| reader | direction | variant | null kind | paired win | mean shift | AUROC | AUROC band |")
    print("|---|---|---|---|---|---|---|---|")
    for k, b in o["null_band"].items():
        r, d, v, kind = k.split("|")
        cell = lambda x: f"{x['percentile']:.2f} / {x['z']:+.1f} / {'OUT' if x['outside_two_sided'] else 'in'}"  # noqa: E731
        print(f"| {r} | {d} | {v} | {kind} | {cell(b['paired_win'])} | {cell(b['mean_shift'])} | "
              f"{cell(b['auroc'])} | [{b['auroc']['band_lo']:.2f},{b['auroc']['band_hi']:.2f}] |")
    print("\n### identity add vs preserve (max |ΔLO|)\n", json.dumps(o["identity_add_vs_preserve"], indent=0))
    print("\n### asymmetry\n", json.dumps(o["asymmetry"], indent=0))
    print("\n### INLP held-out AUROC per iteration\n", json.dumps(o["inlp_heldout_auroc"])[:800])
    print("\n### doses L42\n", {k: v for k, v in o["doses"].items() if k.startswith("L42")})


if __name__ == "__main__":
    main()
