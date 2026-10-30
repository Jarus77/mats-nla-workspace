"""bk_exp11_layer_map: line position is J-enriched in early layers and drops to the gate only around the readers' layers."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

PB = S.ROOT / "experiments" / "11_agam_followups" / "results" / "phaseB"


def leak_z(js):
    names = js["names"]; X = js["scores"].numpy(); ci = {n: i for i, n in enumerate(names)}
    real = (X[ci["jrem+"]] - X[ci["jrem-"]]).mean()
    nulls = []
    for fam in ("iso", "jc"):
        k = 0
        while f"{fam}{k}+" in ci:
            nulls.append((X[ci[f"{fam}{k}+"]] - X[ci[f"{fam}{k}-"]]).mean()); k += 1
    nulls = np.array(nulls)
    return float((real - nulls.mean()) / nulls.std(ddof=1)), len(nulls)


def main():
    MR = torch.load(PB / "map_raw.pt", weights_only=False, map_location="cpu")
    layers = sorted({k[1] for k in MR["cells"]})
    enr = {f: [MR["cells"][(f, L, -1)]["enrichment_raw_384"] for L in layers] for f in ("col", "tense")}
    enr4 = {f: [MR["cells"][(f, L, -1)]["enrichment_raw"] for L in layers] for f in ("col", "tense")}
    lz = [leak_z(MR["jl_scores"][("col", L, -1)])[0] for L in layers]
    rec_json = {r["layer"]: r for r in json.loads((PB.parent / "phaseB.json").read_text())["cells"] if r["feature"] == "col" and r["pos"] == -1}

    fig, (ax, bx) = S.plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"width_ratios": [1.3, 1]})
    ax.axvspan(39.5, 44.5, color=S.C["line"], alpha=0.35, lw=0)
    ax.text(42, 4.25, "readers'\nlayers", ha="center", va="top", color=S.C["muted"], fontsize=12)
    ax.plot(layers, enr["tense"], "-o", color=S.C["muted"], lw=2.2, ms=7, label="tense (past vs present)")
    ax.plot(layers, enr["col"], "-o", color=S.C["probe"], lw=2.8, ms=8, label="line position")
    ax.axhline(1.5, color=S.C["line"], ls="--", lw=1.5); ax.text(57, 1.5, " 1.5 gate", va="center", color=S.C["muted"], fontsize=12)
    ax.axhline(1.0, color=S.C["line"], ls=":", lw=1.5); ax.text(57, 1.0, " random", va="center", color=S.C["muted"], fontsize=12)
    ax.set_xticks(layers, [str(L) for L in layers], fontsize=12)
    ax.set_xlim(22, 57); ax.set_ylim(0.5, 4.3)
    ax.set_xlabel("layer"); ax.set_ylabel("J-space enrichment of the\nclass-mean direction (top 384)")
    ax.legend(loc="lower left", fontsize=12.5)
    ax.set_title("where the feature sits", fontsize=15)

    cols = [S.C["probe"] if z > 3 else S.C["muted"] for z in lz]
    bx.bar(range(len(layers)), lz, 0.62, color=cols)
    for i, z in enumerate(lz):
        if z > 3:
            bx.text(i, z + 0.3, f"{z:+.1f}", ha="center", va="bottom", fontsize=12)
    bx.text(3.5, 1.4, "layers 24-48: |z| < 0.5", ha="center", color=S.C["muted"], fontsize=12)
    bx.axhline(3, color=S.C["line"], ls="--", lw=1.5); bx.text(-0.4, 3.3, "void above +3", ha="left", color=S.C["muted"], fontsize=12)
    bx.axhline(0, color=S.C["muted"], lw=0.8)
    bx.set_xticks(range(len(layers)), [str(L) for L in layers], fontsize=12)
    bx.set_ylim(-2, 22)
    bx.set_xlabel("layer"); bx.set_ylabel("J-lens shift from the J-removed\nline-position push (z vs random)")
    bx.set_title("J-removed push is invisible to the J-lens through 48", fontsize=14)
    fig.tight_layout(w_pad=2.5)

    rec = {24: (3.67, 1.79, 0.0), 32: (2.78, 1.71, -0.2), 40: (1.76, 1.46, 0.2), 42: (1.54, 1.29, 0.0), 44: (1.46, 1.17, 0.4),
           48: (1.45, 1.00, 0.2), 56: (1.42, 0.96, 19.4)}
    checks = []
    for L, (e1, e4, z) in rec.items():
        i = layers.index(L)
        checks.append({"quantity": f"col enrichment r*/4r* at L{L}", "computed": [round(enr['col'][i], 2), round(enr4['col'][i], 2)],
                       "record": [e1, e4], "where": "11 RESULTS.md B1", "match": abs(enr['col'][i] - e1) < 0.006 and abs(enr4['col'][i] - e4) < 0.006})
        checks.append({"quantity": f"col J-lens leak z at L{L}", "computed": round(lz[i], 1), "record": z, "where": "11 RESULTS.md B1",
                       "match": abs(lz[i] - z) < 0.06, "record_json": round(rec_json[L]["jlens_leak_z"], 2)})
    S.save(fig, "bk_exp11_layer_map",
           title="Line position sits inside the J-space early and leaves it only around the readers' layers",
           caption=("Pilot items (60 line-position, 120 tense), read token. Left: J-space enrichment (share of the direction's norm in "
                    "the top-384 J subspace over the random expectation; 1 = random) of each layer's class-mean direction; line position "
                    "falls from 3.7 at layer 24 to about the 1.5 gate at 42-56 while tense stays 3-4x enriched until 52. Right: J-lens "
                    "readout shift when pushing the J-removed line-position direction, as z against 20 random directions (10 isotropic, "
                    "10 J-complement); clean through 48, leaking at 52 (+6.2) and massively at 56. No CI on enrichment (a single direction per cell). Caveat: "
                    "directions are class means over only 60/120 items (cosine 0.38-0.41 with the 40k-row bank direction at 40-44)."),
           section="backup:exp11", role="backup",
           sources=["experiments/11_agam_followups/results/phaseB/map_raw.pt",
                    "experiments/11_agam_followups/results/phaseB.json (cross-check only)"],
           checks=checks,
           notes=("Enrichment values are read from map_raw.pt cells (computed on GPU against per-layer J bases that are not local, so not "
                  "recomputed). Leak z recomputed from map_raw.pt jl_scores: mean(+ minus -) of the J-removed push vs pooled iso+jc nulls "
                  "(the RESULTS text says 'vs the J-complement band'; the analysis code pools both classes). "
                  "Series: " + json.dumps({"layers": layers, "col_r384": [round(x, 2) for x in enr['col']], "tense_r384": [round(x, 2) for x in enr['tense']],
                                          "col_r1536": [round(x, 2) for x in enr4['col']], "leak_z": [round(x, 2) for x in lz]})))
    for c in checks:
        print(c)
    print(layers, [round(x, 2) for x in enr["col"]], [round(x, 2) for x in lz])


if __name__ == "__main__":
    main()
