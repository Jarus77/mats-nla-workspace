"""Local: metrics for the 03 smoke from results/stage_base.pt, stage_nla.pt, gemini_textnla.json.

Writes results/metrics.json and prints markdown tables (pasted into RESULTS.md).
    ../../.venv/bin/python analyze.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mats import metrics as M  # noqa: E402

RES = HERE / "results"
DOSES = ["q90", "q99", "2q99"]
PRICE = {"H100": 3.95, "H200": 4.54}  # Modal list $/GPU-h (approx., 2026-10)
LEAK_T = 0.55
NB = 2000


def cidx(conds, name, sgn):
    return [i for i, c in enumerate(conds) if c[0] == name and c[1] == sgn][0]


def ff(lo, conds, pair, dose, zero=False):
    """lo: [T, C, N] or [C, N]. Template-mean LO -> FollowFeature at dose."""
    lo = np.asarray(lo, float)
    if lo.ndim == 3:
        lo = lo.mean(0)
    p, m = lo[cidx(conds, dose, 1)], lo[cidx(conds, dose, -1)]
    z = lo[cidx(conds, "zero", 0)] if zero else None
    return M.follow_feature(p, m, clusters=pair, lo_zero=z, n_boot=NB, seed=0)


def fmt(r):
    return f"{r['auroc']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]"


def main() -> None:
    B = torch.load(RES / "stage_base.pt", weights_only=False)
    Nl = torch.load(RES / "stage_nla.pt", weights_only=False) if (RES / "stage_nla.pt").exists() else None
    G = json.loads((RES / "gemini_textnla.json").read_text()) if (RES / "gemini_textnla.json").exists() else None
    conds, pair, y = B["conds"], np.asarray(B["pair"]), np.asarray(B["y"])
    out: dict = {"follow_feature": {}, "per_template": {}, "fraction_of_probe": {}, "paired_null": {},
                 "page": {}, "leak": {}, "natural": {}, "fve_guard": {}, "armU": {}}

    readers = {}
    for kind in ("raw", "jrem"):
        if Nl is not None:
            readers[("NLA@42", kind)] = Nl["lo_nla"][kind].numpy()
        for L in (42, 40, 44):
            readers[(f"OL@{L}", kind)] = B["lo_ol"][L][kind].numpy()
        for L in (40, 42, 44):
            readers[(f"JLens@{L}", kind)] = B["lo_jl"][L][kind].numpy()
        readers[("probe@42", kind)] = B["probe_lo"][42][kind].numpy()

    # ---------------- FollowFeature per reader x direction x dose, + null, fraction, page
    for (name, kind), lo in readers.items():
        key = f"{name}|{kind}"
        out["follow_feature"][key] = {d: ff(lo, conds, pair, d) for d in DOSES + ["null_q99"]}
        out["follow_feature"][key + "|zero_sub"] = {d: ff(lo, conds, pair, d, zero=True) for d in DOSES}
        if lo.ndim == 3:
            out["per_template"][key] = {d: [ff(lo[t], conds, pair, d)["auroc"] for t in range(lo.shape[0])]
                                        for d in DOSES}
        mean = lo.mean(0) if lo.ndim == 3 else lo
        probe = B["probe_lo"][42 if "@42" in name else int(name.split("@")[1])][kind].numpy()
        lab = np.concatenate([np.ones(len(y)), np.zeros(len(y))])
        cl = np.concatenate([pair, pair])
        q = cidx(conds, "q99", 1), cidx(conds, "q99", -1)
        out["fraction_of_probe"][key] = M.fraction_of_probe_ci(
            np.concatenate([mean[q[0]], mean[q[1]]]), np.concatenate([probe[q[0]], probe[q[1]]]),
            lab, cl, n_boot=NB)
        nq = cidx(conds, "null_q99", 1), cidx(conds, "null_q99", -1)
        out["paired_null"][key] = M.paired_null_auroc((mean[q[0]], mean[q[1]]), (mean[nq[0]], mean[nq[1]]),
                                                      clusters=pair, n_boot=NB)
        shifts = np.stack([mean[cidx(conds, d, 1)] - mean[cidx(conds, d, -1)] for d in DOSES], 1)
        out["page"][key] = M.page_trend_test(shifts)

    # ---------------- leak detector (arm R sanity; arm U below)
    for L in (40, 42, 44):
        for kind in ("jrem", "raw"):
            r = out["follow_feature"][f"JLens@{L}|{kind}"]
            out["leak"][f"armR_L{L}|{kind}"] = {d: {"auroc": r[d]["auroc"], "void": r[d]["auroc"] > LEAK_T,
                                                    "moved_two_sided": abs(r[d]["auroc"] - 0.5) > LEAK_T - 0.5,
                                                    "mean_shift": r[d]["mean_shift"]}
                                                for d in DOSES}
    for kind, a in B["armU"].items():
        row = {"kl_mean": a["kl_mean"]}
        for L in (40, 42, 44):
            r = M.follow_feature(a[f"L{L}_jlens_lo_plus"], a[f"L{L}_jlens_lo_minus"], clusters=pair,
                                 n_boot=NB)
            row[f"L{L}_jlens_followfeature"] = r
            row[f"L{L}_leak_void"] = r["auroc"] > LEAK_T
            row[f"L{L}_probe_shift_frac_median"] = float(np.median(a[f"L{L}_probe_shift_frac"]))
        out["armU"][kind] = row

    # ---------------- natural arm (unsteered; AUROC by true label, clustered by pair)
    def nat(scores):
        r = M.auroc_ci(np.asarray(scores, float), y, pair, n_boot=NB)
        return {"auroc": r.auroc, "lo": r.lo, "hi": r.hi}

    zero = cidx(conds, "zero", 0)
    if Nl is not None:
        out["natural"]["NLA@42"] = nat(Nl["lo_nla"]["raw"].numpy().mean(0)[zero])
    out["natural"]["OL_sweep"] = {int(L): nat(v.numpy().mean(0)) for L, v in B["lo_ol_sweep"].items()}
    for L in (40, 42, 44):
        out["natural"][f"JLens@{L}"] = nat(B["lo_jl_nat"][L].numpy())
    out["natural"]["TextNLA_local_prompt"] = nat(B["lo_textnla"].numpy().mean(0))
    out["natural"]["probe_G1_by_layer"] = {int(L): v["G1_auroc"] for L, v in B["probes"].items()}
    if G is not None:
        lp = np.asarray(G["lo_prompt"], float)
        ok = np.isfinite(lp)
        out["natural"]["TextNLA_gemini_prompt"] = {**nat(np.where(ok, lp, 0.0)), "n_nan": int((~ok).sum())}
        if "lo_expl" in G:
            ii = np.asarray(G["expl_items"])
            le = np.asarray(G["lo_expl"], float)
            r = M.auroc_ci(np.nan_to_num(le), y[ii], pair[ii], n_boot=NB)
            out["natural"]["TextNLA_gemini_on_NLA_explanation"] = {"auroc": r.auroc, "lo": r.lo,
                                                                     "hi": r.hi, "n": len(ii)}
        out["gemini_cost"] = {k: G[k] for k in ("n_calls", "n_failed", "spend_usd", "s_per_call")}

    # ---------------- FVE guard (jrem direction, k=4, paired per example)
    if Nl is not None:
        fv = Nl["fve_per_example"].numpy()
        names = ["zero", "q90+", "q90-", "q99+", "q99-", "2q99+", "2q99-"]
        out["fve_guard"] = {names[c]: M.fve_guard(fv[0], fv[c], margin=0.05, n_boot=NB) for c in range(1, 7)}
        out["fve_guard"]["natural_mean"] = float(np.nanmean(fv[0]))
        out["fve_guard"]["n_failed_extractions"] = Nl["fve_n_failed"]

    # ---------------- compute / cost
    tb = B["timing"]
    cost = {"base": {"gpu": B["gpu"], "wall_s": tb["total_s"], "sections": tb["sections"],
                     "ol_s_per_pair": tb["ol_s_per_pair"], "ol_gen_s_per_row": tb["ol_gen_s_per_row_b16"],
                     "peak_mem_gb": B["peak_mem_gb"]}}
    if Nl is not None:
        tn = Nl["timing"]
        cost["nla"] = {"gpu": Nl["gpu"], "wall_s": tn["total_s"], "sections": tn["sections"],
                       "nla_s_per_pair": tn["nla_s_per_pair"],
                       "nla_gen_recon_s_per_sample": tn["nla_gen_plus_recon_s_per_sample"],
                       "peak_mem_gb": Nl["peak_mem_gb"]}
    usd = 0.0
    for k, v in cost.items():
        p = PRICE["H200"] if "H200" in v["gpu"] else PRICE["H100"]
        v["usd"] = v["wall_s"] / 3600 * p
        usd += v["usd"]
    cost["total_usd_gpu"] = usd
    cost["n_items"] = len(y)
    out["cost"] = cost
    out["checks"] = {"hook_check": B["hook_check"], "overlap_02": B["overlap_02"], "doses": B["doses"],
                     "norm_ratio": B["norm_ratio"], "enrichment_w": B["enrichment_w"], "rstar": B["rstar"],
                     "probes": {int(k): v for k, v in B["probes"].items()}}
    out["samples"] = {"ol_gens": B["ol_gens"][:4], "jl_top": {int(k): v for k, v in B["jl_top"].items()}}
    if Nl is not None:
        out["samples"]["nla_natural"] = [r[0] for r in Nl["natural_explanations"][:3]]
        out["samples"]["nla_2q99_plus"] = [r[0] for r in Nl["steered_explanations_2q99_plus"][:3]]
    (RES / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    report(out)


def report(o: dict) -> None:
    print("\n## FollowFeature (arm R, template-mean LO; 95% pair-clustered CI)\n")
    print("| reader | dir | q90 | q99 | 2q99 | null q99 | frac probe @q99 | Page p |")
    print("|---|---|---|---|---|---|---|---|")
    for key, r in o["follow_feature"].items():
        if key.endswith("zero_sub"):
            continue
        name, kind = key.split("|")
        fp = o["fraction_of_probe"][key]
        print(f"| {name} | {kind} | {fmt(r['q90'])} | {fmt(r['q99'])} | {fmt(r['2q99'])} | "
              f"{fmt(r['null_q99'])} | {fp['fraction']:.2f} | {o['page'][key]['p']:.3g} |")
    print("\n## zero-dose-subtracted FollowFeature (q99) / paired win (q99)\n")
    for key, r in o["follow_feature"].items():
        if key.endswith("zero_sub"):
            base = o["follow_feature"][key.rsplit("|", 1)[0]]["q99"]
            print(f"- {key[:-9]}: zero-sub {fmt(r['q99'])}; paired win {base['paired_win']:.2f}; "
                  f"mean shift {base['mean_shift']:.2f}")
    print("\n## per-template AUROC (q99)\n")
    for k, v in o["per_template"].items():
        print(f"- {k}: {[round(x, 2) for x in v['q99']]}")
    print("\n## leak\n", json.dumps(o["leak"], indent=0, default=float)[:1500])
    print("\n## arm U\n", json.dumps(o["armU"], indent=0, default=float)[:3000])
    print("\n## natural\n", json.dumps(o["natural"], indent=0, default=float)[:3000])
    print("\n## FVE guard\n", json.dumps(o["fve_guard"], indent=0, default=float)[:3000])
    print("\n## cost\n", json.dumps(o["cost"], indent=1, default=float))
    print("\n## checks\n", json.dumps({k: v for k, v in o["checks"].items() if k != "probes"}, indent=0,
                                      default=float)[:3000])
    print("\n## probes\n", json.dumps(o["checks"]["probes"], indent=0, default=float)[:3000])


if __name__ == "__main__":
    main()
