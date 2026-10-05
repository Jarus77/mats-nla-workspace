"""Round-5 analysis (PILOT -- EXCLUDED FROM MAIN ANALYSIS).

    ../../.venv/bin/python r5_analyze.py   -> results/r5_metrics.json, results/r5_stdout.txt
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

from mats import metrics as M  # noqa: E402
from r5_lib import K_NULL, PLAN, ZERO, conditions  # noqa: E402

RES = HERE / "results"
DATA = RES / "08_col_pilot"
SETS = ("col", "digit", "tense")
NB = 1000


def boot_mean(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def effect(p: np.ndarray, m: np.ndarray, nulls: list | None = None) -> dict:
    e = (p - m) / 2
    lo, hi = boot_mean(e)
    out = {"mean": float(e.mean()), "ci": [lo, hi], "sd": float(e.std(ddof=1)), "win": float(np.mean(p > m))}
    if nulls:
        ne = np.array([((a - b) / 2).mean() for a, b in nulls])
        out["null_mean"], out["null_sd"] = float(ne.mean()), float(ne.std(ddof=1))
        out["z"] = float((out["mean"] - ne.mean()) / ne.std(ddof=1))
        out["null_band"] = [float(np.percentile(ne, 2.5)), float(np.percentile(ne, 97.5))]
    return out


# ----------------------------------------------------------------------------- behaviour
def behaviour() -> dict:
    out: dict = {}
    f0 = DATA / "r5_behave0.pt"
    if f0.exists():
        b = torch.load(f0, weights_only=False)
        out["natural_newline_lo"] = {"mean": float(b["zero"].mean()), "sd": float(b["zero"].std())}
        ncls = {"raw": "iso", "jrem1": "nonj1", "jrem4": "nonj4"}
        for d in ("raw", "jrem1", "jrem4"):
            nulls = [(b[("L42null", d, ncls[d], k, 1)], b[("L42null", d, ncls[d], k, -1)]) for k in range(K_NULL)]
            for q in ("q90", "q99"):
                out[f"L42|{d}|{q}"] = effect(b[("L42", d, q, 1)], b[("L42", d, q, -1)], nulls if q == "q90" else None)
        nulls = [(b[("L42r11null", "nonj4", k, 1)], b[("L42r11null", "nonj4", k, -1)]) for k in range(K_NULL)]
        nulls_iso = [(b[("L42r11null", "iso", k, 1)], b[("L42r11null", "iso", k, -1)]) for k in range(K_NULL)]
        for d in ("reg_raw", "reg_jrem4", "cm_raw", "cm_jrem4"):
            e = effect(b[("L42r11", d, 1)], b[("L42r11", d, -1)], nulls)
            e["z_iso"] = effect(b[("L42r11", d, 1)], b[("L42r11", d, -1)], nulls_iso)["z"]
            out[f"L42r11|{d}"] = e
    f1 = DATA / "r5_behave1.pt"
    if f1.exists():
        b = torch.load(f1, weights_only=False)
        ncls = {"raw": "iso", "jrem1": "nonj1", "jrem4": "nonj4"}
        for d in ("raw", "jrem1", "jrem4"):
            nulls = [(b[("L38null", d, ncls[d], k, 1)], b[("L38null", d, ncls[d], k, -1)]) for k in range(K_NULL)]
            for v in ("centred", "add"):
                for q in ("q90", "q99"):
                    out[f"L38|{v}|{d}|{q}"] = effect(b[("L38", v, d, q, 1)], b[("L38", v, d, q, -1)],
                                                     nulls if (v == "centred" and q == "q90") else None)
    f2 = DATA / "r5_behave2.pt"
    if f2.exists():
        b = torch.load(f2, weights_only=False)
        spec = torch.load(DATA / "r5_spec.pt", weights_only=False)
        y = np.array([it["next_digit"] for it in spec["digit"]["items"]])
        z = b["zero"]
        out["digit_natural"] = {"argmax_digit_acc": float(np.mean(z["argmax_digit"] == y)),
                                "digit_is_top1_token": float(np.mean(z["p_digit_top1"])),
                                "lo75_auroc": M.auroc(z["lo75"][y == 7], z["lo75"][y == 5])}
        nulls = [(b[("digitnull", "nonj4", k, 1)]["lo75"], b[("digitnull", "nonj4", k, -1)]["lo75"]) for k in range(K_NULL)]
        for d in ("raw", "jrem4"):
            for q in ("q90", "r11", "r14"):
                p, m = b[("digit", d, q, 1)], b[("digit", d, q, -1)]
                e = effect(p["lo75"], m["lo75"], nulls if q == "r11" else None)
                e["argmax7_plus"] = float(np.mean(p["argmax_digit"] == 7))
                e["argmax5_minus"] = float(np.mean(m["argmax_digit"] == 5))
                e["flip_5to7_plus"] = float(np.mean((p["argmax_digit"] == 7) & (y == 5)) / max(np.mean(y == 5), 1e-9))
                e["flip_7to5_minus"] = float(np.mean((m["argmax_digit"] == 5) & (y == 7)) / max(np.mean(y == 7), 1e-9))
                out[f"digit|{d}|{q}"] = e
    # generation at L38
    gens = {}
    for f in glob.glob(str(DATA / "r5_gen*.pt")):
        gens.update(torch.load(f, weights_only=False)["texts"])
    if gens:
        spec0 = torch.load(DATA / "spec.pt", weights_only=False)
        items = spec0["items"]
        rem = np.array([it["chars_remaining"] for it in items], float)
        col = np.array([it["col"] for it in items], float)

        def first_nl(ts):  # noqa: ANN001, ANN202
            return np.array([t.find("\n") if "\n" in t else np.nan for t in ts], float)

        z = first_nl(gens[("zero", None, None, 0)])
        ok = np.isfinite(z)
        out["gen_natural"] = {"frac_newline": float(ok.mean()),
                              "spearman_vs_chars_remaining": float(
                                  __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(z[ok], rem[ok]).statistic),
                              "mean_wrap_col": float(np.nanmean(z + col)),
                              "mae_vs_true_remaining": float(np.nanmean(np.abs(z - rem)))}
        for d in ("raw", "jrem1", "jrem4"):
            for q in ("q90", "q99"):
                if ("add", d, q, 1) not in gens or ("add", d, q, -1) not in gens:
                    continue
                p = first_nl(gens[("add", d, q, 1)])
                m = first_nl(gens[("add", d, q, -1)])
                both = np.isfinite(p) & np.isfinite(m)
                diff = (p - m)[both]
                lo, hi = boot_mean(diff)
                out[f"gen|{d}|{q}"] = {"plus_minus_chars_mean": float(diff.mean()), "ci": [lo, hi],
                                       "n_both": int(both.sum()), "plus_vs_zero": float(np.nanmean(p - z)),
                                       "minus_vs_zero": float(np.nanmean(m - z)),
                                       "frac_items_plus_breaks_earlier": float(np.mean(diff < 0))}
    return out


# ----------------------------------------------------------------------------- readers
def load(reader: str) -> dict:
    out: dict = {}
    for f in sorted(glob.glob(str(DATA / f"r5_{reader}_chunk*.pt"))):
        d = torch.load(f, weights_only=False)
        for S, blk in d["lo"].items():
            if S not in out:
                out[S] = [np.full((blk["lo"].shape[0], len(conditions(S)), blk["lo"].shape[2]), np.nan), d["templates"][S]]
            out[S][0][:, blk["idx"], :] = blk["lo"].numpy()
    return out


def fam_scores(lo: np.ndarray, temps: list) -> dict:
    out = {fam: lo[[i for i, t in enumerate(temps) if t[0] == fam]].mean(0) for fam in dict.fromkeys(t[0] for t in temps)}
    out |= {f"t{i}": lo[i] for i in range(len(temps))}
    return out


def cell(S: str, X: np.ndarray, d: str, q: str, boot: bool) -> dict:
    conds = conditions(S)
    ci = {c: i for i, c in enumerate(conds)}
    p, m = X[ci[("real", d, q, 1, d)]], X[ci[("real", d, q, -1, d)]]
    if np.isnan(p).any():
        return {"missing": True}
    r = M.follow_feature(p, m, n_boot=NB) if boot else M.paired_stats(p, m) | {"lo": np.nan, "hi": np.nan}
    out = {"auroc": r["auroc"], "lo": r["lo"], "hi": r["hi"], "paired_win": r["paired_win"],
           "mean_shift": r["mean_shift"], "frac": 2 * (r["auroc"] - 0.5), "bands": {}}
    for qq, ref in PLAN[S]["nulls"]:
        if qq != q or (ref is not None and ref != d):
            continue
        for cls in ("iso", "nonj4"):
            nl = [(X[ci[("null", f"{cls}_{k}", q, 1, ref)]], X[ci[("null", f"{cls}_{k}", q, -1, ref)]]) for k in range(K_NULL)]
            if any(np.isnan(a).any() for a, _ in nl):
                continue
            out["bands"][cls] = M.null_band_contrast((p, m), nl)
    return out


def readers() -> dict:
    jl = torch.load(DATA / "r5_jlens.pt", weights_only=False)
    dec = {"OL@42": load("ol"), "NLA@42": load("nla")}
    out = {}
    for S in SETS:
        tabs = {}
        for name, by in dec.items():
            if S in by:
                tabs[name] = fam_scores(*by[S])
        tabs["JLens@42"] = fam_scores(jl[S]["lo"].numpy(), jl[S]["templates"])
        if S == "col":
            tabs["JLens@42"]["eol"] = jl[S]["eol"].numpy()
        for name, fams in tabs.items():
            for key, X in fams.items():
                for d in PLAN[S]["dirs"]:
                    for q in PLAN[S]["doses"]:
                        out[f"{S}|{name}|{key}|{d}|{q}"] = cell(S, X, d, q, boot=not key.startswith("t"))
    return out


def _z(c: dict, cls: str, k: str = "mean_shift") -> str:
    b = c.get("bands", {}).get(cls)
    return f"{b[k]['z']:+.1f}{'*' if b[k]['outside_two_sided'] else ''}" if b else "–"


def main() -> None:
    out = {"behaviour": behaviour()}
    try:
        out["readers"] = readers()
    except FileNotFoundError as e:
        out["readers"] = {}
        print("readers missing:", e)
    (RES / "r5_metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print("# round-5 pilot -- PILOT, EXCLUDED FROM MAIN ANALYSIS\n\n## behaviour\n")
    for k, v in out["behaviour"].items():
        print(k, json.dumps(v, default=lambda x: round(float(x), 3)))
    print("\n## reader cells (centred): AUROC [CI] / win / shift / z_shift iso,Jc / z_AUROC iso,Jc\n")
    for k, c in out["readers"].items():
        if c.get("missing") or k.split("|")[2].startswith("t"):
            continue
        print(f"{k}: {c['auroc']:.2f} [{c['lo']:.2f},{c['hi']:.2f}] / {c['paired_win']:.2f} / {c['mean_shift']:+.3f} / "
              f"{_z(c, 'iso')} {_z(c, 'nonj4')} / {_z(c, 'iso', 'auroc')} {_z(c, 'nonj4', 'auroc')}")
    print("\n## per-template AUROC\n")
    for k, c in out["readers"].items():
        if not c.get("missing") and k.split("|")[2].startswith("t"):
            print(k, f"{c['auroc']:.2f}")


if __name__ == "__main__":
    main()
