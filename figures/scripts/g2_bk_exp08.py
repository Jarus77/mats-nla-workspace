"""bk_exp08_matched_dose: at equal push size the readers follow language and tense, not line position."""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

E08 = S.ROOT / "experiments" / "08_col_pilot"
sys.path.insert(0, str(E08))
sys.path.insert(0, str(S.ROOT / "src"))
import calib_analyze as CA  # noqa: E402
from calib_lib import K_NULL, conditions  # noqa: E402
from calib_templates import PRIMARY  # noqa: E402

FEATS = [("lang", "language\n(Russian vs Polish)"), ("tense", "tense\n(past vs present)"),
         ("col", "line position\n(characters into the line)")]
READERS = [("NLA", "nla"), ("OL", "ol")]


def pair_auroc(p, m):
    return S.auroc(np.r_[p, m], np.r_[np.ones(len(p)), np.zeros(len(m))])


def main():
    dec = {"NLA": CA.load("nla"), "OL": CA.load("ol")}
    rec = json.loads((E08 / "results" / "calib_metrics.json").read_text())["cells"]
    res, checks = {}, []
    for F, _ in FEATS:
        for R, _ in READERS:
            lo, temps = dec[R][(F, 42)]
            X = CA.scores(lo, temps)[PRIMARY[F]]
            ci = {c: i for i, c in enumerate(conditions(F, 42))}
            for q in ("r11", "r14"):
                p, m = X[ci[("real", "raw", q, 1, "raw")]], X[ci[("real", "raw", q, -1, "raw")]]
                a = pair_auroc(p, m)
                rng = np.random.default_rng(0)
                boots = []
                for _ in range(2000):
                    i = rng.integers(0, len(p), len(p))
                    boots.append(pair_auroc(p[i], m[i]))
                nulls = [pair_auroc(X[ci[("null", f"nonj4_{k}", q, 1, "raw")]], X[ci[("null", f"nonj4_{k}", q, -1, "raw")]])
                         for k in range(K_NULL)]
                res[(F, R, q)] = dict(a=a, lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                                      nmin=min(nulls), nmax=max(nulls), nmean=np.mean(nulls), nsd=np.std(nulls, ddof=1), n=len(p))
                r = rec[f"{F}|{R}@42|{PRIMARY[F]}|raw|{q}"]["auroc"]
                checks.append({"quantity": f"{R} {F} AUROC at {q}", "computed": round(a, 3), "record_json": round(r, 3),
                               "record": None, "where": "08 RESULTS.md (c) table / calib_metrics.json", "match": None})
    md = {("NLA", "lang", "r11"): .87, ("NLA", "tense", "r11"): .85, ("NLA", "col", "r11"): .48,
          ("OL", "lang", "r11"): .80, ("OL", "tense", "r11"): .56, ("OL", "col", "r11"): .48,
          ("NLA", "lang", "r14"): .92, ("NLA", "tense", "r14"): .89, ("NLA", "col", "r14"): .47,
          ("OL", "lang", "r14"): .85, ("OL", "tense", "r14"): .57, ("OL", "col", "r14"): .47}
    for c in checks:
        R, F, _, _, q = c["quantity"].split()
        c["record"] = md[(R, F, q)]
        c["match"] = abs(c["computed"] - c["record"]) < 0.006

    fig, ax = S.plt.subplots(figsize=S.FULL)
    w = 0.34
    for fi, (F, flab) in enumerate(FEATS):
        for ri, (R, key) in enumerate(READERS):
            x = fi + (ri - 0.5) * w
            d = res[(F, R, "r11")]
            ax.add_patch(S.plt.Rectangle((x - w * 0.46, d["nmin"]), w * 0.92, d["nmax"] - d["nmin"],
                                         color=S.C["line"], alpha=0.8, zorder=1, lw=0))
            b = ax.bar(x, d["a"] - 0.3, w * 0.62, bottom=0.3, color=S.C[key], zorder=2,
                       label=("NLA" if R == "NLA" else "Oracle Lens") if fi == 0 else None)
            ax.errorbar(x, d["a"], yerr=[[d["a"] - d["lo"]], [d["hi"] - d["a"]]], color=S.C["ink"], lw=1.4, capsize=4, zorder=3)
            ax.text(x, d["hi"] + 0.012, f"{d['a']:.2f}", ha="center", va="bottom", fontsize=13)
    ax.add_patch(S.plt.Rectangle((0, 0), 0, 0, color=S.C["line"], label="range of 20 random directions, same size"))
    ax.set_xticks(range(3), [f[1] for f in FEATS])
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(-0.6, 2.75)
    ax.set_ylabel("AUROC, + push vs - push (0.5 = chance)")
    S.chance_line(ax)
    ax.annotate("same push size,\nno response\n(inside the random band)", xy=(2 - 0.5 * w + 0.05, 0.585), xytext=(2.3, 0.72), ha="center", fontsize=13,
                color=S.C["muted"], arrowprops=dict(arrowstyle="->", color=S.C["muted"]))
    ax.legend(loc="upper right", fontsize=13)
    S.save(fig, "bk_exp08_matched_dose",
           title="At an equal push, both readers follow language and the NLA follows tense, but neither follows line position",
           caption=("Pilot (excluded from main analysis), layer 42, 60 items per feature. Each item's activation is pushed "
                    "along the feature's raw class-mean (or regression) direction by +/-0.11 of its own norm (centred), and the "
                    "reader scores fixed label phrasings; AUROC separates the + from the - version (0.5 = no response). "
                    "Error bars: 95% item bootstrap (2,000). Grey box: range of AUROCs for 20 random J-complement directions "
                    "pushed the same way. At 0.14 of the norm the picture is the same (NLA 0.92 / 0.89 / 0.47). Caveat: fixed "
                    "templates; the line-position direction is a regression direction, the others class-mean differences."),
           section="backup:exp08", role="backup",
           sources=["experiments/08_col_pilot/results/08_col_pilot/calib_nla_chunk*.pt",
                    "experiments/08_col_pilot/results/08_col_pilot/calib_ol_chunk*.pt",
                    "experiments/08_col_pilot/results/calib_metrics.json (cross-check only)"],
           checks=checks,
           notes=("Primary template families: lang/tense 'label' (3 pairs each, e.g. NLA 'The text is written in Russian' vs "
                  "'... in Polish'; 'The verb is in the past tense' vs '... present tense'; Oracle Lens '- Russian language', '- was' vs '- is'); "
                  "col 'count_qual' ('near the end of a long line' vs 'at the start of a line'; 'a long line' vs 'a short line'). "
                  "Scores are template-family mean log-odds, loaded with calib_analyze.load/scores (read-only import). "
                  "Language here is 02's Russian-Polish direction, not French/German. Null band = 20 J-complement (4r*) "
                  "unit directions at the same per-item norms. "
                  "r14 values: " + json.dumps({f"{k[1]} {k[0]}": round(v['a'], 3) for k, v in res.items() if k[2] == 'r14'})))
    for k, v in res.items():
        print(k, {kk: round(float(vv), 3) for kk, vv in v.items()})
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
