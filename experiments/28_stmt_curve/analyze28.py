"""Exp 28 analysis (laptop; numpy, scipy, matplotlib; no statsmodels or sklearn in the venv, so the logistic fits are
lib28.logistic_fit, a binomial IRLS). Joins results/rows28.json with results/stmt28_{key}.json and writes
  results/table28.md   one line per row (experiment, type, stated rate, fixed shift, paraphrase mean [min, max], SD,
                       span, readable, used z, the source's own fixed shift) plus the statistics
  results/stats28.json Spearman and Pearson with permutation p (all rows, per type, without the language rows), the
                       memo's band table on the paraphrase mean, the AUROC of the shift for stated >= 0.30, and the
                       binomial logistic fits of stated on shift / on the type indicator / on both, with AIC
  results/fig28.png    scatter: x = paraphrase-mean shift with min-max whiskers, y = NLA stated rate, colour = type,
                       the 0.30 line dashed, labels on the counter-rows and the language rows
Rows whose stmt28 file is not yet in results/ are listed as pending and left out of the statistics.

    .venv/bin/python analyze28.py [--n-perm 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib28 as L  # noqa: E402

RES = HERE / "results"
# Reference palette (dataviz skill, light surface): categorical slot 1 blue, slot 2 orange; text in ink tokens.
COLORS = {"self_inducing": "#2a78d6", "structural": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
LABELS = {"self_inducing": "self-inducing (language, code language, caps, tense)", "structural": "structural"}


def load() -> tuple[list[dict], list[str]]:
    rows = json.loads((RES / "rows28.json").read_text())
    joined, pending = [], []
    for r in rows:
        p = RES / f"stmt28_{r['key']}.json"
        if not p.exists():
            pending.append(r["key"])
            continue
        s = json.loads(p.read_text())
        joined.append({**r, "x": s["shift"]["para_mean"], "fixed_shift": s["shift"]["fixed"], "para_min": s["shift"]["para_min"],
                       "para_max": s["shift"]["para_max"], "para_sd": s["shift"]["para_sd"], "para_range": s["shift"]["para_range"],
                       "span": s["span"], "span_gap": s["span_gap"], "readable": s["readable"], "gap": s["gap"],
                       "shifts_para": s["shifts_para"], "is_type": int(r["type"] == "self_inducing")})
    return joined, pending


def _f(v, nd: int = 2) -> str:  # noqa: ANN001
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def row_table(rows: list[dict]) -> str:
    lines = ["| row | exp | type | stated | fixed shift | paraphrase mean [min, max] | SD | span | readable | used z | source's fixed shift |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["x"]):
        tag = " (counter-row)" if r["counter_row"] else (" (language row)" if r["language_row"] else "")
        pr = r.get("prior") or {}
        lines.append(f"| {r['feature']}{tag} | {r['experiment']} | {r['type'].replace('_', '-')} | {_f(r['stated'])} | {_f(r['fixed_shift'])} | "
                     f"{_f(r['x'])} [{_f(r['para_min'])}, {_f(r['para_max'])}] | {_f(r['para_sd'])} | {_f(r['span'])} | {_f(r['readable'])} | "
                     f"{_f(pr.get('used_z'), 1)} | {_f(pr.get('stmt'))} |")
    return "\n".join(lines)


def _corr(rows: list[dict], n_perm: int) -> dict:
    x = np.array([r["x"] for r in rows])
    y = np.array([r["stated"] for r in rows])
    return {"n": len(rows), "spearman": L.spearman_perm(x, y, n_perm), "pearson": L.pearson_perm(x, y, n_perm)}


def statistics(rows: list[dict], n_perm: int) -> dict:
    out = {"n_rows": len(rows), "correlations": {}}
    subsets = {"all": rows, "self_inducing": [r for r in rows if r["type"] == "self_inducing"],
               "structural": [r for r in rows if r["type"] == "structural"], "without_language_rows": [r for r in rows if not r["language_row"]],
               "all_fixed_shift": [{**r, "x": r["fixed_shift"]} for r in rows]}
    for name, sub in subsets.items():
        out["correlations"][name] = _corr(sub, n_perm) if len(sub) >= 3 else {"n": len(sub)}
    out["bands_para_mean"] = L.band_table(rows, "x", "stated")
    out["bands_fixed"] = L.band_table([{**r, "x": r["fixed_shift"]} for r in rows], "x", "stated")
    x = np.array([r["x"] for r in rows])
    stated = np.array([r["stated"] for r in rows])
    out["auroc_stated_ge_0.30"] = {"para_mean": L.auroc_threshold(x, stated), "fixed": L.auroc_threshold([r["fixed_shift"] for r in rows], stated),
                                   "n_high": int((stated >= L.STATED_LINE).sum()), "n_low": int((stated < L.STATED_LINE).sum())}
    n = np.array([r["n_readouts"] or 320 for r in rows], float)
    k = np.round(stated * n)
    t = np.array([r["is_type"] for r in rows], float)
    fits = {"shift": L.logistic_fit(x[:, None], k, n), "type": L.logistic_fit(t[:, None], k, n),
            "shift_plus_type": L.logistic_fit(np.column_stack([x, t]), k, n), "intercept_only": L.logistic_fit(np.zeros((len(k), 0)), k, n)}
    out["logistic"] = {**fits, "delta_aic_type_minus_shift": fits["type"]["aic"] - fits["shift"]["aic"],
                       "note": "binomial logistic by IRLS; k = round(stated * n readouts); AIC = 2 params - 2 loglik; a lower AIC is better"}
    out["paraphrase_range"] = {"n_rows_range_over_0.15": int(sum(r["para_range"] > 0.15 for r in rows)),
                               "median_range": float(np.median([r["para_range"] for r in rows])),
                               "median_sd": float(np.median([r["para_sd"] for r in rows]))}
    out["counter_rows"] = {r["key"]: {"para_mean": r["x"], "fixed": r["fixed_shift"], "stated": r["stated"]} for r in rows if r["counter_row"]}
    return out


def stats_markdown(st: dict) -> str:
    c = st["correlations"]

    def line(name: str, label: str) -> str:
        d = c[name]
        if "spearman" not in d:
            return f"| {label} | {d['n']} | | | | |"
        return (f"| {label} | {d['n']} | {d['spearman']['rho']:.2f} | {d['spearman']['p']:.3f} | {d['pearson']['r']:.2f} | {d['pearson']['p']:.3f} |")
    out = ["### Correlations of the shift with the NLA stated rate (permutation p, two-sided)", "",
           "| rows | n | Spearman | p | Pearson | p |", "|---|---:|---:|---:|---:|---:|",
           line("all", "all rows, paraphrase-mean shift"), line("all_fixed_shift", "all rows, fixed-pair shift"),
           line("without_language_rows", "without the language rows"), line("self_inducing", "self-inducing rows only"),
           line("structural", "structural rows only"), "",
           "### Bands of the paraphrase-mean shift", "", "| band | n | mean stated | min | max |", "|---|---:|---:|---:|---:|"]
    out += [f"| {b['band']} | {b['n']} | {_f(b['mean_stated'])} | {_f(b['min_stated'])} | {_f(b['max_stated'])} |" for b in st["bands_para_mean"]]
    a = st["auroc_stated_ge_0.30"]
    out += ["", f"AUROC of the shift for stated >= 0.30 ({a['n_high']} vs {a['n_low']} rows): paraphrase mean {_f(a['para_mean'])}, fixed pair {_f(a['fixed'])}.", "",
            "### Binomial logistic fits of the stated rate", "", "| model | params | log-lik | AIC | coefficients |", "|---|---:|---:|---:|---|"]
    for name in ("intercept_only", "shift", "type", "shift_plus_type"):
        f = st["logistic"][name]
        out.append(f"| {name.replace('_', ' ')} | {f['k_params']} | {f['loglik']:.1f} | {f['aic']:.1f} | {', '.join(f'{c:.2f}' for c in f['coef'])} |")
    out += ["", f"AIC(type) - AIC(shift) = {st['logistic']['delta_aic_type_minus_shift']:.1f} (negative: the type indicator fits better).", "",
            f"Paraphrase spread: {st['paraphrase_range']['n_rows_range_over_0.15']} of {st['n_rows']} rows have a min-to-max range over 0.15; "
            f"median range {st['paraphrase_range']['median_range']:.3f}, median SD {st['paraphrase_range']['median_sd']:.3f}."]
    return "\n".join(out)


def figure(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for t in ("self_inducing", "structural"):
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
        xs = np.array([r["x"] for r in sub])
        ys = np.array([r["stated"] for r in sub])
        err = np.vstack([xs - np.array([r["para_min"] for r in sub]), np.array([r["para_max"] for r in sub]) - xs])
        ax.errorbar(xs, ys, xerr=err, fmt="none", ecolor=COLORS[t], elinewidth=1.2, capsize=2.5, alpha=0.7, zorder=2)
        ax.scatter(xs, ys, s=46, color=COLORS[t], edgecolor="#fcfcfb", linewidth=1.5, zorder=3, label=LABELS[t])
    ax.axvline(L.STATED_LINE, color=INK2, lw=1, ls="--", zorder=1)
    ax.text(L.STATED_LINE + 0.01, 0.98, "0.30 line", color=INK2, fontsize=8.5, va="top")
    for r in rows:
        if r["counter_row"] or r["language_row"]:
            lab = f"{r['feature']} ({r['experiment']})"
            ax.annotate(lab, (r["x"], r["stated"]), xytext=(6, 5 if r["language_row"] else -11), textcoords="offset points",
                        fontsize=8, color=INK, zorder=4)
    ax.set_xlabel("statement shift, mean of eight paraphrase pairs (whiskers: min to max)", color=INK)
    ax.set_ylabel("NLA stated rate (share of 320 readouts)", color=INK)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="both", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"Exp 28: statement shift vs NLA stated rate on {len(rows)} rows", fontsize=11, color=INK, loc="left")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=L.N_PERM)
    args = ap.parse_args()
    rows, pending = load()
    if not rows:
        raise SystemExit("no stmt28 files in results/ yet")
    st = statistics(rows, args.n_perm)
    st["pending"] = pending
    (RES / "stats28.json").write_text(json.dumps(st, indent=1, default=float))
    figure(rows, RES / "fig28.png")
    md = [f"# Exp 28 table: {len(rows)} rows with a stated rate and an eight-paraphrase statement shift", "",
          "Shift = ((rec(A) - rec(B)) . d_hat) / gap through the NLA critic at layer 42; `fixed shift` is the source experiment's fixed pair "
          "recomputed here, `source's fixed shift` is the value the source recorded (exp 14's NLA ruler for the exp 12/13 rows). "
          "span = frozen-probe AUROC of the critic's rebuild of the true 32-token span; readable = grouped 5-fold probe AUROC on the true activations.",
          "", row_table(rows), "", stats_markdown(st)]
    if pending:
        md += ["", f"Pending (no stmt28 file yet): {', '.join(pending)}"]
    (RES / "table28.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} rows analysed, {len(pending)} pending; table28.md, stats28.json, fig28.png written")


if __name__ == "__main__":
    main()
