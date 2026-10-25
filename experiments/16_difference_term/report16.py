"""Exp 16 report: per row, which reader follows which component after the swap. Figure + markdown.

    python report16.py --analysis results/analysis16_*.json --fig results/fig16_difference_term.png
Bars: AUROC of the reader's mean-phrasing readout against the DONOR label (j) for swap_res (OLens part
frozen, difference term from j) and swap_ol (difference term frozen, OLens part from j). Near 1 =
the readout follows the swapped component; near 0 = it stays with the frozen one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spec16 import ROWS

LABEL = {"tb_tablecol": "table column", "tb_paren": "parenthesis", "ij2_lang": "language", "bk_listidx": "list index"}


def rows(results: list[dict]) -> list[dict]:
    out = []
    for a in results:
        r = a["readers"]
        out.append({"feature": a["feature"], "role": a["role"], "label": LABEL[a["feature"]],
                    "nla_res_j": r["nla"]["swap_res"]["mean"]["follows_j"], "nla_ol_j": r["nla"]["swap_ol"]["mean"]["follows_j"],
                    "ol_res_j": r["ol"]["swap_res"]["mean"]["follows_j"], "ol_ol_j": r["ol"]["swap_ol"]["mean"]["follows_j"],
                    "nla_h": r["nla"]["h"]["mean"]["follows_i"], "ol_h": r["ol"]["h"]["mean"]["follows_i"],
                    "nla_rol": r["nla"]["r_ol"]["mean"]["follows_i"], "probe_rol": a["probe_auroc"]["r_ol"],
                    "probe_res": a["probe_auroc"]["res"], "fve_rol": a["fve"]["r_ol"], "fve_rnla": a["fve"]["r_nla"]})
    return out


def markdown(rs: list[dict]) -> str:
    h = ("| row | NLA on h | NLA on OL-part alone | NLA follows donor: difference term swapped | OLens part swapped | "
         "OL follows donor: difference term swapped | OLens part swapped |\n|---|---:|---:|---:|---:|---:|---:|\n")
    b = "".join(f"| {r['label']} ({r['role']}) | {r['nla_h']:.2f} | {r['nla_rol']:.2f} | **{r['nla_res_j']:.2f}** | **{r['nla_ol_j']:.2f}** | "
                f"{r['ol_res_j']:.2f} | {r['ol_ol_j']:.2f} |\n" for r in rs)
    return h + b


def figure(rs: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    x = np.arange(len(rs)); w = 0.36
    for ax, reader, name in zip(axes, ("nla", "ol"), ("NLA readout", "Oracle Lens readout")):
        ax.bar(x - w / 2, [r[f"{reader}_res_j"] for r in rs], w, color="#1f77b4", label="OLens part frozen, difference term from donor")
        ax.bar(x + w / 2, [r[f"{reader}_ol_j"] for r in rs], w, color="#ff7f0e", label="difference term frozen, OLens part from donor")
        ax.axhline(0.5, color="k", lw=0.6, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in rs])
        ax.set_title(f"{name}: AUROC against the donor's label", fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("follows the donor (1) vs stays with the frozen part (0)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", nargs="+", required=True)
    ap.add_argument("--fig", default="results/fig16_difference_term.png")
    a = ap.parse_args()
    results = sorted((json.loads(Path(p).read_text()) for p in a.analysis), key=lambda d: list(ROWS).index(d["feature"]))
    rs = rows(results)
    print(markdown(rs)); figure(rs, Path(a.fig)); print("figure:", a.fig)


if __name__ == "__main__":
    main()
