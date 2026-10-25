"""Exp 15 report: one z-score table per concept (direction x {model, NLA, OL, J-lens}) and a figure.

    python report15.py --analysis results/vol/analysis15_tb_tablecol.json results/vol/analysis15_tb_paren.json
Columns are the r11 isotropic-null z of the paired +/- shift; the model column is the next-token
log-odds named per concept (newline for the table column, ")" for the parenthesis). The sign is
flipped where the concept's expected behavioural sign is negative so that + always means "toward
values[1] produced the expected behavioural change".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spec15 import DIR_NAMES

MODEL_KEY = {"tb_tablecol": "model_lo_newline", "tb_paren": "model_lo_close"}
MODEL_SIGN = {"tb_tablecol": +1, "tb_paren": -1}  # toward values[1]: more newline / fewer ")"
COLS = (("model", None), ("nla_mean", "NLA"), ("ol_mean", "OL"), ("jlens", "J-lens"))
LABEL = {"cm_raw": "class mean", "cm_jrem4": "class mean, J-removed", "arexpl_raw": "AR(NLA explanations)",
         "arexpl_jrem4": "AR(explanations), J-removed", "arspan_raw": "AR(32-token spans)",
         "arspan_jrem4": "AR(spans), J-removed", "arstmt_raw": "AR(fixed statement)",
         "arstmt_jrem4": "AR(statement), J-removed"}


def z_table(res: dict) -> list[dict]:
    f = res["feature"]
    rows = []
    for d in DIR_NAMES:
        c = res["cells"][f"{d}:r11"]
        row = {"direction": d, "label": LABEL[d],
               "model_z": MODEL_SIGN[f] * c[MODEL_KEY[f]]["iso"]["band"]["z"],
               "model_shift": MODEL_SIGN[f] * c[MODEL_KEY[f]]["iso"]["real"]["mean_shift"]}
        for key, _ in COLS[1:]:
            row[f"{key}_z"] = c[key]["iso"]["band"]["z"]
            row[f"{key}_shift"] = c[key]["iso"]["real"]["mean_shift"]
        rows.append(row)
    return rows


def markdown(res: dict, rows: list[dict]) -> str:
    f = res["feature"]
    head = f"| direction | model {MODEL_KEY[f][9:]} z | NLA z | OL z | J-lens z |\n|---|---:|---:|---:|---:|\n"
    body = "".join(f"| {r['label']} | {r['model_z']:+.1f} | {r['nla_mean_z']:+.1f} | {r['ol_mean_z']:+.1f} | {r['jlens_z']:+.1f} |\n"
                   for r in rows)
    return f"**{f}** (r11 dose, z vs 20 isotropic nulls; + = expected direction)\n\n{head}{body}"


def figure(results: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        rows = z_table(res)
        x = np.arange(len(rows))
        w = 0.2
        series = [("model", "model_z", "#222222"), ("NLA", "nla_mean_z", "#1f77b4"), ("OL", "ol_mean_z", "#ff7f0e"), ("J-lens", "jlens_z", "#2ca02c")]
        for i, (name, key, col) in enumerate(series):
            ax.bar(x + (i - 1.5) * w, [r[key] for r in rows], w, label=name, color=col)
        ax.axhspan(-2, 2, color="grey", alpha=0.15, lw=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in rows], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{res['feature']}: who follows which steering direction", fontsize=10)
        ax.set_ylabel("z vs isotropic nulls (r11)")
        ax.legend(fontsize=8, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", nargs="+", required=True)
    ap.add_argument("--fig", default="results/fig15_who_follows.png")
    a = ap.parse_args()
    results = [json.loads(Path(p).read_text()) for p in a.analysis]
    for res in results:
        print(markdown(res, z_table(res)))
    figure(results, Path(a.fig))
    print("figure:", a.fig)


if __name__ == "__main__":
    main()
