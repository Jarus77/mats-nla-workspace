"""Compact headline table (md + png) from results/analysis14_*.json. Layer 44 for the OL ruler."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spec14 import ROWS  # noqa: E402

NAMES = {"tb_tablecol": "table column", "tb_paren": "inside a parenthesis", "ij2_lang": "language (FR/DE)",
         "bk_listidx": "bullet index"}
L = "44"


def row(f: str, d: dict) -> dict:
    o, n = d["ol"]["layers"][L], d["nla"]
    st_ol = o["stmt"]
    ol_fixed = next(v for k, v in st_ol.items() if k.startswith("ol:") and k.endswith(":fixed"))
    ol_best = max((v for k, v in st_ol.items() if k.startswith("ol:")), key=lambda v: v["ratio"])
    nla_fixed = next(v for k, v in n["stmt"].items() if k.endswith(":fixed"))
    return {"concept": NAMES[f], "role": ROWS[f]["role"],
            "ol_true_text": o["span32"]["probe_retrained_on_recon"],
            "ol_bullets_transfer": o["ol_bullets"]["probe_from_span32"],
            "ol_bullets_retrained": o["ol_bullets"]["probe_retrained_on_recon"],
            "ol_bullets_gap": o["ol_bullets"]["gap_vs_span32"]["ratio"],
            "ol_stmt_fixed": ol_fixed["ratio"], "ol_stmt_best": ol_best["ratio"],
            "nla_recon_probe": n["nla_expl"]["probe"]["recon"], "nla_gap": n["nla_expl"]["gap"]["ratio"],
            "nla_fve": n["nla_expl"]["fve"], "nla_stmt_fixed": nla_fixed["ratio"]}


def main() -> None:
    rows = [row(f, json.loads((HERE / "results" / f"analysis14_{f}.json").read_text())) for f in ROWS]
    cols = [("ol_true_text", "OL ruler:\ntrue text\n(probe)"), ("ol_bullets_transfer", "OL ruler:\nOLens bullets\n(true-text probe)"),
            ("ol_bullets_retrained", "OL ruler:\nOLens bullets\n(own probe)"), ("ol_bullets_gap", "OL ruler:\nbullets\ngap kept"),
            ("ol_stmt_fixed", "OL ruler:\n'- X language'\nstyle statement"), ("ol_stmt_best", "OL ruler:\nbest bullet\nstatement"),
            ("nla_fve", "NLA ruler:\nFVE"), ("nla_recon_probe", "NLA ruler:\nexplanation\n(probe)"), ("nla_gap", "NLA ruler:\ngap kept"),
            ("nla_stmt_fixed", "NLA ruler:\nfixed-label\nstatement")]
    md = "| concept | " + " | ".join(c[1].replace("\n", " ") for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    md += "".join(f"| {r['concept']} | " + " | ".join(f"{r[k]:.2f}" for k, _ in cols) + " |\n" for r in rows)
    (HERE / "results" / "headline.md").write_text(md)
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.axis("off")
    cell = [[f"{r[k]:.2f}" for k, _ in cols] for r in rows]
    t = ax.table(cellText=cell, rowLabels=[f"{r['concept']}  ({'NLA leads' if r['role'] == 'nla_leads' else 'control'})" for r in rows],
                 colLabels=[c[1] for c in cols], loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.6)
    for (i, j), c in t.get_celld().items():
        if i == 0:
            c.set_height(c.get_height() * 2.2); c.set_text_props(weight="bold"); c.set_facecolor("#EEF0EC")
        elif j < 0:
            c.set_text_props(weight="bold"); c.set_facecolor("#EEF0EC")
        else:
            v = float(cell[i - 1][j]); k = cols[j][0]
            if k in ("ol_true_text", "ol_bullets_transfer", "ol_bullets_retrained", "nla_recon_probe"):
                c.set_facecolor(plt.cm.Oranges(max(0.0, (v - 0.5) / 0.5) * 0.6) if v > 0.5 else "#FFFFFF")
            else:
                c.set_facecolor(plt.cm.Blues(min(1.0, max(0.0, v)) * 0.6))
    fig.suptitle("Exp 14: what each reconstructor keeps. Probe columns = AUROC (0.5 chance). Gap/statement columns = "
                 "fraction of the true class gap (1 = full).\nOL ruler = the AR Oracle Lens was trained against, read at "
                 "layer 44 in its own output space. NLA ruler = the NLA's AR at layer 42.", fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / "results" / "headline.png", dpi=170, bbox_inches="tight", pad_inches=0.3)
    print(md)


if __name__ == "__main__":
    main()
