"""v2 talk chart: exp 29, the NLA along its training trajectory, restyled from g5_s8_trajectory.py.

Data come from g5_s8_trajectory.load() (the same raw exp 29 files); this script only draws.
Writes figures/out/v2_trajectory.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g5_s8_trajectory as g5  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from v2_common import P, RC, save  # noqa: E402


def draw(d: dict) -> None:
    ck = g5.CKPTS
    x = np.array([-2.0, -0.6] + [1.0 + i for i in range(8)])
    st = np.array([[d["states"][c][r]["p"] for c in ck] for r in g5.STRUCT])
    med, lo, hi = np.median(st, 0), st.min(0), st.max(0)
    code = np.array([d["states"][c]["code_lang"]["p"] for c in ck])
    fve = np.array([d["fve"][c]["py_string"] for c in ck])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.2, 3.25))
        ax.axvspan(-2.55, -1.45, color="#ECE6DA", lw=0, zorder=0)
        ax.axvline(0.2, color="#CFC9BD", lw=1, ls=":", zorder=0)
        ax.text(0.25, 1.02, "RL starts", fontsize=11, color=P["muted"], va="bottom")
        ax.fill_between(x, lo, hi, color=P["nla_pale"], alpha=0.45, lw=0, zorder=1)
        series = ((fve, P["model"], "reconstruction (FVE),\nPython strings", ((0, 0.06), (0.32, -0.03), (0, -0.17))),
                  (code, "#8C8578", "states code language\n(control)", ((0, 0.06), (0, 0.06), (0, 0.06))),
                  (med, P["nla"], "states structure\n(median of 7 rows)", ((0, 0.06), (0, -0.17), (0, 0.06))))
        for y, col, lab, offs in series:
            ax.plot(x, y, color=col, lw=2.6, zorder=3)
            ax.scatter(x[[0, 1, -1]], y[[0, 1, -1]], s=36, color=col, zorder=4)
            ax.text(x[-1] + 0.3, y[-1], lab, color=col, fontsize=11.5, va="center", ha="left", linespacing=1.0)
            for i, (dx, dy) in zip((0, 1, 9), offs):
                ax.text(x[i] + dx, y[i] + dy, f"{y[i]:.2f}", color=col, fontsize=11, ha="left" if dx else "center",
                        va="bottom",
                        fontweight="bold")
        ax.text(4.5, hi[2:].max() + 0.03, "range of the 7 structural rows", fontsize=10.5, color=P["nla"],
                ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(["text\nonly", "warm\nstart", "100", "", "300", "", "500", "", "700", "800"])
        ax.set_xlim(-2.55, 8.6)
        ax.set_ylim(-0.1, 1.08)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("RL step", fontsize=11.5, color=P["muted"], labelpad=2)
        save(fig, "v2_trajectory")


if __name__ == "__main__":
    data = g5.load()
    draw(data)
    st = [[data["states"][c][r]["p"] for c in g5.CKPTS] for r in g5.STRUCT]
    print("struct median", np.round(np.median(st, 0), 3).tolist())
    print("struct min/max", np.round(np.min(st, 0), 3).tolist(), np.round(np.max(st, 0), 3).tolist())
    print("code", [round(data["states"][c]["code_lang"]["p"], 3) for c in g5.CKPTS])
    print("fve", [round(data["fve"][c]["py_string"], 3) for c in g5.CKPTS])
