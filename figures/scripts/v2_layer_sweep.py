"""v2 talk chart: exp 26, Oracle Lens at layers 20-60, restyled from g5_s8_layer_sweep.py.

Data come from g5_s8_layer_sweep.load() (the same raw exp 26 files); this script only draws.
Writes figures/out/v2_layer_sweep.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g5_s8_layer_sweep as g5  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from v2_common import P, RC, save  # noqa: E402


def series(d: dict) -> dict:
    L = g5.LAYERS
    car = np.array([[d[r]["carries"][x] for x in L] for r in g5.CORE])
    prb = np.array([[d[r]["readable"][x] for x in L] for r in g5.CORE])
    names = np.array([max(d[r]["names_k"][x] / d[r]["n"] for r in g5.STRUCT) for x in L])
    return {"x": np.array(L), "car_med": np.median(car, 0), "car_lo": car.min(0), "car_hi": car.max(0),
            "prb_med": np.median(prb, 0), "names_max": names}


def draw(s: dict) -> None:
    x = s["x"]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.6, 2.9))
        ax.axhline(0.5, color="#CFC9BD", lw=1, ls=":", zorder=0)
        ax.text(60.5, 0.5, "chance", fontsize=10.5, color=P["muted"], va="center", ha="left")
        ax.fill_between(x, s["car_lo"], s["car_hi"], color=P["ol_pale"], alpha=0.6, lw=0, zorder=1)
        rows = ((s["prb_med"], P["model"], "probe", 0.07), (s["car_med"], P["ol"], "carries", -0.12),
                (s["names_max"], P["ol"], "names (highest row)", 0.07))
        for y, col, lab, dy in rows:
            ls = "--" if lab.startswith("names") else "-"
            ax.plot(x, y, color=col, lw=2.4, ls=ls, zorder=3)
            ax.scatter(x[[0, -1]], y[[0, -1]], s=26, color=col, zorder=4)
            ax.text(x[0], y[0] + dy, f"{y[0]:.2f}", color=col, fontsize=10.5, ha="center", fontweight="bold")
            ax.text(x[-1], y[-1] + dy, f"{y[-1]:.2f}", color=col, fontsize=10.5, ha="center", fontweight="bold")
            ax.text(40, y[len(x) // 2] + (0.045 if lab != "carries" else -0.11), lab, color=col, fontsize=11.5,
                    ha="center")
        ax.set_xticks([20, 32, 42, 52, 60])
        ax.set_ylim(-0.08, 1.12)
        ax.set_xlim(17, 63)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Oracle Lens read layer", fontsize=11, color=P["muted"], labelpad=2)
        save(fig, "v2_layer_sweep")


if __name__ == "__main__":
    s = series(g5.load())
    draw(s)
    for k, v in s.items():
        print(k, np.round(v, 3).tolist())
