"""Shared palette and styling for the v2 talk line charts (cream background, no frame, direct labels)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
BG = "#F6F4EE"
P = {"nla": "#C2552E", "ol": "#6A5AA6", "jl": "#2F7A74", "base": "#A7A196", "track": "#E7E2D8",
     "model": "#34424C", "ink": "#1D232A", "muted": "#8A8478", "nla_pale": "#E6B8A5", "ol_pale": "#CFC8E6"}

RC = {"font.family": "DejaVu Sans", "font.size": 13, "axes.facecolor": BG, "figure.facecolor": BG,
      "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
      "axes.spines.bottom": True, "axes.edgecolor": "#CFC9BD", "xtick.color": P["muted"], "ytick.color": P["muted"],
      "xtick.labelsize": 12, "ytick.labelsize": 12, "text.color": P["ink"], "axes.grid": False,
      "savefig.dpi": 250, "savefig.bbox": "tight", "savefig.pad_inches": 0.05}


def save(fig, name: str) -> Path:
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
