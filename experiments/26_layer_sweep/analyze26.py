"""Tables and figures from results/sweep26_{row}.json (laptop; matplotlib, no seaborn).
  results/table26.md       one markdown table per measure (rows x layers): readable, ol_prompted, carries, names
  results/fig26_{row}.png  per row: AUROC vs layer (probe, Oracle Lens prompted, Oracle Lens readout through the
                           critic) with the 0.70 and 0.80 lines and 42 marked; a strip below with the strict naming rate
  results/fig26.png        the eight rows in a 2 x 4 grid
    .venv/bin/python analyze26.py [results_dir]      (results_dir defaults to ./results; used by the dry run)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from lib26 import LAYER, LINES, MEASURES, ROWS, cell, table  # noqa: E402

TITLES = {"readable": "Readable: grouped 5-fold probe AUROC on the true activations at the layer",
          "ol_prompted": "Oracle Lens prompted AUROC at the layer (exp 22's frozen phrasings; max of the mean of three and the fixed label)",
          "carries": "Oracle Lens k=1 readout at the layer, rebuilt by the NLA critic, scored by the frozen probe against the true layer-42 activations",
          "names": "Strict naming rate: share of the layer's readouts in which the row's state regex fires"}
SERIES = (("readable", "probe (readable)", "#2a78d6", "o"),
          ("ol_prompted", "Oracle Lens prompted", "#eb6834", "s"),
          ("carries", "Oracle Lens readout through the critic", "#1baf7a", "^"))
NAMES_COLOR = "#52514e"
GRID = "#d8d7d2"
YLIM = (0.4, 1.0)


def load_rows(res: Path) -> list[dict]:
    rows = []
    for row in ROWS:
        p = res / f"sweep26_{row}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run run26.sh (stage recon) first")
        rows.append(json.loads(p.read_text()))
    return rows


def _series(r: dict, measure: str) -> tuple[list[int], list[float]]:
    layers = sorted(int(L) for L in r["layers"])
    pairs = [(L, cell(r.get(measure, {}), L)) for L in layers]
    pairs = [(L, v) for L, v in pairs if v is not None]
    return [L for L, _ in pairs], [v for _, v in pairs]


def draw_row(ax_main, ax_names, r: dict, legend: bool, label_size: int, xlabel: bool = True,  # noqa: ANN001
             hide_ticklabels: tuple[int, ...] = ()) -> None:
    layers = sorted(int(L) for L in r["layers"])
    for y in LINES:
        ax_main.axhline(y, color=GRID, lw=1, ls="--", zorder=1)
    for ax in (ax_main, ax_names):
        ax.axvline(LAYER, color=GRID, lw=1, zorder=1)
    for key, label, color, marker in SERIES:
        xs, ys = _series(r, key)
        if xs:
            ax_main.plot(xs, ys, color=color, lw=2, marker=marker, ms=5, label=label, zorder=3)
    xs, ys = _series(r, "names")
    if xs:
        ax_names.plot(xs, ys, color=NAMES_COLOR, lw=1.5, ls=":", marker=".", ms=5, label="strict naming rate", zorder=3)
    ax_main.set_ylim(*YLIM)
    ax_main.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_main.set_title(r["feature"], fontsize=label_size + 2, loc="left")
    ax_main.tick_params(labelbottom=False, labelsize=label_size)
    ax_names.set_ylim(-0.05, 1.05)
    ax_names.set_yticks([0, 0.5, 1])
    ax_names.set_xticks(layers)
    ax_names.set_xticklabels(["" if L in hide_ticklabels else f"{L}" for L in layers], fontsize=label_size)
    for t in ax_names.get_xticklabels():
        if t.get_text() == str(LAYER):
            t.set_fontweight("bold")
    ax_names.tick_params(labelsize=label_size)
    if xlabel:
        ax_names.set_xlabel("layer (block output; 42 = the primary read layer)", fontsize=label_size)
    for ax in (ax_main, ax_names):
        ax.set_xlim(layers[0] - 1.5, layers[-1] + 1.5)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if legend:
        ax_main.set_ylabel("AUROC", fontsize=label_size)
        ax_names.set_ylabel("names", fontsize=label_size)
        ax_main.legend(loc="lower right", fontsize=label_size - 1, frameon=False)
        ax_names.legend(loc="upper left", fontsize=label_size - 1, frameon=False)


def fig_row(r: dict, out: Path) -> None:
    fig = plt.figure(figsize=(7.5, 5.2), dpi=150)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08, figure=fig)
    ax_main = fig.add_subplot(gs[0])
    ax_names = fig.add_subplot(gs[1], sharex=ax_main)
    draw_row(ax_main, ax_names, r, legend=True, label_size=10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_grid(rows: list[dict], out: Path) -> None:
    ncol = 4
    nrow = (len(rows) + ncol - 1) // ncol
    fig = plt.figure(figsize=(5.0 * ncol, 4.3 * nrow), dpi=150)
    outer = GridSpec(nrow, ncol, hspace=0.3, wspace=0.22, figure=fig)
    for i, r in enumerate(rows):
        g, c = divmod(i, ncol)
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[g, c], height_ratios=[3, 1], hspace=0.08)
        ax_main = fig.add_subplot(inner[0])
        ax_names = fig.add_subplot(inner[1], sharex=ax_main)
        draw_row(ax_main, ax_names, r, legend=False, label_size=9, xlabel=(g == nrow - 1), hide_ticklabels=(40, 44))
        if c == 0:
            ax_main.set_ylabel("AUROC", fontsize=10)
            ax_names.set_ylabel("names", fontsize=10)
    handles = [plt.Line2D([], [], color=col, lw=2, marker=m, ms=5, label=lab) for _, lab, col, m in SERIES]
    handles.append(plt.Line2D([], [], color=NAMES_COLOR, lw=1.5, ls=":", marker=".", ms=5, label="strict naming rate (strip, 0 to 1)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Exp 26: AUROC by layer on the exp 22 public bank (dashed lines 0.70 and 0.80; 42 marked)", fontsize=13, y=0.995)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def render(res: Path) -> str:
    rows = load_rows(res)
    md = []
    for m in MEASURES:
        md += [f"### {m}", "", TITLES[m], "", table(rows, m), ""]
    text = "\n".join(md)
    (res / "table26.md").write_text(text)
    for r in rows:
        fig_row(r, res / f"fig26_{r['feature']}.png")
    fig_grid(rows, res / "fig26.png")
    return text


def main() -> None:
    res = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
    print(render(res))


if __name__ == "__main__":
    main()
