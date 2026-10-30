"""Drawing primitives for the g6 talk diagrams (boxes, arrows, canvas) on top of style.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from style import C, ROOT, SLIDE, SLIDE_RC, slide_style  # noqa: E402,F401

slide_style()  # every g6 diagram is a main slide figure

WHITE = "#FFFFFF"
SOFT = "#F3F6F8"  # neutral panel fill


# Slide canvas: the saved PNG adds savefig.pad_inches on each side, so the drawing area is the slide figure area
# minus that pad. Then the PNG is ~9.4 in wide and a font's nominal size is the size seen on the slide.
PAD = SLIDE_RC["savefig.pad_inches"]
SW, SH = SLIDE[0] - 2 * PAD - 0.1, SLIDE[1] - 2 * PAD - 0.05  # 0.1 in spare for line widths at the edge
FS = 15  # floor for every diagram text (style.MIN_PT)


XMAX, YMAX = 92.4, 42.0  # drawing frame for the slide canvas (about 10 units per inch)


def canvas(w: float = SW, h: float = SH, xmax: float | None = XMAX, ymax: float | None = YMAX):
    """Figure with one invisible axes in a 0..xmax x 0..ymax coordinate frame (10 units per inch by default; the slide frame is XMAX x YMAX). A white
    background patch spans the frame so the tight bounding box keeps the full slide aspect."""
    xmax = w * 10 if xmax is None else xmax
    ymax = h * 10 if ymax is None else ymax
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), xmax, ymax, fc=WHITE, ec="none", zorder=-10))
    return fig, ax


def box(ax, x, y, w, h, text="", *, fc=WHITE, ec=None, lw=1.6, fs=FS, color=None, weight="normal",
        ha="center", va="center", pad=0.6, radius=1.2, ls="-", z=2, linespacing=1.25, alpha=1.0):
    """Rounded box with lower-left corner (x, y); text centred unless ha/va say otherwise."""
    ec = ec or C["muted"]
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha)
    ax.add_patch(p)
    if text:
        tx = {"center": x + w / 2, "left": x + pad, "right": x + w - pad}[ha]
        ty = {"center": y + h / 2, "top": y + h - pad, "bottom": y + pad}[va]
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs, color=color or C["ink"], weight=weight,
                zorder=z + 1, linespacing=linespacing)
    return p


def arrow(ax, p0, p1, *, color=None, lw=2.0, style="-|>", ms=16, ls="-", rad=0.0, z=3, shrink=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color or C["muted"], lw=lw,
                        ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=shrink, shrinkB=shrink)
    ax.add_patch(a)
    return a


def text(ax, x, y, s, *, fs=FS, color=None, ha="center", va="center", weight="normal", style="normal",
         z=5, linespacing=1.25, bg=None):
    kw = {}
    if bg:
        kw["bbox"] = dict(boxstyle="round,pad=0.2", fc=bg, ec="none")
    return ax.text(x, y, s, fontsize=fs, color=color or C["ink"], ha=ha, va=va, weight=weight,
                   style=style, zorder=z, linespacing=linespacing, **kw)
