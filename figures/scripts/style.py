"""Shared figure style for the talk deck.

Colour roles match deck/lib/theme.js and never change between figures:
NLA vermilion, Oracle Lens violet, J-lens teal, text-only / prompt baselines ochre,
probe / model-internal measures slate, chance and thresholds light grey.
"""
from __future__ import annotations

import json
import math
import re
import struct
import sys
from pathlib import Path

import matplotlib
import matplotlib.text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "out"
MANIFEST = ROOT / "figures" / "manifest"

C = {
    "nla": "#E2553A",
    "ol": "#7A5BC2",
    "jl": "#0F9D8A",
    "text": "#C8921E",
    "probe": "#2F4858",
    "muted": "#5E6E78",
    "line": "#C9D3D9",
    "ink": "#15222C",
    "nla_pale": "#F3B6A8",
    "ol_pale": "#CBBEE8",
    "jl_pale": "#9FD9CF",
    "text_pale": "#E9D3A1",
}
READER_LABEL = {"nla": "NLA", "ol": "Oracle Lens", "jl": "J-lens", "text": "Prompt-only text"}

# 16:9 slide body area. Full-width figure ~ 12 x 5.6 in; half-width ~ 6 x 5.6 in.
FULL = (12.0, 5.6)
HALF = (6.0, 5.6)

# Google Slides deck: 10 x 5.625 in slide with a one-line title above the figure, so the figure area is about
# 9.4 x 4.3 in (aspect ~2.2). Main figures are drawn at this size so that the point sizes below are the sizes the
# audience sees when the PNG is placed 9.4 in wide. MIN_PT is the floor for any text in a main figure.
SLIDE = (9.4, 4.3)
# savefig pads 0.08 in on each side, so a canvas of SLIDE_NET lands at ~9.4 in wide after a tight save.
SLIDE_NET = (9.22, 4.22)
SLIDE_W = 9.4
MIN_PT = 15
ASPECT_RANGE = (2.1, 2.3)
SLIDE_RC = {
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "savefig.pad_inches": 0.08,
}


def slide_style() -> None:
    """Call once at the top of a main-figure script: every default text size becomes >= MIN_PT."""
    plt.rcParams.update(SLIDE_RC)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": C["muted"],
    "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "text.color": C["ink"],
    "axes.grid": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with Wilson 95% interval: (p, lo, hi)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, mid - half), min(1.0, mid + half)


def auroc(scores, labels) -> float:
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_ci(stat, *arrays, n_boot: int = 2000, seed: int = 0, groups=None):
    """Percentile 95% CI of stat(*arrays) over item resamples (or cluster resamples if groups given)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    vals = []
    if groups is None:
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            vals.append(stat(*[a[idx] for a in arrays]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq))
            idx = np.concatenate([members[g] for g in pick])
            vals.append(stat(*[a[idx] for a in arrays]))
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def chance_line(ax, y: float = 0.5, label: str | None = "chance", horizontal: bool = True, fontsize: int | None = None):
    if horizontal:
        ax.axhline(y, color=C["line"], lw=1.5, ls="--", zorder=0)
        if label:
            fs = fontsize or (MIN_PT if plt.rcParams["font.size"] >= MIN_PT else 12)
            ax.text(ax.get_xlim()[1], y, f" {label}", va="center", ha="left", color=C["muted"], fontsize=fs)
    else:
        ax.axvline(y, color=C["line"], lw=1.5, ls="--", zorder=0)


def bar_labels(ax, bars, fmt: str = "{:.2f}", inside: bool = False, fontsize: int | None = None):
    fontsize = fontsize or (MIN_PT if plt.rcParams["font.size"] >= MIN_PT else 13)
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        y = h - 0.02 if inside else h + 0.01
        ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(h), ha="center",
                va="top" if inside else "bottom", fontsize=fontsize, color=C["ink"])


# ---- shared vocabulary -------------------------------------------------------------------------------------
# "states" = the judge's free-readout verdict that the readout states the property; "names" = a regex match.
# Reader labels are always "NLA", "Oracle Lens", "J-lens". Raw row keys never appear in text.
ROW_WORDS = {
    "py_arg": "call argument", "py_string": "inside a Python string", "js_string": "inside a JavaScript string",
    "latex_math": "inside LaTeX math", "json_depth": "JSON nesting depth", "html_depth": "HTML nesting depth",
    "md_heading": "Markdown heading level", "md_list": "list nesting depth", "fmt_listdepth": "list nesting depth",
    "code_lang": "code language", "tb_tablecol": "table column", "tb_paren": "inside parentheses",
    "bk_listidx": "list item index",
}
_KEY = r"(?<![\w/.]){}(?![\w.])"
_TERM_SUBS = [
    (re.compile(r"\b(NLA|OL|Oracle Lens|J-?[Ll]ens|JLens|jlens)\s?@\s?L?(\d{2})\b"), r"\1 at layer \2"),
    (re.compile(r"\s?@\s?L(\d{2})\b"), r" at layer \1"),
    (re.compile(r"(?<![\w/.])L(\d{2})(?![\w.])"), r"layer \1"),
    (re.compile(r"\bOL\b"), "Oracle Lens"),
    (re.compile(r"\bJ-Lens\b|\bJLens\b|\bJlens\b|\bJ lens\b|(?<![\w/.])jlens(?![\w/.])"), "J-lens"),
    (re.compile(r"\biff\b"), "exactly when"),
] + [(re.compile(_KEY.format(k)), v) for k, v in ROW_WORDS.items()]
_TERM_LINT = [p for p, _ in _TERM_SUBS] + [re.compile(r"\bFVE\b")]


def _fve(text: str) -> str:
    """'FVE' without expansion -> 'reconstruction (FVE)' on first use in a text, plain 'reconstruction' after."""
    keep = "\0RFVE\0"
    t = re.sub(r"reconstruction \(FVE\)", keep, text)
    t = re.sub(r"\breconstructions?\s+FVE\b", "FVE", t)
    first = keep not in t
    def rep(m):
        nonlocal first
        out = keep if first else "reconstruction"
        first = False
        return out
    t = re.sub(r"(?<!\()\bFVE\b(?!\))", rep, t)
    return t.replace(keep, "reconstruction (FVE)")


def plain_terms(text: str) -> str:
    """Rewrite record shorthand (OL, L42, raw row keys, iff, bare FVE) into the deck's vocabulary."""
    if not isinstance(text, str):
        return text
    for pat, rep in _TERM_SUBS:
        text = pat.sub(rep, text)
    return _fve(text)


def term_lint(fig) -> list[str]:
    """Figure texts that still use shorthand the deck forbids (fix these in the script, not here)."""
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not (s and t.get_visible()):
            continue
        bare = re.sub(r"reconstruction\s*\(FVE", "", s)  # 'reconstruction (FVE)' / '(FVE, ...)' is the allowed form
        if any(p.search(s) for p in _TERM_LINT[:-1]) or _TERM_LINT[-1].search(bare):
            bad.append(s)
    return bad


def _layout(fig, png: Path) -> dict:
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    min_pt = min((t.get_fontsize() for t in texts), default=float("nan"))
    with open(png, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    w_in = w / plt.rcParams["savefig.dpi"]
    return {"png_px": [int(w), int(h)], "aspect": round(w / h, 3), "min_font_pt": round(float(min_pt), 1),
            "effective_min_pt_at_9p4in": round(float(min_pt) * SLIDE_W / w_in, 1)}


def save(fig, name: str, *, title: str, caption: str, section: str, role: str,
         sources: list[str], checks: list[dict] | None = None, notes: str = "") -> None:
    """Write out/<name>.png and .svg, and manifest/<name>.json.

    title: the claim, verb-forward, used as the slide title.
    caption: what it shows, how to read it, key technical detail (n, CI method, threshold).
    section: talk section ("S1".."S8", "headline") or "backup:expNN".
    role: "main" or "backup".
    sources: repo-relative paths of the raw files read.
    checks: [{"quantity", "computed", "record", "where", "match": bool}] comparing to RESULTS.md / talk numbers.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    for s in term_lint(fig):
        print(f"[{name}] TERM: figure text uses shorthand: {s!r}", file=sys.stderr)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg")
    layout = _layout(fig, OUT / f"{name}.png")
    plt.close(fig)
    if role == "main":
        lo, hi = ASPECT_RANGE
        if not lo <= layout["aspect"] <= hi:
            print(f"[{name}] LAYOUT: aspect {layout['aspect']} outside {lo}-{hi}", file=sys.stderr)
        if layout["effective_min_pt_at_9p4in"] < MIN_PT - 0.05:
            print(f"[{name}] LAYOUT: smallest text {layout['effective_min_pt_at_9p4in']} pt at 9.4 in", file=sys.stderr)
    checks = [dict(c, quantity=plain_terms(c.get("quantity", ""))) for c in (checks or [])]
    (MANIFEST / f"{name}.json").write_text(json.dumps({
        "name": name, "title": plain_terms(title), "caption": plain_terms(caption), "section": section,
        "role": role, "sources": sources, "checks": checks, "notes": plain_terms(notes), "layout": layout,
    }, indent=2))
