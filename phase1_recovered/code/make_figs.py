#!/usr/bin/env python3
"""Build the paper figures (PDF + PNG@200dpi) plus figN_data.csv.

Re-run:  python3 make_figs.py
Reads only from ../paper_data/ (never writes there).
Every plotted number is recomputed at the end via an independent pure-python
path and compared against what was drawn; a short table is printed and the
script exits non-zero on any mismatch.

Figure text never uses en/em dashes (author constraint): ranges are written
with a plain hyphen, e.g. "inject 12-14".
"""
import csv
import json
import os
import statistics
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "paper_data")
OUT = HERE

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "lines.linewidth": 1.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 200,
})

# Okabe-Ito (colorblind-safe).  Fixed slot order; categories always map the
# same way regardless of which figure they appear in.
BLUE, ORANGE, GREEN, PINK, VERMILLION, SKY = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
CATS = ["countries", "animals", "foods", "colors", "emotions"]
CAT_COLOR = dict(zip(CATS, [BLUE, ORANGE, GREEN, PINK, VERMILLION]))
CAT_MARKER = dict(zip(CATS, ["o", "s", "^", "D", "v"]))       # secondary encoding
CAT_LS = dict(zip(CATS, ["-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5))]))
GREY = "#7f7f7f"
LANDED_THRESH = 25
NAMED_THRESH = 5
INJ_BAND = (12, 14)
READ_LAYER = 22
GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
KS = [1, 3, 5, 10, 25]
GAMMAS_B = [0.2, 0.4, 0.6, 0.8]
LABEL_COLOR = "#222222"

SINGLE_W, FULL_W = 3.3, 6.5


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def shade_band(ax):
    ax.axvspan(INJ_BAND[0] - 0.5, INJ_BAND[1] + 0.5, color=GREY, alpha=0.18,
               lw=0, zorder=0)
    ax.axvline(READ_LAYER, color="k", lw=0.8, ls="--", zorder=1)


def panel_label(ax, s):
    ax.text(-0.22, 1.0, s, transform=ax.transAxes, fontweight="bold")


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
def load_jsonl(name):
    with open(os.path.join(DATA, name)) as f:
        return [json.loads(l) for l in f if l.strip()]

main = load_jsonl("res_results_main.jsonl")          # gamma 0 .. 0.4
ext = load_jsonl("res_results_ext.jsonl")            # gamma 0.6, 0.8
trials = main + ext
with open(os.path.join(DATA, "res_check3_band.json")) as f:
    band = json.load(f)
with open(os.path.join(DATA, "res_layer_scan.json")) as f:
    scan = json.load(f)

for r in trials:
    r["landed"] = r["lens_rank"] <= LANDED_THRESH
    r["named"] = r["rank_c"] < NAMED_THRESH
    r["det"] = (r["p_yes"] > 0.5) and r["named"]

CONCEPTS = sorted({r["concept"] for r in main if r["kind"] == "concept"})
assert len(CONCEPTS) == 50
assert {r["concept"] for r in ext} == set(CONCEPTS)

# per (kind, gamma) -> concept -> list of trials
cell = defaultdict(lambda: defaultdict(list))
for r in trials:
    cell[(r["kind"], r["gamma"])][r["concept"]].append(r)
for key, d in cell.items():
    assert len(d) == 50 and all(len(v) == 20 for v in d.values()), key

# ----------------------------------------------------------------------------
# Concept-clustered bootstrap (shared by fig 1a / 1b)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(20260903)
NBOOT = 1000
BOOT_IDX = rng.integers(0, len(CONCEPTS), size=(NBOOT, len(CONCEPTS)))

def concept_means(kind, gamma, fn):
    d = cell[(kind, gamma)]
    return np.array([np.mean([fn(r) for r in d[c]]) for c in CONCEPTS])

def boot_ci(cm):
    """Resample the 50 concepts with replacement.  Balanced design (20 passages
    per concept) so the mean of concept means equals the pooled trial mean."""
    boots = cm[BOOT_IDX].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def stat(kind, gamma, fn):
    cm = concept_means(kind, gamma, fn)
    lo, hi = boot_ci(cm)
    n = sum(len(v) for v in cell[(kind, gamma)].values())
    return float(cm.mean()), lo, hi, n

# ============================================================================
# FIG 1: (a) dose-response over gamma, (b) naming rate vs cutoff k
# ============================================================================
METRICS1 = {
    "p_yes": lambda r: r["p_yes"],
    "det": lambda r: float(r["det"]),
    "landed": lambda r: float(r["landed"]),
}
SERIES1 = [
    ("concept_p_yes", "concept", "p_yes"),
    ("concept_det", "concept", "det"),
    ("random_p_yes", "random", "p_yes"),
    ("random_det", "random", "det"),
    ("concept_landed", "concept", "landed"),
    ("random_landed", "random", "landed"),   # CSV only, not plotted
]
fig1a = {}
for name, kind, met in SERIES1:
    fig1a[name] = [(g,) + stat(kind, g, METRICS1[met]) for g in GAMMAS
                   if (kind, g) in cell]
clean_p_yes = fig1a["concept_p_yes"][0][1]
assert fig1a["concept_p_yes"][0][0] == 0.0

# panel b: fraction of trials with rank_c < k
fig1b = {}   # (kind, gamma) -> list of (k, mean, lo, hi, n)
for kind, g in [("concept", g) for g in GAMMAS_B] + [("random", 0.8)]:
    fig1b[(kind, g)] = [(k,) + stat(kind, g, lambda r, k=k: float(r["rank_c"] < k))
                        for k in KS]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(FULL_W, 2.5),
                               gridspec_kw=dict(width_ratios=[1.25, 1.0]))
# --- (a)
ax = axA
ax.axhline(clean_p_yes, color=GREY, lw=0.9, ls="-", zorder=1)
ax.text(0.02, clean_p_yes + 0.015, f"clean P(yes) = {clean_p_yes:.2f}",
        color=GREY, fontsize=6.5, ha="left", va="bottom")
STYLE1 = {
    "concept_p_yes": dict(color=BLUE, ls="-", marker="o", label="concept: P(yes)"),
    "concept_det": dict(color=VERMILLION, ls="-", marker="o", label="concept: detection"),
    "random_p_yes": dict(color=BLUE, ls="--", marker="s", label="random: P(yes)"),
    "random_det": dict(color=VERMILLION, ls="--", marker="s", label="random: detection"),
    "concept_landed": dict(color="k", ls=":", marker="^", label="landed (concept)"),
}
for name, st in STYLE1.items():
    pts = fig1a[name]
    g = [p[0] for p in pts]; m = [p[1] for p in pts]
    lo = [p[2] for p in pts]; hi = [p[3] for p in pts]
    ax.fill_between(g, lo, hi, color=st["color"], alpha=0.15, lw=0, zorder=2)
    ax.plot(g, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.2,
            mfc="white" if name.startswith("random") else st["color"],
            mew=1.0, label=st["label"], zorder=3)
ax.set_xlabel(r"injection strength $\gamma$")
ax.set_ylabel("fraction / probability")
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax.xaxis.set_minor_locator(FixedLocator([0.05, 0.1]))
ax.set_xlim(-0.015, 0.815)
ax.set_ylim(-0.02, 1.02)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.10), frameon=False, ncol=1,
          handlelength=2.4, borderaxespad=0.2, labelspacing=0.3)
panel_label(ax, "(a)")

# --- (b)
ax = axB
GAMMA_COLOR = {0.2: "#9ecae1", 0.4: "#4292c6", 0.6: "#08519c", 0.8: "#08306b"}
GAMMA_MARKER = {0.2: "o", 0.4: "s", 0.6: "^", 0.8: "D"}
pts = fig1b[("random", 0.8)]
ax.fill_between([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts],
                color=GREY, alpha=0.2, lw=0, zorder=2)
ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GREY, ls="--", marker="x",
        ms=3.2, label=r"random, $\gamma$ = 0.8", zorder=3)
for g in GAMMAS_B:
    pts = fig1b[("concept", g)]
    ax.fill_between([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts],
                    color=GAMMA_COLOR[g], alpha=0.15, lw=0, zorder=2)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GAMMA_COLOR[g], ls="-",
            marker=GAMMA_MARKER[g], ms=3.2, label=rf"concept, $\gamma$ = {g:g}",
            zorder=3)
ax.set_xscale("log")
ax.set_xticks(KS)
ax.set_xticklabels([str(k) for k in KS])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.set_xlim(0.85, 30)
ax.set_ylim(-0.02, 1.02)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("naming cutoff $k$")
ax.set_ylabel("fraction named (rank$_c$ < $k$)")
ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
ax.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.2,
          labelspacing=0.3)
panel_label(ax, "(b)")
fig.subplots_adjust(wspace=0.32)
save(fig, "fig1")

rows = []
for name, _, _ in SERIES1:
    for g, m, lo, hi, n in fig1a[name]:
        rows.append(["a", name, g, "", f"{m:.6f}", f"{lo:.6f}", f"{hi:.6f}", n,
                     len(CONCEPTS), "yes" if name in STYLE1 else "no"])
rows.append(["a", "clean_p_yes_reference_line", 0.0, "", f"{clean_p_yes:.6f}", "", "",
             fig1a["concept_p_yes"][0][4], len(CONCEPTS), "yes"])
for (kind, g), pts in fig1b.items():
    for k, m, lo, hi, n in pts:
        rows.append(["b", f"{kind}_named_k", g, k, f"{m:.6f}", f"{lo:.6f}",
                     f"{hi:.6f}", n, len(CONCEPTS), "yes"])
write_csv("fig1_data.csv",
          ["panel", "series", "gamma", "k", "mean", "ci95_lo", "ci95_hi", "n_trials",
           "n_concepts_bootstrap", "plotted"], rows)

# ============================================================================
# FIG 2: band localisation (accuracy + excess kurtosis vs source layer)
# ============================================================================
layers = np.arange(35)
acc_j, acc_l = np.array(band["acc"]["j"]), np.array(band["acc"]["logit"])
kur_j, kur_l = np.array(band["kur"]["j"]), np.array(band["kur"]["logit"])
assert all(len(a) == 35 for a in (acc_j, acc_l, kur_j, kur_l))
BAND_TXT = f"inject\n{INJ_BAND[0]}-{INJ_BAND[1]}"      # plain hyphen (author constraint)

fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.3))
for ax, (yj, yl, ylab) in zip(axes, [(acc_j, acc_l, "top-1 next-token accuracy"),
                                     (kur_j, kur_l, "excess kurtosis")]):
    shade_band(ax)
    ax.plot(layers, yl, color=ORANGE, ls="--", marker="s", ms=2.2, mfc="white",
            mew=0.8, label="logit lens", zorder=3)
    ax.plot(layers, yj, color=BLUE, ls="-", marker="o", ms=2.2,
            label="J-lens", zorder=4)
    ax.set_xlabel("source layer")
    ax.set_ylabel(ylab)
    ax.set_xticks(np.arange(0, 35, 5))
    ax.set_xlim(-0.7, 34.7)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    ax.set_ylim(0, max(yj.max(), yl.max()) * 1.08)
    top = ax.get_ylim()[1] * 0.97
    ax.text(INJ_BAND[0] + 1, top, BAND_TXT, ha="center", va="top", fontsize=6.5,
            color="#444444")
    ax.text(READ_LAYER + 0.6, top, f"read\n{READ_LAYER}", ha="left", va="top",
            fontsize=6.5, color="#444444")
axes[0].legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.88))
axes[1].legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.85))
panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
fig.subplots_adjust(wspace=0.32)
save(fig, "fig2")

write_csv("fig2_data.csv",
          ["layer", "acc_jlens", "acc_logitlens", "kurtosis_jlens", "kurtosis_logitlens",
           "in_injection_band", "is_read_layer"],
          [[int(l), repr(float(acc_j[l])), repr(float(acc_l[l])), repr(float(kur_j[l])),
            repr(float(kur_l[l])), int(INJ_BAND[0] <= l <= INJ_BAND[1]),
            int(l == READ_LAYER)] for l in layers])

# ============================================================================
# Per-concept summaries at gamma = 0.4 (figs 3 and 5)
# ============================================================================
pc = {}
for c in CONCEPTS:
    rs = cell[("concept", 0.4)][c]
    assert len(rs) == 20
    pc[c] = dict(category=rs[0]["category"], n=len(rs),
                 n_landed=sum(r["landed"] for r in rs),
                 n_named=sum(r["named"] for r in rs),
                 med_rank=float(np.median([r["lens_rank"] for r in rs])))
    pc[c]["x"] = pc[c]["n_landed"] / pc[c]["n"]
    pc[c]["y"] = pc[c]["n_named"] / pc[c]["n"]


def annotate(ax, c, xy, spec):
    dx, dy, ha, va = spec
    ax.annotate(c, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=6.5, color=LABEL_COLOR, zorder=5,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5,
                                shrinkA=0, shrinkB=2.5))

# ============================================================================
# FIG 3: per-concept scatter at gamma = 0.4 (landed vs named)
# ============================================================================
# Points sit on a 1/20 grid and many coincide (16 concepts at exactly (0,0)).
# Small deterministic jitter (+-0.012, below the 0.05 grid step) for
# visibility only; the CSV carries both true and jittered coordinates.
jrng = np.random.default_rng(3)
JIT = 0.012
for c in CONCEPTS:
    pc[c]["xj"] = pc[c]["x"] + jrng.uniform(-JIT, JIT)
    pc[c]["yj"] = pc[c]["y"] + jrng.uniform(-JIT, JIT)

LABELS3 = {   # concept -> (dx_pt, dy_pt, ha, va)
    "purple":    ( 3, -11, "right", "center"),
    "Brazil":    (-9,  9, "right", "center"),
    "Canada":    (-9, 22, "right", "center"),
    "navy":      (-6, -10, "right", "center"),
    "France":    (-8,  0, "right", "center"),
    "whale":     (-3, -8, "right", "top"),
    "spider":    ( 7,  0, "left", "center"),
    "chocolate": (-7, -3, "right", "top"),
    "boredom":   ( 7,  0, "left", "center"),
    "bread":     ( 7,  0, "left", "center"),
    "guilt":     ( 7,  0, "left", "center"),
}
REQUESTED3 = ["purple", "boredom", "whale", "spider", "chocolate", "Brazil",
              "Canada", "navy", "France", "guilt", "bread"]
assert set(REQUESTED3) == set(LABELS3)

fig, ax = plt.subplots(figsize=(SINGLE_W, 3.15))
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.8, zorder=1)
ax.text(0.27, 0.29, "landed = named", color=GREY, fontsize=6, rotation=45,
        ha="left", va="bottom", rotation_mode="anchor")
for cat in CATS:
    cs = [c for c in CONCEPTS if pc[c]["category"] == cat]
    ax.scatter([pc[c]["xj"] for c in cs], [pc[c]["yj"] for c in cs],
               s=22, marker=CAT_MARKER[cat], color=CAT_COLOR[cat],
               edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3, label=cat)
for c, spec in LABELS3.items():
    annotate(ax, c, (pc[c]["xj"], pc[c]["yj"]), spec)
n_origin = sum(1 for c in CONCEPTS if pc[c]["x"] == 0 and pc[c]["y"] == 0)
ax.annotate(f"{n_origin} concepts at (0, 0)", (0, 0), xytext=(14, -12),
            textcoords="offset points", ha="left", va="center", fontsize=6,
            color="#555555", arrowprops=dict(arrowstyle="-", color="#555555",
                                             lw=0.5, shrinkA=0, shrinkB=4))
ax.set_xlabel(f"fraction of passages landed (lens rank ≤ {LANDED_THRESH})")
ax.set_ylabel(f"fraction of passages named (rank$_c$ < {NAMED_THRESH})")
ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_aspect("equal")
ax.grid(color="#e5e5e5", lw=0.5, zorder=0)
ax.legend(loc="center", bbox_to_anchor=(0.36, 0.62), frameon=False,
          handletextpad=0.3, borderpad=0.2, labelspacing=0.35)
save(fig, "fig3")

write_csv("fig3_data.csv",
          ["concept", "category", "gamma", "n_passages", "n_landed", "n_named",
           "frac_landed", "frac_named", "plotted_x_jittered", "plotted_y_jittered",
           "labeled"],
          [[c, pc[c]["category"], 0.4, pc[c]["n"], pc[c]["n_landed"],
            pc[c]["n_named"], f"{pc[c]['x']:.4f}", f"{pc[c]['y']:.4f}",
            f"{pc[c]['xj']:.5f}", f"{pc[c]['yj']:.5f}",
            "yes" if c in LABELS3 else "no"] for c in CONCEPTS])

# ============================================================================
# FIG 4: layer scan, median lens rank per category vs source layer
# ============================================================================
fig4 = {}
for cat in CATS:
    R = np.array([d["ranks"] for d in scan if d["category"] == cat], dtype=float)
    assert R.shape == (50, 35)
    fig4[cat] = np.median(R, axis=0)

fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))
shade_band(ax)
ax.axhline(LANDED_THRESH, color=GREY, lw=0.8, ls="--", zorder=1)
ax.text(34.3, LANDED_THRESH / 1.25, f"landed (rank ≤ {LANDED_THRESH})",
        color="#555555", fontsize=6, ha="right", va="top")
for cat in CATS:
    ax.plot(layers, np.maximum(fig4[cat], 1), color=CAT_COLOR[cat], ls=CAT_LS[cat],
            marker=CAT_MARKER[cat], ms=2.0, mew=0.6, label=cat, zorder=3)
ax.set_yscale("log")
ax.set_ylim(1, 3e5)
ax.set_xlim(-0.7, 34.7)
ax.set_xticks(np.arange(0, 35, 5))
ax.set_xlabel("source layer")
ax.set_ylabel("median lens rank ($\\gamma$ = 0.4)")
ax.text(INJ_BAND[0] + 1, 1.6e5, "inject", ha="center", va="center", fontsize=6.5,
        color="#444444")
ax.text(READ_LAYER + 0.6, 1.6e5, "read", ha="left", va="center", fontsize=6.5,
        color="#444444")
ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
ax.legend(loc="lower left", frameon=False, handlelength=2.6, borderaxespad=0.3,
          labelspacing=0.3)
save(fig, "fig4")

write_csv("fig4_data.csv",
          ["layer", "category", "median_lens_rank", "n_trials", "gamma"],
          [[int(l), cat, repr(float(fig4[cat][l])), 50, 0.4]
           for cat in CATS for l in layers])

# ============================================================================
# FIG 5: per-concept median lens rank (layer 22, gamma 0.4) vs naming rate,
#        with per-category least-squares fits on log10(rank + 1)
# ============================================================================
# lens_rank is 0-indexed (France has median 0), so the log axis shows rank + 1.
for c in CONCEPTS:
    pc[c]["x5"] = pc[c]["med_rank"] + 1.0
fig5_fit = {}
for cat in CATS:
    cs = [c for c in CONCEPTS if pc[c]["category"] == cat]
    lx = np.log10([pc[c]["x5"] for c in cs]); yy = np.array([pc[c]["y"] for c in cs])
    slope, intercept = np.polyfit(lx, yy, 1)
    r = np.corrcoef(lx, yy)[0, 1]
    fig5_fit[cat] = dict(slope=float(slope), intercept=float(intercept), r=float(r),
                         xmin=float(10 ** lx.min()), xmax=float(10 ** lx.max()), n=len(cs))

LABELS5 = {
    "purple":    ( 6, -9, "left", "center"),
    "navy":      ( 6,  9, "left", "center"),
    "Brazil":    (-6, -9, "right", "center"),
    "Canada":    ( 0,  8, "center", "bottom"),
    "France":    ( 7,  0, "left", "center"),
    "whale":     (-7,  0, "right", "center"),
    "spider":    ( 7,  0, "left", "center"),
    "chocolate": ( 7,  0, "left", "center"),
    "boredom":   ( 7,  0, "left", "center"),
    "guilt":     ( 7,  0, "left", "center"),
}
REQUESTED5 = ["purple", "boredom", "whale", "spider", "chocolate", "Brazil",
              "Canada", "France", "navy", "guilt"]
assert set(REQUESTED5) == set(LABELS5)

fig, ax = plt.subplots(figsize=(SINGLE_W, 2.9))
ax.axvline(LANDED_THRESH + 1, color=GREY, lw=0.8, ls="--", zorder=1)
ax.text(LANDED_THRESH + 1, 1.05, f"rank {LANDED_THRESH}", color="#555555",
        fontsize=6, ha="center", va="bottom")
for cat in CATS:
    cs = [c for c in CONCEPTS if pc[c]["category"] == cat]
    f = fig5_fit[cat]
    xx = np.array([f["xmin"], f["xmax"]])
    ax.plot(xx, f["intercept"] + f["slope"] * np.log10(xx), color=CAT_COLOR[cat],
            ls=CAT_LS[cat], lw=1.1, alpha=0.9, zorder=2)
    ax.scatter([pc[c]["x5"] for c in cs], [pc[c]["y"] for c in cs], s=22,
               marker=CAT_MARKER[cat], color=CAT_COLOR[cat], edgecolor="white",
               linewidth=0.5, alpha=0.9, zorder=3, label=cat)
for c, spec in LABELS5.items():
    annotate(ax, c, (pc[c]["x5"], pc[c]["y"]), spec)
ax.set_xscale("log")
ax.set_xlim(0.8, 4000)
ax.set_ylim(-0.06, 1.06)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("median lens rank + 1 at layer 22 ($\\gamma$ = 0.4)")
ax.set_ylabel(f"fraction of passages named (rank$_c$ < {NAMED_THRESH})")
ax.grid(color="#e5e5e5", lw=0.5, zorder=0)
ax.legend(loc="upper right", frameon=False, handletextpad=0.3, borderpad=0.2,
          labelspacing=0.3)
save(fig, "fig5")

write_csv("fig5_data.csv",
          ["concept", "category", "gamma", "n_passages", "median_lens_rank_L22",
           "plotted_x_rank_plus_1", "n_named", "frac_named", "labeled",
           "fit_slope_per_log10", "fit_intercept", "fit_pearson_r", "fit_x_range"],
          [[c, pc[c]["category"], 0.4, pc[c]["n"], repr(pc[c]["med_rank"]),
            repr(pc[c]["x5"]), pc[c]["n_named"], f"{pc[c]['y']:.4f}",
            "yes" if c in LABELS5 else "no",
            f"{fig5_fit[pc[c]['category']]['slope']:.6f}",
            f"{fig5_fit[pc[c]['category']]['intercept']:.6f}",
            f"{fig5_fit[pc[c]['category']]['r']:.4f}",
            f"[{fig5_fit[pc[c]['category']]['xmin']:g}, {fig5_fit[pc[c]['category']]['xmax']:g}]"]
           for c in CONCEPTS])

# ============================================================================
# FIG 6: histograms of trial-level P(yes) at gamma = 0.4, concept vs random
# ============================================================================
BINS = np.linspace(0, 1, 21)
p_con = np.array([r["p_yes"] for c in CONCEPTS for r in cell[("concept", 0.4)][c]])
p_ran = np.array([r["p_yes"] for c in CONCEPTS for r in cell[("random", 0.4)][c]])
h_con, _ = np.histogram(p_con, bins=BINS)
h_ran, _ = np.histogram(p_ran, bins=BINS)
m_con, m_ran = float(p_con.mean()), float(p_ran.mean())

fig, ax = plt.subplots(figsize=(SINGLE_W, 2.4))
ax.hist(p_ran, bins=BINS, color=ORANGE, alpha=0.55, edgecolor="white", lw=0.4,
        label=f"random (n={len(p_ran)})", zorder=2)
ax.hist(p_con, bins=BINS, color=BLUE, alpha=0.55, edgecolor="white", lw=0.4,
        label=f"concept (n={len(p_con)})", zorder=3)
ymax = max(h_con.max(), h_ran.max())
ax.axvline(m_ran, color=ORANGE, ls="--", lw=1.0, zorder=4)
ax.axvline(m_con, color=BLUE, ls="-", lw=1.0, zorder=4)
ax.text(m_ran - 0.012, ymax * 1.02, f"mean {m_ran:.2f}", color=VERMILLION,
        fontsize=6.5, ha="right", va="bottom")
ax.text(m_con + 0.012, ymax * 1.02, f"mean {m_con:.2f}", color=BLUE,
        fontsize=6.5, ha="left", va="bottom")
ax.set_xlim(0, 1)
ax.set_ylim(0, ymax * 1.18)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("trial-level P(yes) at $\\gamma$ = 0.4")
ax.set_ylabel("number of trials")
ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
ax.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.5, 0.92))
save(fig, "fig6")

write_csv("fig6_data.csv",
          ["bin_lo", "bin_hi", "count_concept", "count_random", "mean_concept",
           "mean_random", "gamma"],
          [[f"{BINS[i]:.2f}", f"{BINS[i+1]:.2f}", int(h_con[i]), int(h_ran[i]),
            f"{m_con:.6f}", f"{m_ran:.6f}", 0.4] for i in range(20)])

# ============================================================================
# VERIFICATION: independent pure-python recomputation of every plotted number
# ============================================================================
print("\n=== verification: plotted vs independent recomputation ===")
ok = True

def check(label, plotted, recomputed, tol=1e-9, quiet=False):
    global ok
    good = abs(plotted - recomputed) <= tol
    ok &= good
    if not quiet or not good:
        print(f"{'OK ' if good else 'BAD'} {label:<44s} plotted={plotted:<12.6g} recomputed={recomputed:<12.6g}")
    return good

raw = load_jsonl("res_results_main.jsonl") + load_jsonl("res_results_ext.jsonl")
assert len(raw) == 13000

# --- fig 1a: pooled trial means (balanced design => equals mean of concept means)
for name, kind, met in SERIES1:
    for g, m, lo, hi, n in fig1a[name]:
        sub = [r for r in raw if r["kind"] == kind and r["gamma"] == g]
        if met == "p_yes":
            vals = [r["p_yes"] for r in sub]
        elif met == "det":
            vals = [1.0 if (r["p_yes"] > 0.5 and r["rank_c"] < 5) else 0.0 for r in sub]
        else:
            vals = [1.0 if r["lens_rank"] <= 25 else 0.0 for r in sub]
        assert len(vals) == n == 1000
        check(f"fig1a {name} g={g:g}", m, sum(vals) / len(vals))
        assert lo <= m <= hi, (name, g, lo, m, hi)
        sd = statistics.pstdev(vals)
        naive_half = 1.96 * sd / len(vals) ** 0.5
        boot_half = (hi - lo) / 2
        print(f"    CI95 [{lo:.3f}, {hi:.3f}]  half-width: concept-bootstrap={boot_half:.3f}"
              f"  naive-trial={naive_half:.3f}")
check("fig1a clean P(yes) line", clean_p_yes,
      sum(r["p_yes"] for r in raw if r["kind"] == "concept" and r["gamma"] == 0.0) / 1000)

# --- fig 1b
for (kind, g), pts in fig1b.items():
    sub = [r for r in raw if r["kind"] == kind and r["gamma"] == g]
    assert len(sub) == 1000
    for k, m, lo, hi, n in pts:
        check(f"fig1b {kind} g={g:g} k={k} [{lo:.3f},{hi:.3f}]", m,
              sum(1 for r in sub if r["rank_c"] < k) / len(sub))

# --- fig 2: straight copy of the JSON
with open(os.path.join(DATA, "res_check3_band.json")) as f:
    braw = json.load(f)
for l in range(35):
    for arr, key in ((acc_j, ("acc", "j")), (acc_l, ("acc", "logit")),
                     (kur_j, ("kur", "j")), (kur_l, ("kur", "logit"))):
        assert arr[l] == braw[key[0]][key[1]][l]
print("OK  fig2 all 4x35 values identical to res_check3_band.json")
print(f"    at read layer 22: acc_j={acc_j[22]:.4f} acc_logit={acc_l[22]:.4f}"
      f" kur_j={kur_j[22]:.3f} kur_logit={kur_l[22]:.3f}")

# --- fig 3 / fig 5 per-concept values
good3 = True
for c in CONCEPTS:
    sub = [r for r in raw if r["kind"] == "concept" and r["gamma"] == 0.4 and r["concept"] == c]
    assert len(sub) == 20
    xl = sum(1 for r in sub if r["lens_rank"] <= 25) / 20
    yn = sum(1 for r in sub if r["rank_c"] < 5) / 20
    med = statistics.median(r["lens_rank"] for r in sub)
    good3 &= (xl == pc[c]["x"]) and (yn == pc[c]["y"]) and (med == pc[c]["med_rank"]) \
        and (pc[c]["x5"] == med + 1)
ok &= good3
print("OK  fig3/fig5 all 50 (frac_landed, frac_named, median_rank) triples match"
      if good3 else "BAD fig3/fig5 per-concept mismatch")
print("    labeled concepts (fig3 + fig5):")
for c in sorted(set(REQUESTED3) | set(REQUESTED5)):
    print(f"      {c:<10s} {pc[c]['category']:<10s} landed={pc[c]['x']:.2f}"
          f" named={pc[c]['y']:.2f} median_rank_L22={pc[c]['med_rank']:g}")
# fig 5 fits: recompute closed-form OLS in pure python
print("    fig5 per-category OLS on log10(rank+1):")
for cat in CATS:
    cs = [c for c in CONCEPTS if pc[c]["category"] == cat]
    import math
    lx = [math.log10(pc[c]["med_rank"] + 1) for c in cs]; yy = [pc[c]["y"] for c in cs]
    mx, my = sum(lx) / len(lx), sum(yy) / len(yy)
    sxx = sum((a - mx) ** 2 for a in lx); sxy = sum((a - mx) * (b - my) for a, b in zip(lx, yy))
    b1 = sxy / sxx; b0 = my - b1 * mx
    check(f"fig5 fit slope {cat}", fig5_fit[cat]["slope"], b1, tol=1e-9, quiet=True)
    check(f"fig5 fit intercept {cat}", fig5_fit[cat]["intercept"], b0, tol=1e-9, quiet=True)
    f = fig5_fit[cat]
    print(f"      {cat:<10s} slope={f['slope']:+.3f} per decade  intercept={f['intercept']:.3f}"
          f"  r={f['r']:+.2f}  x-range=[{f['xmin']:g}, {f['xmax']:g}]  n={f['n']}")

# --- fig 4: pure-python median
with open(os.path.join(DATA, "res_layer_scan.json")) as f:
    sraw = json.load(f)
good4 = True
for cat in CATS:
    for l in range(35):
        vals = [d["ranks"][l] for d in sraw if d["category"] == cat]
        assert len(vals) == 50
        good4 &= statistics.median(vals) == fig4[cat][l]
ok &= good4
print("OK  fig4 all 5x35 medians match" if good4 else "BAD fig4")
print("    median lens rank at read layer 22 / min over layers:")
for cat in CATS:
    print(f"      {cat:<10s} L22={fig4[cat][22]:>8.1f}   min={fig4[cat].min():>8.1f} @L{int(fig4[cat].argmin())}")
mainmap = {(r["concept"], r["passage"]): r for r in raw if r["kind"] == "concept" and r["gamma"] == 0.4}
agree = sum(1 for d in sraw if mainmap[(d["concept"], d["passage"])]["lens_rank"] == d["ranks"][22])
print(f"    cross-check: layer-scan ranks[22] == main lens_rank for {agree}/{len(sraw)} shared trials")

# --- fig 6: pure-python binning (right-closed last bin, as np.histogram)
good6 = True
for kind, hist, mean in (("concept", h_con, m_con), ("random", h_ran, m_ran)):
    vals = [r["p_yes"] for r in raw if r["kind"] == kind and r["gamma"] == 0.4]
    assert len(vals) == 1000
    counts = [0] * 20
    for v in vals:
        i = min(int(v * 20), 19)
        counts[i] += 1
    good6 &= counts == hist.tolist() and sum(counts) == 1000
    check(f"fig6 mean p_yes {kind}", mean, sum(vals) / len(vals))
ok &= good6
print("OK  fig6 both 20-bin histograms match" if good6 else "BAD fig6 histogram mismatch")
print(f"    random bins: {h_ran.tolist()}")
print(f"    concept bins: {h_con.tolist()}")

# --- ext-data sanity summary (not plotted; for the report)
print("\n=== ext data summary (concept arm) ===")
for g in [0.4, 0.6, 0.8]:
    sub = [r for r in raw if r["kind"] == "concept" and r["gamma"] == g]
    print(f"  g={g:g}: mass={statistics.mean(r['mass'] for r in sub):.3f}"
          f" p_yes={statistics.mean(r['p_yes'] for r in sub):.3f}"
          f" landed={statistics.mean(r['lens_rank'] <= 25 for r in sub):.3f}"
          f" named={statistics.mean(r['rank_c'] < 5 for r in sub):.3f}"
          f" det={statistics.mean(r['p_yes'] > 0.5 and r['rank_c'] < 5 for r in sub):.3f}")
    for cat in CATS:
        s2 = [r for r in sub if r["category"] == cat]
        print(f"      {cat:<10s} mass={statistics.mean(r['mass'] for r in s2):.3f}"
              f" landed={statistics.mean(r['lens_rank'] <= 25 for r in s2):.2f}"
              f" named={statistics.mean(r['rank_c'] < 5 for r in s2):.2f}")

# --- dash constraint: no en/em dashes anywhere in figure text
for fn in os.listdir(OUT):
    if fn.endswith(".py"):
        src = open(os.path.join(OUT, fn), encoding="utf-8").read()
        assert "\u2013" not in src and "\u2014" not in src, "en/em dash found in script"
print("OK  no en/em dashes in figure text")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
