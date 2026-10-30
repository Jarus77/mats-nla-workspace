"""s8_layer_sweep (main) and bk_exp26_layer_rows (backup): exp 26, Oracle Lens at layers 20-60.

Every number is read from experiments/26_layer_sweep/results: sweep26_{row}.json (probe, prompted, carries per layer)
and ol26_{row}_L{L}.json (the free readouts, from which the naming rate is recounted with lib26's regexes).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402
from style import C  # noqa: E402

ROOT = style.ROOT
EXP = ROOT / "experiments" / "26_layer_sweep"
RES = EXP / "results"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "src"))
import lib26  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAYERS = lib26.sweep_layers()
CORE = ("py_string", "js_string", "latex_math", "json_depth")          # readable structural rows
STRUCT = ("py_string", "js_string", "latex_math", "json_depth", "py_arg", "md_heading", "md_list")
ALL = STRUCT + ("code_lang",)
WORDS = {"py_string": "inside a Python string", "js_string": "inside a JavaScript string",
         "latex_math": "inside LaTeX math", "json_depth": "JSON nesting depth",
         "py_arg": "call argument (1st vs 3rd)", "md_heading": "Markdown heading level",
         "md_list": "list nesting depth", "code_lang": "code language (control)"}
N_POS = N_NEG = 80


def hanley_ci(a: float, n1: int = N_POS, n2: int = N_NEG) -> tuple[float, float]:
    """Hanley-McNeil (1982) analytic 95% interval for an AUROC (no per-item scores were saved)."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
    return max(0.0, a - 1.96 * se), min(1.0, a + 1.96 * se)


def load() -> dict:
    items = json.loads((ROOT / "experiments/22_public_bank/results/items_base.json").read_text())
    out = {}
    for row in ALL:
        sw = json.loads((RES / f"sweep26_{row}.json").read_text())
        sel, y, _ = lib26.select_items([i for i in items if i["feature"] == row], lib26.MARKED[row])
        assert int(y.sum()) == N_POS and len(y) == N_POS + N_NEG
        names_k = {}
        for L in LAYERS:
            ol = json.loads((RES / f"ol26_{row}_L{L}.json").read_text())["ol"]
            texts = [ol.get(it["id"]) for it in sel]
            names_k[L] = sum(lib26.names_state(t, row) for t in texts)
        out[row] = {m: {L: float(lib26.cell(sw[m], L)) for L in LAYERS} for m in ("readable", "ol_prompted", "carries", "names")}
        out[row]["names_k"] = names_k
        out[row]["n"] = len(sel)
    return out


def checks_for(d: dict) -> list[dict]:
    ck = []
    # recount of the naming rate against the stored sweep26 value, every row x layer
    bad = [(r, L) for r in ALL for L in LAYERS if abs(d[r]["names_k"][L] / d[r]["n"] - d[r]["names"][L]) > 1e-9]
    ck.append({"quantity": "names rate recounted from ol26 readouts with lib26.NAME_RE vs sweep26 'names' (96 cells)",
               "computed": f"{96 - len(bad)} of 96 identical", "record": "sweep26_{row}.json names", "where": "results/sweep26_*.json",
               "match": not bad})
    rng = lambda rows, m, Ls=LAYERS: (min(d[r][m][L] for r in rows for L in Ls), max(d[r][m][L] for r in rows for L in Ls))  # noqa: E731
    r5 = CORE + ("code_lang",)
    lo, hi = rng(r5, "readable")
    lo_nc, _ = rng(CORE, "readable")
    ck.append({"quantity": "probe AUROC, readable rows (string, math, depth, code_lang), layers 20-60",
               "computed": f"{lo:.3f}-{hi:.3f} (min over string/math/depth rows {lo_nc:.3f}; code_lang at 32 {d['code_lang']['readable'][32]:.3f})",
               "record": "0.96 to 1.00 at every layer (code_lang 0.89 at 32)", "where": "RESULTS.md table",
               "match": round(lo_nc, 2) >= 0.96 and round(d['code_lang']['readable'][32], 2) == 0.89})
    at20 = [d[r]["readable"][20] for r in CORE + ("py_arg", "code_lang")]
    ck.append({"quantity": "probe AUROC at layer 20 on the six readable rows", "computed": f"{min(at20):.3f}-{max(at20):.3f}",
               "record": "already 0.98 to 1.00 at layer 20 (P1 text)", "where": "RESULTS.md prediction 1",
               "match": round(min(at20), 2) >= 0.98})
    lo, hi = rng(("py_arg",), "readable")
    ck.append({"quantity": "probe AUROC py_arg", "computed": f"{lo:.3f}-{hi:.3f}", "record": "0.94 to 0.99",
               "where": "RESULTS.md table", "match": (round(lo, 2), round(hi, 2)) == (0.94, 0.99)})
    lo, hi = rng(("md_heading",), "readable")
    lo2, hi2 = rng(("md_list",), "readable")
    ck.append({"quantity": "P1 Markdown half: probe 'near 0.6 at every layer' on md_heading and md_list",
               "computed": f"md_heading {lo:.3f}-{hi:.3f} (0.77 at 20, 0.78 at 32); md_list {lo2:.3f}-{hi2:.3f}",
               "record": "RESULTS: 'The Markdown half holds (md_heading 0.63 to 0.78, md_list about 0.5)'",
               "where": "RESULTS.md prediction 1",
               "match": False})
    lo, hi = rng(r5, "ol_prompted")
    ck.append({"quantity": "Oracle Lens prompted AUROC, readable rows", "computed": f"{lo:.3f}-{hi:.3f}", "record": "0.84 to 1.00",
               "where": "RESULTS.md table", "match": (round(lo, 2), round(hi, 2)) == (0.84, 1.00)})
    late = [L for L in LAYERS if L >= 36]
    spans = {r: max(d[r]["ol_prompted"][L] for L in late) - min(d[r]["ol_prompted"][L] for L in late) for r in ALL}
    ck.append({"quantity": "P2: prompted AUROC span 36-60, largest row", "computed": f"max span {max(spans.values()):.3f} ({max(spans, key=spans.get)})",
               "record": "under 0.10 on every row (holds)", "where": "RESULTS.md prediction 2", "match": max(spans.values()) < 0.10})
    lo, hi = rng(r5, "carries")
    ck.append({"quantity": "carries (OL readout through the critic), readable rows", "computed": f"{lo:.3f}-{hi:.3f}",
               "record": "0.76 to 1.00", "where": "RESULTS.md table", "match": (round(lo, 2), round(hi, 2)) == (0.76, 1.00)})
    pa = d["py_arg"]["carries"]
    ck.append({"quantity": "py_arg carries at 20-24 and after", "computed": f"20: {pa[20]:.3f}, 24: {pa[24]:.3f}; 28-60: {min(pa[L] for L in LAYERS if L >= 28):.3f}-{max(pa[L] for L in LAYERS if L >= 28):.3f}",
               "record": "0.63 to 0.66 at 20 to 24, then 0.44 to 0.59", "where": "RESULTS.md table",
               "match": round(pa[20], 2) == 0.63 and round(pa[24], 2) == 0.66 and round(min(pa[L] for L in LAYERS if L >= 28), 2) == 0.44 and round(max(pa[L] for L in LAYERS if L >= 28), 2) == 0.59})
    mn = min(d[r]["carries"][L] for r in CORE for L in LAYERS if d[r]["readable"][L] >= 0.80)
    ck.append({"quantity": "P3: min carries where probe >= 0.80 on string/math/depth rows", "computed": f"{mn:.3f}",
               "record": "minimum 0.76 (py_string at 56), holds", "where": "RESULTS.md prediction 3", "match": round(mn, 2) == 0.76})
    mx = max(d[r]["names_k"][L] / d[r]["n"] for r in STRUCT for L in LAYERS)
    ck.append({"quantity": "P4: max strict naming rate on any structural row and layer", "computed": f"{mx:.4f}",
               "record": "at most 0.02 (holds); D50 'names it in at most 2% of readouts'", "where": "RESULTS.md prediction 4; decisions D50",
               "match": mx <= 0.02})
    cl = [d["code_lang"]["names_k"][L] / 160 for L in LAYERS]
    ck.append({"quantity": "code_lang names (language-name regex)", "computed": f"{min(cl):.3f}-{max(cl):.3f}", "record": "0.03 to 0.14",
               "where": "RESULTS.md table", "match": (round(min(cl), 2), round(max(cl), 2)) == (0.03, 0.14)})
    lo, hi = rng(CORE, "readable")
    ck.append({"quantity": "D50: probe on the readable rows", "computed": f"string/math/depth {lo:.3f}-{hi:.3f}; py_arg {rng(('py_arg',), 'readable')[0]:.3f}-{rng(('py_arg',), 'readable')[1]:.3f}",
               "record": "probe 0.96 to 1.00 on the readable rows, already at layer 20", "where": "decisions D50",
               "match": round(lo, 2) >= 0.96})
    return ck


def main_fig(d: dict) -> None:
    with plt.rc_context(style.SLIDE_RC):
        _main_fig(d)


def _main_fig(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.25))
    x = np.array(LAYERS)
    ax.axvspan(41, 43, color=C["line"], alpha=0.35, lw=0, zorder=0)
    ax.text(42, 1.03, "layer 42 (NLA's layer)", ha="center", va="bottom", fontsize=15, color=C["muted"])
    for yv, lab in ((0.5, "chance"), (0.80, "0.80 line")):
        ax.axhline(yv, color=C["line"], lw=1.4, ls="--", zorder=0)
        ax.text(19.3, yv - 0.015, lab, va="top", ha="left", fontsize=15, color=C["muted"])
    series = (("readable", C["probe"], "o", "probe"),
              ("ol_prompted", C["ol_pale"], "s", "prompted"),
              ("carries", C["ol"], "^", "readout\ncarries"))
    label_y = {}
    for m, col, mk, lab in series:
        vals = np.array([[d[r][m][L] for L in LAYERS] for r in CORE])
        med, lo, hi = np.median(vals, 0), vals.min(0), vals.max(0)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0, zorder=1)
        lw = 3.0 if m == "carries" else 2.0
        ax.plot(x, med, color=col, lw=lw, marker=mk, ms=6, zorder=3)
        label_y[m] = (med[-1], col, lab)
    # names: pooled over all seven structural rows, Wilson 95%
    k = np.array([sum(d[r]["names_k"][L] for r in STRUCT) for L in LAYERS])
    n = np.array([sum(d[r]["n"] for r in STRUCT) for _ in LAYERS])
    w = np.array([style.wilson(int(a), int(b)) for a, b in zip(k, n)])
    ax.fill_between(x, w[:, 1], w[:, 2], color=C["ol"], alpha=0.25, lw=0, zorder=1)
    ax.plot(x, w[:, 0], color=C["ol"], lw=3.0, ls=(0, (1.2, 1.2)), marker="D", ms=5, zorder=4)
    # exceptions
    pa = [d["py_arg"]["carries"][L] for L in LAYERS]
    ax.plot(x, pa, color=C["ol"], lw=1.5, ls="--", alpha=0.75, zorder=2)
    ax.text(26, 0.37, "call argument: carried only at 20-24", fontsize=15, color=C["ol"], va="center", ha="left")
    cl = [d["code_lang"]["names_k"][L] / 160 for L in LAYERS]
    ax.plot(x, cl, color=C["muted"], lw=1.5, ls="--", marker=".", zorder=2)
    ax.text(44.5, 0.19, "names code language\n(control): 0.03-0.14", fontsize=15, color=C["muted"],
            va="center", ha="left", linespacing=1.1)
    # direct labels on the right
    for m, (yv, col, lab) in label_y.items():
        dy = {"readable": 0.03, "ol_prompted": -0.0, "carries": -0.07}[m]
        ax.text(61.0, yv + dy, lab, color=col if m != "ol_prompted" else "#8E79C9", fontsize=15, va="center",
                ha="left", fontweight="bold" if m == "carries" else "normal", linespacing=1.05)
    ax.text(61.0, 0.06, "names\n(regex):\n0 to 2%", color=C["ol"], fontsize=15,
            fontweight="bold", va="center", ha="left", linespacing=1.05)
    ax.set_xlim(19, 60.6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([L for L in LAYERS if L not in (40, 44)])
    ax.set_xlabel("layer Oracle Lens reads (all lines are Oracle Lens except the probe)")
    ax.set_ylabel("AUROC / share")
    caption = ("Oracle Lens at its twelve read layers (20 to 60 step 4, plus 42), exp 22 public bank, 160 items per row "
               "(80 marked, 80 unmarked). Upper three lines: median over the four readable structural rows (Python and "
               "JavaScript strings, LaTeX math, JSON depth), band = min to max of those rows; no per-item scores were "
               "saved, so the AUROC lines carry no CI (per-row Hanley-McNeil intervals in the backup). Dotted line: strict "
               "regex naming rate pooled over all seven structural rows (n = 1,120 readouts per layer), Wilson 95% band. "
               "The Markdown rows are omitted from the upper lines because the probe reads them at only 0.49 to 0.78; "
               "they carry about 0.6 and 0.5 and are named in at most 2% of readouts (backup).")
    style.save(fig, "s8_layer_sweep",
               title="Oracle Lens carries the state and never names it, at every layer from 20 to 60",
               caption=caption, section="S8", role="main",
               sources=[f"experiments/26_layer_sweep/results/sweep26_{r}.json" for r in ALL]
               + ["experiments/26_layer_sweep/results/ol26_{row}_L{layer}.json (96 files)",
                  "experiments/22_public_bank/results/items_base.json"],
               checks=checks_for(d),
               notes=("Carries = Oracle Lens k=1 free readout at the layer, rebuilt by the NLA reconstructor, scored by "
                      "exp 14's frozen probe against the TRUE layer-42 activations. Prompted = exp 22's frozen phrasings, "
                      "max of mean-of-three and fixed-label AUROC. Names recounted here from the ol26 readouts with "
                      "lib26.NAME_RE (exp 23 state regexes; py_arg ordinal-argument regex; code_lang language-name regex); "
                      "identical to sweep26 'names' in all 96 cells. Record discrepancy: RESULTS.md says P1's Markdown "
                      "half holds, but the prediction was 'near 0.6 at every layer' and md_heading reaches 0.77 (layer 20) "
                      "and 0.78 (layer 32); partial at best. Layer 42 is off Oracle Lens's trained grid and read with the "
                      "carrier saying 'layer 42' (exp 22's convention)."))


def backup_fig(d: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 8.2), sharex=True, sharey=True)
    x = np.array(LAYERS)
    order = ("py_string", "js_string", "latex_math", "json_depth", "py_arg", "md_heading", "md_list", "code_lang")
    for ax, row in zip(axes.flat, order):
        ax.axvline(42, color=C["line"], lw=1)
        ax.axhline(0.5, color=C["line"], lw=1.2, ls="--")
        ax.axhline(0.8, color=C["line"], lw=1.2, ls="--")
        for m, col, mk in (("readable", C["probe"], "o"), ("ol_prompted", C["ol_pale"], "s"), ("carries", C["ol"], "^")):
            v = np.array([d[row][m][L] for L in LAYERS])
            ci = np.array([hanley_ci(a) for a in v])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
            ax.plot(x, v, color=col, marker=mk, ms=4, lw=2.4 if m == "carries" else 1.8)
        w = np.array([style.wilson(d[row]["names_k"][L], d[row]["n"]) for L in LAYERS])
        ncol = C["muted"] if row == "code_lang" else C["ol"]
        ax.fill_between(x, w[:, 1], w[:, 2], color=ncol, alpha=0.2, lw=0)
        ax.plot(x, w[:, 0], color=ncol, ls=(0, (1.2, 1.2)), lw=2.4, marker="D", ms=3.5)
        ax.set_title(WORDS[row], fontsize=14)
        ax.set_ylim(-0.02, 1.04)
        ax.set_xticks([20, 30, 40, 50, 60])
        ax.tick_params(labelsize=12)
    for ax in axes[1]:
        ax.set_xlabel("layer", fontsize=13)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC / share", fontsize=13)
    handles = [plt.Line2D([], [], color=C["probe"], marker="o", lw=1.8, label="probe (in the activation)"),
               plt.Line2D([], [], color=C["ol_pale"], marker="s", lw=1.8, label="Oracle Lens prompted"),
               plt.Line2D([], [], color=C["ol"], marker="^", lw=2.4, label="Oracle Lens readout carries (critic)"),
               plt.Line2D([], [], color=C["ol"], ls=(0, (1.2, 1.2)), lw=2.4, marker="D", label="Oracle Lens names it (regex)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=12.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    caption = ("Per-row version of s8_layer_sweep: the eight exp 22 rows, 160 items each (80/80). AUROC bands are "
               "Hanley-McNeil analytic 95% intervals (no per-item scores saved, so no bootstrap); naming bands Wilson 95% "
               "over 160 readouts. Dashed lines at chance and 0.80; grey vertical line at 42. Code language naming uses a "
               "language-name regex and is the positive control; the Markdown rows are barely readable by the probe, so "
               "there is little to carry.")
    style.save(fig, "bk_exp26_layer_rows",
               title="Every row is flat across layers; only the call argument loses carrying after layer 24",
               caption=caption, section="backup:exp26", role="backup",
               sources=[f"experiments/26_layer_sweep/results/sweep26_{r}.json" for r in ALL]
               + ["experiments/26_layer_sweep/results/ol26_{row}_L{layer}.json (96 files)"],
               checks=checks_for(d),
               notes="Hanley-McNeil SE uses n1 = n2 = 80 per row. Names recounted from readouts (identical to sweep26).")


if __name__ == "__main__":
    data = load()
    main_fig(data)
    backup_fig(data)
    for c in checks_for(data):
        print(("OK  " if c["match"] else "DIFF"), c["quantity"], "|", c["computed"], "|", c["record"])
