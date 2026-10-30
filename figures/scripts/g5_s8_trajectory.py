"""s8_trajectory (main) and bk_exp29_trajectory_rows (backup): exp 29, the NLA along its training trajectory.

Raw sources: experiments/29_trajectory/results/judged/{ckpt}_{row}.json (judge verdicts per readout),
gen29_{ckpt}_{row}.json (k=1 readouts; names recounted with lib26/lib29 regex), recon29_{ckpt}_{row}.json (critic
AUROC, FVE). analyze29.collect (reads the same raw files) supplies the text-bag AUROC for the P4 check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402
from style import C  # noqa: E402

ROOT = style.ROOT
EXP = ROOT / "experiments" / "29_trajectory"
RES = EXP / "results"
for p in (EXP, ROOT / "src", ROOT / "experiments" / "12_concept_recall"):
    sys.path.insert(0, str(p))
import lib29  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CKPTS = ["textnla", "av_base"] + [f"iter_{s:06d}" for s in range(100, 900, 100)]
XLAB = ["text\nonly", "warm\nstart", "100", "200", "300", "400", "500", "600", "700", "800"]
STRUCT = lib29.STRUCTURAL_ROWS
READABLE = lib29.READABLE_ROWS
WORDS = {"py_string": "inside a Python string", "js_string": "inside a JavaScript string",
         "latex_math": "inside LaTeX math", "json_depth": "JSON nesting depth",
         "py_arg": "call argument (1st vs 3rd)", "md_heading": "Markdown heading level",
         "md_list": "list nesting depth", "code_lang": "code language (control)"}
ITEMS22 = ROOT / "experiments/22_public_bank/results/items_base.json"


def states_cell(ck: str, row: str) -> dict:
    d = json.loads((RES / "judged" / f"{ck}_{row}.json").read_text())
    j = d["judged"]
    items = np.array([r["item"] for r in j])
    st = np.array([bool(r["stated"]) for r in j], float)
    lo, hi = style.bootstrap_ci(np.mean, st, groups=items, n_boot=2000)
    return {"p": float(st.mean()), "lo": lo, "hi": hi, "n": len(st), "n_items": len(set(items))}


def load() -> dict:
    items = json.loads(ITEMS22.read_text())
    out = {"states": {}, "names": {}, "carries": {}, "fve": {}, "cos": {}}
    for ck in CKPTS:
        for m in out:
            out[m][ck] = {}
        for row in lib29.ROWS:
            out["states"][ck][row] = states_cell(ck, row)
            gen = json.loads((RES / f"gen29_{ck}_{row}.json").read_text())
            its = [it for it in items if it["feature"] == row]
            sel, _, _ = lib29.select_split(its, lib29.MARKED[row])
            texts = [gen["sel"].get(its[i]["id"]) or "(empty)" for i in sel]
            k = sum(lib29.names_rate([t], row) for t in texts)
            out["names"][ck][row] = dict(zip(("p", "lo", "hi"), style.wilson(int(round(k)), len(texts))), k=k, n=len(texts))
            rec = json.loads((RES / f"recon29_{ck}_{row}.json").read_text())
            out["carries"][ck][row] = rec["auroc"]
            out["fve"][ck][row] = rec["fve"]
            out["cos"][ck][row] = rec["class_direction_cos"]
    return out


_CACHE: list = []


def checks_for(d: dict) -> list[dict]:
    if not _CACHE:
        _CACHE.append([{**c, "match": bool(c["match"])} for c in _checks(d)])
    return _CACHE[0]


def _checks(d: dict) -> list[dict]:
    S = lambda ck, r: d["states"][ck][r]["p"]  # noqa: E731
    rl = CKPTS[2:]
    ck = []
    t = [S("textnla", r) for r in STRUCT]
    a = [S("av_base", r) for r in STRUCT]
    r_ = [S(c, r) for c in rl for r in STRUCT]
    ck.append({"quantity": "states (judge), structural rows: textnla / av_base / RL 100-800",
               "computed": f"{min(t):.3f}-{max(t):.3f} / {min(a):.3f}-{max(a):.3f} / {min(r_):.3f}-{max(r_):.3f}",
               "record": "0.17 to 0.49 / 0.01 to 0.21 / 0.00 to 0.29", "where": "RESULTS.md table; D50; talk S8 note",
               "match": [round(v, 2) for v in (min(t), max(t), min(a), max(a), min(r_), max(r_))] == [0.17, 0.49, 0.01, 0.21, 0.00, 0.29]})
    slope = np.polyfit(np.arange(len(rl)), [np.mean([S(c, r) for r in STRUCT]) for c in rl], 1)[0]
    ck.append({"quantity": "trend of mean structural states over RL steps 100-800 (OLS slope per 100 steps)",
               "computed": f"{slope:+.4f}", "record": "no trend", "where": "RESULTS.md", "match": abs(slope) < 0.01})
    cl = [S(c, "code_lang") for c in rl]
    ck.append({"quantity": "code_lang states: textnla, av_base, RL range", "computed": f"{S('textnla','code_lang'):.3f}, {S('av_base','code_lang'):.3f}, {min(cl):.3f}-{max(cl):.3f}",
               "record": "0.50, 0.55, 0.71 to 0.77 from step 100", "where": "RESULTS.md table",
               "match": [round(v, 2) for v in (S('textnla', 'code_lang'), S('av_base', 'code_lang'), min(cl), max(cl))] == [0.50, 0.55, 0.71, 0.77]})
    N = lambda c: d["names"][c]["code_lang"]["p"]  # noqa: E731
    nl = [N(c) for c in rl]
    ck.append({"quantity": "code_lang names (regex): textnla, av_base, RL range", "computed": f"{N('textnla'):.3f}, {N('av_base'):.3f}, {min(nl):.3f}-{max(nl):.3f}",
               "record": "0.42, 0.46, 0.62 to 0.75", "where": "RESULTS.md table",
               "match": [round(v, 2) for v in (N('textnla'), N('av_base'), min(nl), max(nl))] == [0.42, 0.46, 0.62, 0.75]})
    F = lambda c: d["fve"][c]["py_string"]  # noqa: E731
    ck.append({"quantity": "FVE py_string: textnla, av_base, step 100, step 800", "computed": f"{F('textnla'):.3f}, {F('av_base'):.3f}, {F('iter_000100'):.3f}, {F('iter_000800'):.3f}",
               "record": "-0.05, 0.18, 0.53, 0.59", "where": "RESULTS.md table; D50 '0.18 to 0.59'",
               "match": [round(v, 2) for v in (F('textnla'), F('av_base'), F('iter_000100'), F('iter_000800'))] == [-0.05, 0.18, 0.53, 0.59]})
    ca = lambda cs: [d["carries"][c][r] for c in cs for r in READABLE]  # noqa: E731
    ck.append({"quantity": "carries, readable rows: textnla / av_base / RL", "computed": f"{min(ca(['textnla'])):.3f}-{max(ca(['textnla'])):.3f} / {min(ca(['av_base'])):.3f}-{max(ca(['av_base'])):.3f} / {min(ca(rl)):.3f}-{max(ca(rl)):.3f}",
               "record": "0.84-0.99 / 0.87-0.99 / 0.91-1.00", "where": "RESULTS.md table",
               "match": [round(v, 2) for v in (min(ca(['textnla'])), max(ca(['textnla'])), min(ca(['av_base'])), max(ca(['av_base'])), min(ca(rl)), max(ca(rl)))] == [0.84, 0.99, 0.87, 0.99, 0.91, 1.00]})
    # pre-registered predictions, recomputed
    p1 = all(S(c, r) <= 0.25 for c in CKPTS for r in STRUCT)
    viol = [(c, r, round(S(c, r), 2)) for c in CKPTS for r in STRUCT if S(c, r) > 0.25]
    ck.append({"quantity": "P1 structural states <= 0.25 at every checkpoint", "computed": f"{'holds' if p1 else 'fails'}; violations {viol}",
               "record": "FAILS on TextNLA (py_string 0.47, js_string 0.49, md_heading 0.30); js_string step 200 0.29",
               "where": "RESULTS.md prediction 1", "match": not p1})
    d2 = S("iter_000400", "code_lang") - S("av_base", "code_lang")
    ck.append({"quantity": "P2 code_lang states rise av_base -> step 400 (>= 0.15)", "computed": f"{d2:+.3f} ({'holds' if d2 >= 0.15 else 'fails'})",
               "record": "Holds: +0.16", "where": "RESULTS.md prediction 2", "match": d2 >= 0.15 and round(d2, 2) == 0.16})
    d3 = np.mean([d["carries"]["iter_000400"][r] - d["carries"]["av_base"][r] for r in READABLE])
    t3 = np.mean([d["carries"]["textnla"][r] - d["carries"]["av_base"][r] for r in READABLE])
    ck.append({"quantity": "P3 carrying rise av_base -> 400 (>= 0.05) and |textnla - av_base| <= 0.05",
               "computed": f"rise {d3:+.3f}, textnla {t3:+.3f} ({'holds' if d3 >= 0.05 and abs(t3) <= 0.05 else 'fails'})",
               "record": "FAILS: +0.02; TextNLA -0.04", "where": "RESULTS.md prediction 3",
               "match": d3 < 0.05 and round(d3, 2) == 0.02 and round(t3, 2) == -0.04})
    try:
        import analyze29
        s = analyze29.collect(RES, "qwen")
        spans = {r: max(s["textbag"][c][r] for c in s["checkpoints"]) - min(s["textbag"][c][r] for c in s["checkpoints"]) for r in lib29.STRING_ROWS}
        ck.append({"quantity": "P4 text-bag AUROC span across ten checkpoints (< 0.05 on string rows)",
                   "computed": ", ".join(f"{r} {v:.3f}" for r, v in spans.items()) + f" (tokenizer {s['tokenizer']})",
                   "record": "FAILS narrowly: 0.04, 0.09, 0.06", "where": "RESULTS.md prediction 4",
                   "match": [round(v, 2) for v in spans.values()] == [0.04, 0.09, 0.06]})
    except Exception as e:  # noqa: BLE001
        ck.append({"quantity": "P4 text-bag span", "computed": f"not recomputed ({type(e).__name__})", "record": "0.04, 0.09, 0.06",
                   "where": "RESULTS.md", "match": False})
    ck.append({"quantity": "D50 / talk S8 wording 'code-language naming (0.55 to 0.75)'",
               "computed": f"judge states {S('av_base','code_lang'):.2f} -> {min(cl):.2f}-{max(cl):.2f}; regex names {N('av_base'):.2f} -> {min(nl):.2f}-{max(nl):.2f}",
               "record": "0.55 to 0.75", "where": "decisions D50; talk script S8 update",
               "match": False})
    return ck


def main_fig(d: dict) -> None:
    with plt.rc_context(style.SLIDE_RC):
        _main_fig(d)


def _main_fig(d: dict) -> None:
    fs = style.MIN_PT
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.2), sharex=True)
    x = np.array([-1.9, 0.0] + [1.0 + i for i in range(8)])
    for ax in axes:
        ax.axvspan(-2.4, -1.4, color=C["text_pale"], alpha=0.35, lw=0, zorder=0)
        ax.axvline(0.5, color=C["line"], lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(["text", "0", "", "", "", "400", "", "", "", "800"])
        ax.set_xlim(-2.6, x[-1] + 0.4)
    # panel 1: structural statements (judge)
    ax = axes[0]
    vals = np.array([[d["states"][c][r]["p"] for c in CKPTS] for r in STRUCT])
    for v in vals:
        ax.plot(x, v, color=C["nla_pale"], lw=1.0, alpha=0.8, zorder=1)
    med = np.median(vals, 0)
    ax.plot(x, med, color=C["nla"], lw=3.0, marker="o", ms=6, zorder=3)
    ax.axhline(0.25, color=C["line"], lw=1.4, ls="--")
    ax.text(x[-1] + 0.45, 0.27, "bar 0.25", ha="right", va="bottom", fontsize=fs, color=C["muted"])
    ax.set_ylim(-0.02, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("states structure (judge)")
    ax.set_ylabel("share of readouts")
    ax.annotate("text only:\nstrings 0.47-0.49", xy=(x[0], vals.max(0)[0]), xytext=(-1.6, 0.72),
                fontsize=fs, color=C["text"], linespacing=1.05,
                arrowprops=dict(arrowstyle="-", color=C["text"], lw=1))
    ax.text(5.0, 0.42, "flat\nthrough RL", ha="center", fontsize=fs, color=C["nla"], linespacing=1.05)
    # panel 2: code language, judge vs regex
    ax = axes[1]
    for key, col, lab, mk in (("states", C["nla"], "states (judge)", "o"), ("names", C["nla_pale"], "names (regex)", "s")):
        p = np.array([d[key][c]["code_lang"]["p"] for c in CKPTS])
        lo = np.array([d[key][c]["code_lang"]["lo"] for c in CKPTS])
        hi = np.array([d[key][c]["code_lang"]["hi"] for c in CKPTS])
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, p, color=col, lw=3.0 if key == "states" else 2.0, marker=mk, ms=5)
        ax.text(x[-1] + 0.4, (0.93 if key == "states" else 0.47), lab, ha="right", va="center", fontsize=fs,
                color=col if key == "states" else "#D0806F")
    ax.set_ylim(-0.02, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels([])
    ax.set_title("code language (control)")
    ax.text(x[-1] + 0.4, 0.2, "judge 0.55 at step 0,\n0.77 at 100", ha="right", va="center",
            fontsize=fs, color=C["nla"], linespacing=1.05)
    # panel 3: reconstruction
    ax = axes[2]
    fv = np.array([[d["fve"][c][r] for c in CKPTS] for r in lib29.ROWS])
    for r, v in zip(lib29.ROWS, fv):
        if r != "py_string":
            ax.plot(x, v, color=C["line"], lw=1.0, zorder=1)
    ps = np.array([d["fve"][c]["py_string"] for c in CKPTS])
    ax.plot(x, ps, color=C["probe"], lw=3.0, marker="o", ms=6, zorder=3)
    for i, dx, dy, ha in ((0, 0.25, -0.15, "left"), (1, -0.2, 0.07, "right"), (2, -0.1, 0.07, "right")):
        ax.text(x[i] + dx, ps[i] + dy, f"{ps[i]:.2f}", ha=ha, fontsize=fs, color=C["probe"])
    ax.axhline(0, color=C["line"], lw=1.2, ls="--")
    ax.set_ylim(-0.3, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("reconstruction (FVE)")
    ax.text(x[-1] + 0.4, 1.0, f"Python strings,\n{ps[9]:.2f} at 800", ha="right", va="top", fontsize=fs,
            color=C["probe"], linespacing=1.05)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1))
    fig.text(0.5, 0.01, "RL step (0 = warm start, before RL); shaded = text-only warm start", ha="center", va="bottom", fontsize=fs)
    caption = ("The NLA at ten points of its history on the exp 22 public bank: text-only warm start (base model given the "
               "prefix text under the warm-start instruction; shaded), the activation warm start (SFT, no RL), and GRPO "
               "adapters every 100 steps to 800. Left: judge-stated rate on the seven structural rows (40 items x 8 readouts "
               "per cell; faint = each row, bold = median). Middle: code language, judge (item-clustered bootstrap 95%) and "
               "strict regex 'names' rate on 160 k=1 readouts (Wilson 95%). Right: reconstruction (FVE) of the critic's rebuild "
               "of k=1 readouts, bold = Python strings, grey = the other rows (no per-item values saved, so no CI). "
               "Caveat: one final critic scores every checkpoint, so carrying and reconstruction "
               "are 'in a direction the final critic reads'.")
    style.save(fig, "s8_trajectory",
               title="RL raises reconstruction and stating the code language, not stating structure; the warm start sets that",
               caption=caption, section="S8", role="main",
               sources=["experiments/29_trajectory/results/judged/{ckpt}_{row}.json (80 files)",
                        "experiments/29_trajectory/results/gen29_{ckpt}_{row}.json (80 files)",
                        "experiments/29_trajectory/results/recon29_{ckpt}_{row}.json (80 files)",
                        "experiments/22_public_bank/results/items_base.json"],
               checks=checks_for(d),
               notes=("Predictions: P2 holds (+0.16 code_lang states, av_base -> step 400); P1 fails on TextNLA (and js_string "
                      "step 200 at 0.29 is above 0.25 too); P3 fails on the ceiling (carrying rise +0.02); P4 fails narrowly "
                      "(text-bag span 0.09 on js_string). Judge: Gemini 3.8 Flash, exp 22's states prompt, T=0 (existing "
                      "outputs only). TextNLA prompt = experiments/06_resolving_decomposition/data/nla_instruction.txt (the "
                      "warm-start instruction), NOT the NLA actor prompt. Record wording: D50 and the talk S8 note say "
                      "'code-language naming (0.55 to 0.75)', which splices the judge start (0.55) with the regex top (0.75); "
                      "judge is 0.55 -> 0.71-0.77, regex 0.46 -> 0.62-0.75."))


def backup_fig(d: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 8.2), sharex=True, sharey=True)
    x = np.arange(len(CKPTS))
    for ax, row in zip(axes.flat, lib29.ROWS):
        ax.axvspan(-0.5, 0.5, color=C["text_pale"], alpha=0.35, lw=0)
        ax.axvline(1.5, color=C["line"], lw=1)
        ax.axhline(0.5, color=C["line"], lw=1, ls="--")
        p = np.array([d["states"][c][row]["p"] for c in CKPTS])
        lo = np.array([d["states"][c][row]["lo"] for c in CKPTS])
        hi = np.array([d["states"][c][row]["hi"] for c in CKPTS])
        ax.fill_between(x, lo, hi, color=C["nla"], alpha=0.18, lw=0)
        ax.plot(x, p, color=C["nla"], lw=2.6, marker="o", ms=4)
        ca = np.array([d["carries"][c][row] for c in CKPTS])
        ax.plot(x, ca, color=C["probe"], lw=1.8, marker="^", ms=4)
        fv = np.array([d["fve"][c][row] for c in CKPTS])
        ax.plot(x, fv, color=C["muted"], lw=1.4, ls="--")
        ax.set_title(WORDS[row], fontsize=14)
        ax.set_ylim(-0.25, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(["T", "W", "100", "", "", "400", "", "", "", "800"], fontsize=11.5)
        ax.tick_params(labelsize=12)
    for ax in axes[1]:
        ax.set_xlabel("checkpoint (T text-only, W warm start)", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("share / AUROC /\nreconstruction", fontsize=13)
    handles = [plt.Line2D([], [], color=C["nla"], marker="o", lw=2.6, label="states it (judge, bootstrap 95%)"),
               plt.Line2D([], [], color=C["probe"], marker="^", lw=1.8, label="carries it (critic AUROC)"),
               plt.Line2D([], [], color=C["muted"], ls="--", lw=1.4, label="reconstruction (FVE), the reward")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    caption = ("Per-row exp 29: judge-stated rate (40 items x 8 readouts, item-clustered bootstrap 95%), critic carrying "
               "AUROC and reconstruction (FVE; no per-item values saved, no CI) at each checkpoint; reconstruction on JSON depth goes below the "
               "axis floor at T and W (-0.73, -0.53). Carrying is near ceiling on the readable rows from the warm start "
               "on; stated rates move only on the code-language row, and drop from text-only to the activation warm start.")
    style.save(fig, "bk_exp29_trajectory_rows",
               title="Per row, only the code-language row's stated rate moves with RL",
               caption=caption, section="backup:exp29", role="backup",
               sources=["experiments/29_trajectory/results/judged/*", "experiments/29_trajectory/results/recon29_*"],
               checks=checks_for(d), notes="Same data as s8_trajectory.")


if __name__ == "__main__":
    data = load()
    main_fig(data)
    backup_fig(data)
    for c in checks_for(data):
        print(("OK  " if c["match"] else "DIFF"), c["quantity"], "|", c["computed"], "|", c["record"])
