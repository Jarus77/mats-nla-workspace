"""s8_naming_lora (main): exp 30, a supervised naming adapter on the NLA (arms base / control C / statement S).

Raw sources: experiments/30_naming_lora/results/eval30_{base,C,S}.json (per-item verdicts, readouts, FVE; gate FVE;
the 500 random-point readouts), train30_{S,C}.json, experiments/23_random_points/results/items_base.json (labels).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402
from style import C  # noqa: E402

ROOT = style.ROOT
EXP = ROOT / "experiments" / "30_naming_lora"
RES = EXP / "results"
for p in (EXP, ROOT / "src", ROOT / "experiments" / "23_random_points"):
    sys.path.insert(0, str(p))
from lib30 import statement_accuracy  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXPL = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)        # mats.readers.nla.EXPLANATION_RE
ROWS = [("py_string", "Python string"), ("py_arg", "call argument"), ("json_depth", "JSON nesting depth"),
        ("code_lang", "code language"), ("md_heading", "Markdown heading level"), ("md_list", "list nesting depth"),
        ("js_string", "JavaScript string"), ("latex_math", "LaTeX math")]
GROUP_BREAKS = (4, 6)
PAIR = {"inside": "The current position is inside a string literal", "outside": "The current position is outside a string literal"}


def extract(t: str) -> str:
    m = EXPL.search(t or "")
    return m.group(1).strip() if m else ""


def load() -> dict:
    ev = {a: json.loads((RES / f"eval30_{a}.json").read_text()) for a in ("base", "C", "S")}
    items23 = json.loads((ROOT / "experiments/23_random_points/results/items_base.json").read_text())
    out = {"ev": ev, "rows": {}, "rp": {}}
    for row, _ in ROWS:
        for a in ev:
            per = ev[a]["rows"][row]["per_item"]
            v = [x["verdict"] for x in per.values()]
            out["rows"][(a, row)] = {k: v.count(k) for k in ("correct", "wrong", "none")} | {"n": len(v)}
    st = [{"id": it["id"], "value": "inside" if it["meta"]["labels"]["in_string"] else "outside"} for it in items23]
    for a in ev:
        texts = {i: extract(r) for i, r in ev[a]["random_points"]["readouts"].items()}
        acc = statement_accuracy(texts, st, PAIR)
        for cond in ("inside", "outside"):
            ids = [s["id"] for s in st if s["value"] == cond]
            says = {"inside": 0, "outside": 0, "nothing": 0, "both": 0}
            for i in ids:
                t = re.sub(r"\s+", " ", texts[i]).lower()
                hi, ho = PAIR["inside"].lower() in t, PAIR["outside"].lower() in t
                says["both" if hi and ho else "inside" if hi else "outside" if ho else "nothing"] += 1
            out["rp"][(a, cond)] = says | {"n": len(ids)}
        out["rp"][(a, "acc")] = acc
    # paired per-item FVE difference S - C on the six trained rows (same per-row baseline for both arms)
    diffs, grp = [], []
    for row, _ in ROWS[:6]:
        pS, pC = ev["S"]["rows"][row]["per_item"], ev["C"]["rows"][row]["per_item"]
        for i in pS:
            diffs.append(pS[i]["fve"] - pC[i]["fve"])
            grp.append(row)
    diffs = np.array(diffs)
    out["fve_diff"] = (float(diffs.mean()), *style.bootstrap_ci(np.mean, diffs))
    out["train"] = {a: json.loads((RES / f"train30_{a}.json").read_text()) for a in ("S", "C")}
    return out


def checks_for(d: dict) -> list[dict]:
    ev, ck = d["ev"], []
    g = {a: ev[a]["gate"]["fve"] for a in ev}
    ck.append({"quantity": "gate FVE base / control / statement", "computed": f"{g['base']:.3f} / {g['C']:.3f} / {g['S']:.3f}",
               "record": "0.504 / 0.512 / 0.507", "where": "RESULTS.md; D49; talk S8 note",
               "match": [round(g[a], 3) for a in ("base", "C", "S")] == [0.504, 0.512, 0.507]})
    ck.append({"quantity": "P1 first clause |S - C| <= 0.02", "computed": f"{abs(g['S'] - g['C']):.3f} (holds)", "record": "holds",
               "where": "RESULTS.md / table30.md", "match": abs(g["S"] - g["C"]) <= 0.02})
    ck.append({"quantity": "P1 second clause: both arms within 0.03 of 0.761", "computed": f"S {g['S'] - 0.761:+.3f}, C {g['C'] - 0.761:+.3f}: FAILS",
               "record": "not scored (table30 and RESULTS score P1 as PASS on the first clause only)", "where": "RESULTS.md predictions / table30.md",
               "match": False})
    cond = {"py_string": (15, 19), "py_arg": (13, 16), "json_depth": (22, 23), "code_lang": (9, 9), "md_heading": (15, 26),
            "md_list": (11, 24), "js_string": (10, 17), "latex_math": (1, 1)}
    for row, (c, n) in cond.items():
        r = d["rows"][("S", row)]
        ck.append({"quantity": f"arm S {row}: correct of stated (held-out files, n=32)", "computed": f"{r['correct']} of {r['correct'] + r['wrong']}",
                   "record": f"{c} of {n}", "where": "RESULTS.md table", "match": (r["correct"], r["correct"] + r["wrong"]) == (c, n)})
    base_named = sum(d["rows"][(a, r)]["correct"] + d["rows"][(a, r)]["wrong"] for a in ("base", "C") for r, _ in ROWS)
    ck.append({"quantity": "base and control statements on held-out rows", "computed": f"{base_named} statements in 512 readouts",
               "record": "naming 0.00 on every row", "where": "RESULTS.md table", "match": base_named == 0})
    rpS = d["rp"][("S", "inside")]
    n = rpS["n"]
    comp = f"{rpS['inside']/n:.3f} / {rpS['outside']/n:.3f} / {rpS['nothing']/n:.3f} (counts {rpS['inside']}/{rpS['outside']}/{rpS['nothing']} of {n}; both {rpS['both']})"
    ck.append({"quantity": "random points, arm S, point inside a string: says inside / outside / nothing", "computed": comp,
               "record": "0.16 / 0.24 / 0.60", "where": "RESULTS.md random-points table; D49",
               "match": [round(rpS[k] / n, 2) for k in ("inside", "outside", "nothing")] == [0.16, 0.24, 0.60]})
    ro = d["rp"][("S", "outside")]
    n2 = ro["n"]
    ck.append({"quantity": "random points, arm S, point not in a string: says inside / outside / nothing",
               "computed": f"{ro['inside']/n2:.3f} / {ro['outside']/n2:.3f} / {ro['nothing']/n2:.3f} (counts {ro['inside']}/{ro['outside']}/{ro['nothing']} of {n2})",
               "record": "0.04 / 0.22 / 0.74", "where": "RESULTS.md random-points table",
               "match": [round(ro[k] / n2, 2) for k in ("inside", "outside", "nothing")] == [0.04, 0.22, 0.74]})
    rc = d["rp"][("C", "inside")]
    ck.append({"quantity": "random points, control arm, inside a string: says inside / outside / nothing (fixed statement)",
               "computed": f"{rc['inside']/rc['n']:.3f} / {rc['outside']/rc['n']:.3f} / {rc['nothing']/rc['n']:.3f}; strict lib23 regex named_when_true {ev['C']['random_points']['strict_regex']['named_when_true']:.3f}",
               "record": "0.02 / 0.00 / 0.98", "where": "RESULTS.md random-points table",
               "match": rc["inside"] / rc["n"] >= 0.015})
    acc = d["rp"][("S", "acc")]
    ck.append({"quantity": "random points arm S statement check total (correct / wrong / none of 500)",
               "computed": f"{acc['correct']} / {acc['wrong']} / {acc['none']}", "record": "eval30_S statement_check 92 / 47 / 361",
               "where": "results/eval30_S.json", "match": (acc["correct"], acc["wrong"], acc["none"]) == (92, 47, 361)})
    tw = {a: d["train"][a]["wall_s"] for a in ("S", "C")}
    ck.append({"quantity": "training wall time", "computed": f"train30 wall_s S {tw['S']:.0f} s, C {tw['C']:.0f} s (accumulates across resumes/stages; pipeline.log shows about 13 min per arm)",
               "record": "table30.md prints 35699 s and 36492 s as training time; RESULTS says 13 minutes each", "where": "table30.md; pipeline.log",
               "match": False})
    return [{**c, "match": bool(c["match"])} for c in ck]


def main_fig(d: dict) -> None:
    with plt.rc_context(style.SLIDE_RC):
        _main_fig(d)


def _main_fig(d: dict) -> None:
    fs = style.MIN_PT
    GREY = "#EEF1F3"
    fig = plt.figure(figsize=(8.8, 4.15))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.9, hspace=0.75,
                          left=0.31, right=0.975, top=0.9, bottom=0.25)
    ax = fig.add_subplot(gs[:, 0])
    ypos, y = [], 0.0
    for i in range(len(ROWS)):
        if i in GROUP_BREAKS:
            y += 0.85
        ypos.append(y)
        y += 1
    ypos = np.array(ypos)
    for yy, (row, lab) in zip(ypos, ROWS):
        r = d["rows"][("S", row)]
        n = r["n"]
        c, w, z = r["correct"] / n, r["wrong"] / n, r["none"] / n
        ax.barh(yy, c, color=C["nla"], height=0.74)
        ax.barh(yy, w, left=c, color=C["nla_pale"], height=0.74)
        ax.barh(yy, z, left=c + w, color=GREY, height=0.74)
        k = r["correct"] + r["wrong"]
        if k:
            p, lo, hi = style.wilson(r["correct"], k)
            ax.text(1.03, yy, f"{r['correct']}/{k}", va="center", ha="left", fontsize=fs,
                    color=C["ink"] if p >= 0.75 else C["muted"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in ROWS])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(ypos[-1] + 0.5, ypos[0] - 1.25)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("share of 32 held-out items per row")
    ax.set_title("naming adapter, held-out files", loc="left", x=-0.02)
    ax.text(1.03, ypos[0] - 0.5, "right", fontsize=fs, color=C["muted"], ha="left", va="bottom", style="italic")
    for i, t in ((0, "probe reads it"), (4, "probe barely reads it"), (6, "untrained rows")):
        ax.text(-0.02, ypos[i] - 0.5, t, fontsize=fs, color=C["muted"], ha="right", va="bottom", style="italic")
    hand = [plt.Rectangle((0, 0), 1, 1, color=C["nla"]), plt.Rectangle((0, 0), 1, 1, color=C["nla_pale"]),
            plt.Rectangle((0, 0), 1, 1, color=GREY, ec=C["line"])]
    fig.legend(hand, ["names the true value", "names the wrong value", "names nothing"], loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=3, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    # gate reconstruction: horizontal bars
    ax2 = fig.add_subplot(gs[0, 1])
    ev = d["ev"]
    g = [ev[a]["gate"]["fve"] for a in ("base", "C", "S")]
    yb = [0, 1, 2]
    ax2.barh(yb, g, color=[C["nla_pale"], C["nla_pale"], C["nla"]], height=0.72)
    for yy, v in zip(yb, g):
        ax2.text(v + 0.015, yy, f"{v:.3f}", va="center", ha="left", fontsize=fs)
    ax2.axvline(0.761, color=C["line"], lw=1.4, ls="--")
    ax2.set_yticks(yb)
    ax2.set_yticklabels(["base", "control", "naming"])
    ax2.tick_params(axis="y", length=0)
    ax2.set_ylim(2.5, -0.6)
    ax2.set_xlim(0, 0.9)
    ax2.set_xticks([0, 0.761])
    ax2.set_xticklabels(["0", "target 0.761"])
    ax2.get_xticklabels()[1].set_horizontalalignment("right")
    ax2.set_title("cost: reconstruction (FVE)", loc="left", x=-0.42, pad=24)
    m, lo, hi = d["fve_diff"]
    ax2.text(1.0, 1.0, f"naming - control {m:+.3f}", transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=fs, color=C["ink"])
    # random points
    ax3 = fig.add_subplot(gs[1, 1])
    conds = (("inside", "in string"), ("outside", "not in"))
    for j, (cond, lab) in enumerate(conds):
        r = d["rp"][("S", cond)]
        n = r["n"]
        left = 0.0
        wrong = "outside" if cond == "inside" else "inside"
        for key, col in ((cond, C["nla"]), (wrong, C["nla_pale"]), ("nothing", GREY)):
            v = r[key] / n
            ax3.barh(j, v, left=left, color=col, height=0.72)
            left += v
        ax3.text(0.98, j, f"'outside' {r['outside']/n:.2f}", va="center", ha="right", fontsize=fs, color=C["ink"])
    ax3.set_ylim(1.5, -0.5)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels([f"{lab}\n(n={d['rp'][('S', c)]['n']})" for c, lab in conds], linespacing=1.0)
    ax3.tick_params(axis="y", length=0)
    ax3.set_xlim(0, 1)
    ax3.set_xticks([0, 0.5, 1])
    ax3.set_xlabel("share of random read points", labelpad=2)
    ax3.set_title("random read points (exp 23)", loc="left", x=-0.42)
    caption = ("A 400-step supervised adapter (continuing the NLA's RL LoRA) trained to append one true statement of the "
               "row's state; control arm = same explanations without the statement. 'Names' = the readout contains the "
               "filled statement template (string match, no judge). Left: naming arm on held-out source files (32 items "
               "per row); base and control name nothing on any row. 'right' = correct among named, Wilson 95% intervals "
               "wide at these n (e.g. 15/19: 0.57-0.91). Top right: gate reconstruction (FVE) on 192 held-out items "
               f"(single file values; naming minus control per-item paired difference {m:+.3f}, bootstrap 95% "
               f"[{lo:+.3f}, {hi:+.3f}]); dashed line = pre-registered target 0.761. Bottom right: exp 23's 500 random read "
               "points, recounted from the saved readouts; the adapter names 'outside' about as often whatever the truth.")
    style.save(fig, "s8_naming_lora",
               title="Naming costs the reward nothing and is right where the state is readable, but does not generalise",
               caption=caption, section="S8", role="main",
               sources=["experiments/30_naming_lora/results/eval30_base.json", "experiments/30_naming_lora/results/eval30_C.json",
                        "experiments/30_naming_lora/results/eval30_S.json", "experiments/30_naming_lora/results/train30_S.json",
                        "experiments/30_naming_lora/results/train30_C.json", "experiments/23_random_points/results/items_base.json"],
               checks=checks_for(d),
               notes=("Statement templates (exp 22 FIXED nla), e.g. 'The current position is inside a string literal' / "
                      "'... outside a string literal'; a readout 'names' a value when it contains the filled template "
                      "(lib30.statement_accuracy). Record discrepancies: (1) random points, statement arm, inside a string: "
                      "recount 16/32/86 of 134 = 0.12/0.24/0.64 vs RESULTS/D49 0.16/0.24/0.60; not-in-string recount "
                      "0.04/0.21/0.75 vs 0.04/0.22/0.74; control arm recount 0.00/0.00/1.00 vs 0.02/0.00/0.98 (the 0.02 "
                      "looks like lib23's strict regex rate 0.03, not the statement check). (2) P1's second clause (both "
                      "arms within 0.03 of 0.761) is unscored and fails: gate FVE is 0.51, about 0.25 below. (3) "
                      "train30 wall_s (35699 s, 36492 s) accumulates across stages/resumes; pipeline.log shows about 13 "
                      "minutes per arm; table30.md prints the accumulated numbers as training time. Wall time is not "
                      "plotted. P2-P4 fail as defined; P5 unscorable."))


if __name__ == "__main__":
    data = load()
    main_fig(data)
    for c in checks_for(data):
        print(("OK  " if c["match"] else "DIFF"), c["quantity"], "|", c["computed"], "|", c["record"])
