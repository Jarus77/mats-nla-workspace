"""Diagram: the one-picture answer (closing slide). Numbers from exps 27, 29, 30 raw JSON."""
from __future__ import annotations

import json

from g6_common import C, ROOT, SOFT, WHITE, arrow, box, canvas, text
from g6_d_readers import activation_icon
from style import save

NAME = "s8_d_thesis"
E27 = ROOT / "experiments/27_random_2000/results"
E29 = ROOT / "experiments/29_trajectory/results"
E30 = ROOT / "experiments/30_naming_lora/results"
STRUCT = ["py_string", "js_string", "latex_math", "py_arg", "json_depth", "md_heading", "md_list"]
RL = ["av_base"] + [f"iter_{s:06d}" for s in range(100, 900, 100)]


def load():
    t27 = json.loads((E27 / "table27.json").read_text())
    j27 = json.loads((E27 / "judge27.json").read_text())["rates"]
    carry = t27["carrying"]
    t29 = json.loads((E29 / "table29.json").read_text())
    S = json.loads((E30 / "eval30_S.json").read_text())
    Cc = json.loads((E30 / "eval30_C.json").read_text())
    ol_pos = sum(j27[k]["ol"]["n_pos"] for k in ("e27", "e23"))
    ol_yes = sum(j27[k]["ol"]["yes_pos"] for k in ("e27", "e23"))
    nla_pos = sum(j27[k]["nla"]["n_pos"] for k in ("e27", "e23"))
    nla_yes = sum(j27[k]["nla"]["yes_pos"] for k in ("e27", "e23"))
    st = [t29["states"][c][r] for c in RL for r in STRUCT]
    readable_rows = ["py_string", "py_arg", "json_depth", "code_lang"]
    cond = [S["rows"][r]["correct"] / (S["rows"][r]["correct"] + S["rows"][r]["wrong"]) for r in readable_rows]
    return {
        "carry": carry, "ol_yes": ol_yes, "ol_pos": ol_pos, "nla_rate": nla_yes / nla_pos, "nla_pos": nla_pos,
        "st_min": min(st), "st_max": max(st),
        "lang0": t29["states"]["av_base"]["code_lang"], "lang800": t29["states"]["iter_000800"]["code_lang"],
        "fve0": t29["fve"]["av_base"]["py_string"], "fve800": t29["fve"]["iter_000800"]["py_string"],
        "fveS": S["gate"]["fve"], "fveC": Cc["gate"]["fve"], "cond_min": min(cond), "cond_max": max(cond),
    }


def find_carry(carry):
    """Code-only in-string carrying row: (readable, ol, nla)."""
    def walk(o):
        if isinstance(o, dict):
            yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
    for d in walk(carry):
        if str(d.get("kind", "")).startswith("code") and d.get("state") == "in_string":
            return d
    return None


def main():
    n = load()
    row = find_carry(n["carry"])
    keys = {k: row[k] for k in row if isinstance(row[k], (int, float))} if row else {}
    rd = keys.get("readable", keys.get("probe"))
    ol = keys.get("ol_text_bag")
    nl = keys.get("nla_text_bag")

    fig, ax = canvas()  # 92.4 x 42 units

    # carry box: what the readers' words hold
    box(ax, 0.5, 6, 21.5, 31, fc=SOFT, ec=C["probe"], lw=2.0)
    text(ax, 11.25, 32.6, "readers' words\ncarry the state", fs=15, weight="bold", color=C["probe"], linespacing=1.1)
    text(ax, 1.5, 24.4, "in a code string,\nAUROC from text:", fs=15, ha="left", linespacing=1.1)
    text(ax, 1.5, 17.0, f"activation {rd:.2f}", fs=15, ha="left", color=C["probe"], weight="bold")
    text(ax, 1.5, 13.6, f"NLA {nl:.2f}", fs=15, ha="left", color=C["nla"], weight="bold")
    text(ax, 1.5, 10.2, f"Oracle Lens {ol:.2f}", fs=15, ha="left", color=C["ol"], weight="bold")

    # gate: language passes, structure is held back
    gx = 29.6
    arrow(ax, (22.3, 26), (gx - 0.8, 31), color=C["muted"], lw=2.2, ms=16)
    arrow(ax, (22.3, 17), (gx - 0.8, 11), color=C["muted"], lw=2.2, ms=16)
    for y, closed in ((31, False), (11, True)):
        ax.plot([gx, gx], [y + 2.4, y + 4.2], color=C["ink"], lw=5, solid_capstyle="butt")
        ax.plot([gx, gx], [y - 4.2, y - 2.4], color=C["ink"], lw=5, solid_capstyle="butt")
        if closed:
            ax.plot([gx, gx], [y - 2.4, y + 2.4], color=C["nla"], lw=6, solid_capstyle="butt")
    text(ax, gx, 21, "gate", fs=15, weight="bold", color=C["ink"])
    arrow(ax, (gx + 0.8, 31), (35.2, 31), color=C["probe"], lw=2.4, ms=16)
    arrow(ax, (gx + 0.8, 11), (35.2, 11), color=C["nla"], lw=2.4, ms=16, ls="--")

    # outcomes
    bx0, bw = 35.4, 24.8
    text(ax, bx0, 39.6, "language", fs=15, weight="bold", color=C["probe"], ha="left")
    box(ax, bx0, 24, bw, 13, fc=WHITE, ec=C["probe"], lw=1.8)
    text(ax, bx0 + bw / 2, 34.4, "stated", fs=15, weight="bold", color=C["probe"])
    text(ax, bx0 + bw / 2, 28.6, f"NLA, code language\n{n['lang0']:.2f} → {n['lang800']:.2f} over RL", fs=15,
         linespacing=1.1)
    box(ax, bx0, 3.6, bw, 14.6, fc=WHITE, ec=C["nla"], lw=1.8)
    text(ax, bx0 + bw / 2, 15.6, "rarely stated", fs=15, weight="bold", color=C["nla"])
    text(ax, bx0 + bw / 2, 9.0, f"inside a string:\nOracle Lens {n['ol_yes']} of {n['ol_pos']},\nNLA {n['nla_rate']:.2f}",
         fs=15, linespacing=1.1)
    text(ax, bx0, 1.5, "structure", fs=15, weight="bold", color=C["nla"], ha="left")

    # what the gate is
    rx = 61.4
    box(ax, rx, 1.0, 92.2 - rx, 40.0, fc=SOFT, ec=C["ink"], lw=2.0)
    text(ax, (rx + 92.2) / 2, 37.2, "the gate is the\nwarm-start habit", fs=15, weight="bold", linespacing=1.1)
    text(ax, rx + 1.0, 31.0, "not the reward:", fs=15, weight="bold", ha="left", color=C["nla"])
    text(ax, rx + 1.0, 29.4, va="top", s=f"structure stated {n['st_min']:.2f}-{n['st_max']:.2f},\nflat over 800 RL steps;\n"
         f"reconstruction {n['fve0']:.2f} → {n['fve800']:.2f}", fs=15, ha="left", linespacing=1.1)
    text(ax, rx + 1.0, 16.0, "not capacity:", fs=15, weight="bold", ha="left", color=C["nla"])
    text(ax, rx + 1.0, 14.4, va="top", s=f"an adapter names it;\nreconstruction {n['fveS']:.3f}\n"
         f"(control {n['fveC']:.3f});\ncorrect {n['cond_min']:.2f}-{n['cond_max']:.2f}", fs=15, ha="left",
         linespacing=1.1)
    arrow(ax, (rx, 21), (gx + 3.0, 21), color=C["ink"], lw=1.6, ls="--", ms=14)

    checks = [
        {"quantity": "exp 27 code in-string carrying (readable / NLA / Oracle Lens)", "computed": f"{rd} / {nl} / {ol}",
         "record": "0.86 / 0.81 / 0.72 (talk script S6 update)", "where": "exp 27 table27", "match":
         (round(rd, 2), round(nl, 2), round(ol, 2)) == (0.86, 0.81, 0.72) if rd is not None else False},
        {"quantity": "Oracle Lens judge states in-string (pooled e27+e23)", "computed": f"{n['ol_yes']} of {n['ol_pos']}",
         "record": "none of 598", "where": "talk script S6", "match": n["ol_yes"] == 0 and n["ol_pos"] == 598},
        {"quantity": "NLA judge states in-string (pooled)", "computed": f"{n['nla_rate']:.3f} of {n['nla_pos']}",
         "record": "0.22", "where": "talk script S6", "match": round(n["nla_rate"], 2) == 0.22},
        {"quantity": "exp 29 structural states range, av_base..iter 800, 7 structural rows (judge)",
         "computed": f"{n['st_min']:.2f}-{n['st_max']:.2f}", "record": "0.01-0.21 activation warm start, 0.00-0.29 RL",
         "where": "decisions.md D50", "match": round(n["st_min"], 2) == 0.00 and round(n["st_max"], 2) == 0.29},
        {"quantity": "exp 29 code-language states rate av_base -> iter 800", "computed": f"{n['lang0']:.3f} -> {n['lang800']:.3f}",
         "record": "0.55 to 0.75 (talk script S8, D50)", "where": "talk script S8", "match":
         round(n["lang800"], 2) == 0.75, "note": "raw 0.759 rounds to 0.76; record says 0.75 (minor)"},
        {"quantity": "exp 29 reconstruction (FVE), inside a Python string, av_base -> iter 800", "computed": f"{n['fve0']:.3f} -> {n['fve800']:.3f}",
         "record": "0.18 to 0.59", "where": "talk script S8", "match": (round(n['fve0'], 2), round(n['fve800'], 2)) == (0.18, 0.59)},
        {"quantity": "exp 30 gate reconstruction (FVE), statement arm vs control", "computed": f"{n['fveS']:.4f} vs {n['fveC']:.4f}",
         "record": "0.507 vs 0.512", "where": "exp 30 RESULTS", "match": (round(n['fveS'], 3), round(n['fveC'], 3)) == (0.507, 0.512)},
        {"quantity": "exp 30 correct when it names, readable rows (Python string, call argument, JSON nesting depth, code language)",
         "computed": f"{n['cond_min']:.2f}-{n['cond_max']:.2f}", "record": "statements correct 0.79 to 1.00",
         "where": "talk script S8", "match": (round(n['cond_min'], 2), round(n['cond_max'], 2)) == (0.79, 1.00),
         "note": "correct / (correct + wrong); unconditional accuracy is 0.28-0.69"},
    ]
    save(fig, NAME, title="The readers carry structure but a warm-start habit keeps it unsaid",
         caption=("Both readers' outputs carry position-in-text state (decodable from the text alone at random read "
                  "points inside code strings, exp 27), but only language-like properties get stated (the judge's "
                  "verdict on free readouts; exps 27 and 29). Along the NLA's own RL run the structural states rate "
                  "stays flat while reconstruction (FVE, inside a Python string) rises (exp 29), and a 400-step "
                  "supervised adapter makes it name the state (a fixed-statement match) without costing "
                  "reconstruction, correct when it names (exp 30): the gate is the warm-start register, not the "
                  "reward or the reader's capacity. The readers capture position-in-text state and do not say it."),
         section="S8", role="main",
         sources=[str(p.relative_to(ROOT)) for p in [E27 / "table27.json", E27 / "judge27.json", E29 / "table29.json",
                                                    E30 / "eval30_S.json", E30 / "eval30_C.json"]],
         checks=checks,
         notes=("Terms: 'stated' is the judge's verdict (exp 27 blind judge, exp 29 states judge); 'names' in exp 30 is "
                "the readout containing the row's fixed statement of either value. "
                "'correct' in exp 30 is conditional on the adapter naming the state (held-out clusters, n = 32 per row); "
                "transfer to held-out rows and random points is weak (exp 30 P2-P4 failed). Text-only warm start "
                "(TextNLA) names more (0.17-0.49), which is why the gate is attributed to the warm-start register."))


if __name__ == "__main__":
    main()
