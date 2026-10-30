"""s6_random_points (exps 23, 27; exp 30 recount as a check): at random read points with nothing prompted, the
readers' readouts carry the string state almost as well as the activation does, and Oracle Lens never states it."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_common as G  # noqa: E402
import style  # noqa: E402
from style import C, plt  # noqa: E402

E23 = G.EXP / "23_random_points" / "results"
E27 = G.EXP / "27_random_2000" / "results"
E30 = G.EXP / "30_naming_lora" / "results"
sys.path.insert(0, str(G.EXP / "23_random_points"))
from lib23 import names_state  # noqa: E402

STATES = {"in_string": lambda v: bool(v), "in_math": lambda v: bool(v), "paren_depth": lambda v: v > 0,
          "brace_depth": lambda v: v >= 2, "list_depth": lambda v: v > 0, "heading_level": lambda v: v > 0}


def judge_rates(checks):
    import json

    rows = [json.loads(line) for line in open(E27 / "judged27.jsonl")]
    out = {}
    for reader in ("ol", "nla"):
        for lab in (True, False):
            v = [r["verdict"] for r in rows if r["reader"] == reader and r["label"] is lab and r["verdict"] is not None]
            k, n = int(sum(v)), len(v)
            out[(reader, lab)] = (k, n, *style.wilson(k, n))
    rec = G.load(E27 / "judge27.json")["rates"]["pooled"]
    for reader in ("ol", "nla"):
        checks.append(G.check(f"exp27+23 judge {G.READER[reader]} yes when in string (pooled)", out[(reader, True)][2], round(rec[reader]["rate_pos"], 4), "exp27 judge27.json pooled"))
        checks.append(G.check(f"exp27+23 judge {G.READER[reader]} n in string", out[(reader, True)][1], rec[reader]["n_pos"], "exp27 judge27.json pooled", tol=0))
        checks.append(G.check(f"exp27+23 judge {G.READER[reader]} yes when not in string", out[(reader, False)][2], round(rec[reader]["rate_neg"], 4), "exp27 judge27.json pooled"))
    checks.append(G.check("judge NLA in-string rate (2dp)", round(out[("nla", True)][2], 2), 0.22, "talk script S6 update / exp27 RESULTS"))
    checks.append(G.check("judge Oracle Lens in-string yes count", out[("ol", True)][0], 0, "talk script 'Oracle Lens on none of 598'", tol=0))
    checks.append(G.check("judge Oracle Lens in-string n", out[("ol", True)][1], 598, "talk script '598'", tol=0))
    return out


def regex_counts(items, ol, nla, checks):
    out = {}
    for st, fn in STATES.items():
        hold = [it for it in items if fn(it["meta"]["labels"][st])]
        rest = [it for it in items if not fn(it["meta"]["labels"][st])]
        for name, d in (("ol", ol), ("nla", nla)):
            out[(st, name)] = (sum(names_state(d[it["id"]], st) for it in hold), len(hold),
                               sum(names_state(d[it["id"]], st) for it in rest), len(rest))
    ol_any = sum(any(names_state(ol[it["id"]], st) for st in STATES) for it in items)
    ol_hold = sum(out[(st, "ol")][0] for st in STATES)
    checks.append(G.check("exp27 Oracle Lens readouts naming any state by strict regex (of 1,939)", ol_any, 0,
                          "exp27 RESULTS reading 2 '0 of 1,939 by the regex, on every state'", tol=0))
    checks.append(G.check("exp27 Oracle Lens namings when the state holds, summed over six states", ol_hold, 0,
                          "exp27 RESULTS reading 2 (implied)", tol=0))
    return out, ol_any


def carrying(items, ol, nla, acts, kinds, state, seed=0):
    fn = STATES[state]
    idx = [i for i, it in enumerate(items) if it["meta"]["kind"] in kinds]
    y = np.array([fn(items[i]["meta"]["labels"][state]) for i in idx], int)
    g = np.array([items[i]["cluster"] for i in idx])
    res = {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    s = G.oof_scores(acts["acts"][idx].float(), y, g, k=5, lam=1e-3)
    res["readable"] = G.auroc_ci(s, y, g, seed=seed)
    for name, d in (("ol", ol), ("nla", nla)):
        s = G.text_bag_scores([d[items[i]["id"]] for i in idx], y, g)
        res[name] = G.auroc_ci(s, y, g, seed=seed)
    return res


def exp30_recount(checks):
    items = G.load(E23 / "items_base.json")
    lab = {it["id"]: bool(it["meta"]["labels"]["in_string"]) for it in items}
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()  # noqa: E731
    pair = {"inside": norm("The current position is inside a string literal"),
            "outside": norm("The current position is outside a string literal")}
    out = {}
    for arm, f in (("base", "eval30_base.json"), ("control", "eval30_C.json"), ("statement", "eval30_S.json")):
        rd = G.load(E30 / f)["random_points"]["readouts"]
        for holds in (True, False):
            ids = [i for i in rd if lab[i] is holds]
            ins = [pair["inside"] in norm(rd[i]) for i in ids]
            outs = [pair["outside"] in norm(rd[i]) for i in ids]
            n = len(ids)
            r = (sum(a and not b for a, b in zip(ins, outs)) / n, sum(b and not a for a, b in zip(ins, outs)) / n,
                 sum(a and b for a, b in zip(ins, outs)) / n, sum(not a and not b for a, b in zip(ins, outs)) / n)
            out[(arm, holds)] = (n, r)
    s_in = out[("statement", True)][1]
    checks.append(G.check("exp30 statement arm, in-string points: names inside only (substring)", s_in[0], 0.16, "exp30 RESULTS random-point table", tol=0.005))
    checks.append(G.check("exp30 statement arm, in-string points: names outside only (substring)", s_in[1], 0.24, "exp30 RESULTS random-point table", tol=0.005))
    checks.append(G.check("exp30 statement arm, in-string points: names neither (substring)", s_in[3], 0.60, "exp30 RESULTS random-point table", tol=0.005))
    s_out = out[("statement", False)][1]
    for q, v, r in (("inside only", s_out[0], 0.04), ("outside only", s_out[1], 0.22), ("neither", s_out[3], 0.74)):
        checks.append(G.check(f"exp30 statement arm, not-in-string points: names {q} (substring)", v, r, "exp30 RESULTS random-point table", tol=0.005))
    c_in = out[("control", True)][1]
    checks.append(G.check("exp30 control arm, in-string points: names inside only (substring)", c_in[0], 0.02, "exp30 RESULTS random-point table", tol=0.005))
    return out


def main():
    import torch

    checks = []
    sources = [G.rel(E27 / f) for f in ("judged27.jsonl", "judge27.json", "items_base.json", "ol27.json", "nla27.json", "vol/acts27.pt", "table27.json")]
    sources += [G.rel(E23 / f) for f in ("items_base.json", "ol23.json", "nla23.json", "vol/acts23.pt", "table23.json")]
    sources += [G.rel(E30 / f) for f in ("eval30_base.json", "eval30_C.json", "eval30_S.json")]
    jr = judge_rates(checks)
    items27 = G.load(E27 / "items_base.json")
    ol27, nla27 = G.load(E27 / "ol27.json")["ol"], G.load(E27 / "nla27.json")["nla"]
    rx, ol_any = regex_counts(items27, ol27, nla27, checks)
    acts27 = torch.load(E27 / "vol" / "acts27.pt", weights_only=False, map_location="cpu")
    assert acts27["ids"] == [it["id"] for it in items27]
    c27 = carrying(items27, ol27, nla27, acts27, ("py", "js"), "in_string")
    items23 = G.load(E23 / "items_base.json")
    ol23, nla23 = G.load(E23 / "ol23.json")["ol"], G.load(E23 / "nla23.json")["nla"]
    acts23 = torch.load(E23 / "vol" / "acts23.pt", weights_only=False, map_location="cpu")
    c23 = carrying(items23, ol23, nla23, acts23, ("py", "js"), "in_string")
    t27 = {(r["kind"], r["state"]): r for r in G.load(E27 / "table27.json")["carrying"]}[("code (py+js)", "in_string")]
    t23 = {(r["kind"], r["state"]): r for r in G.load(E23 / "table23.json")["carrying"]}[("code (py+js)", "in_string")]
    for tag, c, t in (("exp27", c27, t27), ("exp23", c23, t23)):
        checks.append(G.check(f"{tag} code in-string readable", c["readable"][0], round(t["readable"], 4), f"{tag} table json", tol=0.001))
        checks.append(G.check(f"{tag} code in-string Oracle Lens text bag", c["ol"][0], round(t["ol_text_bag"], 4), f"{tag} table json", tol=0.001))
        checks.append(G.check(f"{tag} code in-string NLA text bag", c["nla"][0], round(t["nla_text_bag"], 4), f"{tag} table json", tol=0.001))
    for q, v, r in (("readable", c27["readable"][0], 0.86), ("ol", c27["ol"][0], 0.72), ("nla", c27["nla"][0], 0.81)):
        checks.append(G.check(f"exp27 code in-string {q} (2dp)", round(v, 2), r, "talk script S6 update (0.86 / 0.81 / 0.72)"))
    for q, v, r in (("readable", c23["readable"][0], 0.86), ("ol", c23["ol"][0], 0.72), ("nla", c23["nla"][0], 0.79)):
        checks.append(G.check(f"exp23 code in-string {q} (2dp)", round(v, 2), r, "talk script S6 exhibit (0.86 / 0.72 / 0.79)"))
    # exp 23 talk-script naming line: "Oracle Lens 0 of 500"
    rx23, ol_any23 = regex_counts(items23, ol23, nla23, [])
    checks.append(G.check("exp23 Oracle Lens readouts naming any state by strict regex (of 500)", ol_any23, 0, "talk script S6 'Oracle Lens 0 of 500' / exp23 RESULTS reading 1", tol=0))
    r30 = exp30_recount(checks)

    # ---- figure ----
    style.slide_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=G.FIG, gridspec_kw=dict(width_ratios=[1, 1.25]), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    x = np.arange(2)
    w = 0.40
    for j, (lab, alpha_fill) in enumerate(((True, 1.0), (False, 0.0))):
        for i, (reader, col, pale) in enumerate((("ol", C["ol"], C["ol_pale"]), ("nla", C["nla"], C["nla_pale"]))):
            k, n, p, lo, hi = jr[(reader, lab)]
            xx = i + (j - 0.5) * w
            b = a1.bar(xx, p, w * 0.92, color=col if lab else pale, edgecolor=col, lw=1.5)
            a1.errorbar(xx, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=C["ink"], elinewidth=1.3, capsize=4)
            if reader == "nla":
                a1.text(xx, hi + 0.008, f"{k}/{n}", ha="center", va="bottom", fontsize=15)
    a1.set_xticks(x)
    a1.set_xticklabels(["Oracle Lens", "NLA"])
    a1.set_ylim(0, 0.34)
    a1.set_ylabel("share stating 'inside a\nstring' (blind judge)")
    a1.set_title("states it", loc="left")
    h1 = plt.Rectangle((0, 0), 1, 1, color=C["muted"])
    h2 = plt.Rectangle((0, 0), 1, 1, facecolor=C["line"], edgecolor=C["muted"])
    a1.legend([h1, h2], ["inside a string", "not inside"], loc="upper left", handlelength=1.2, handletextpad=0.4,
               borderaxespad=0.1)
    # carrying
    labels = ["activation\n(probe)", "NLA\nwords", "Oracle Lens\nwords"]
    keys = ["readable", "nla", "ol"]
    cols = [C["probe"], C["nla"], C["ol"]]
    pales = [C["line"], C["nla_pale"], C["ol_pale"]]
    for i, (k, col, pale) in enumerate(zip(keys, cols, pales)):
        for j, (c, tag, fc) in enumerate(((c23, "exp 23 (500)", pale), (c27, "exp 27 (1,939)", col))):
            a, lo, hi = c[k]
            xx = i + (j - 0.5) * w
            a2.bar(xx, a - 0.5, w * 0.92, bottom=0.5, color=fc, edgecolor=col, lw=1.2)
            a2.errorbar(xx, a, yerr=[[a - lo], [hi - a]], fmt="none", ecolor=C["ink"], elinewidth=1.3, capsize=4)
            a2.text(xx, 0.515, f"{a:.2f}", ha="center", va="bottom", fontsize=15,
                    color="white" if j == 1 else C["ink"])
    a2.set_xticks(range(3))
    a2.set_xticklabels(labels)
    a2.set_ylim(0.5, 1.04)
    a2.set_ylabel("AUROC (0.5 = chance)")
    a2.set_title("carries it: in a string or not", loc="left")
    hp = plt.Rectangle((0, 0), 1, 1, color=C["line"], ec=C["muted"])
    hd = plt.Rectangle((0, 0), 1, 1, color=C["muted"])
    a2.legend([hp, hd], [f"exp 23 ({c23['n_pos']} vs {c23['n_neg']})", f"exp 27 ({c27['n_pos']} vs {c27['n_neg']})"],
              loc="upper right", ncol=1, handlelength=1.2, handletextpad=0.4, borderaxespad=0.1)
    a1.annotate("0/598 inside\n0/604 not", (0.0, 0.004), xytext=(0.05, 0.085), fontsize=15, color=C["ol"],
                fontweight="bold", ha="center", arrowprops=dict(arrowstyle="->", color=C["ol"], lw=1.2))

    s_in, s_out = r30[("statement", True)], r30[("statement", False)]
    style.save(
        fig, "s6_random_points",
        title="At random read points Oracle Lens never states 'inside a string', yet its words carry the state",
        caption=("Random read points in random public documents, nothing prompted, one readout per reader. Left: share of "
                 "readouts a blind judge (readout only) says state the position is inside a string literal, exps 27 + 23 "
                 "pooled, every in-string point and matched negatives; Wilson 95%; unparsed verdicts dropped. Right: "
                 "in-string vs not on the Python + JavaScript points, AUROC of a probe on the layer-42 activation and of a "
                 "bag-of-tokens classifier on each reader's readout text alone; grouped 5-fold by source file, cluster-"
                 "bootstrap 95% over files. Caveat: by the strict regex the NLA names the string state on only 0.02 of "
                 "in-string readouts and Oracle Lens on none; the judge's 0.22 (states) counts the NLA's own wording "
                 "('mid-quotation')."),
        section="S6", role="main", sources=sources, checks=checks,
        notes=("Judge: google/gemini-3.8-flash T=0 (exp 27 judge27.py prompt: 'does the text STATE that the current position "
               "... is inside a string literal ...'). Strict-regex naming at exp 27 (lib23.names_state), counts named-when-"
               "holds / n holds, named-when-not / n not: " + "; ".join(
                   f"{st} {nm}: {v[0]}/{v[1]}, {v[2]}/{v[3]}" for (st, nm), v in rx.items())
               + f". Oracle Lens readouts naming any of the six states at exp 27: {ol_any} of 1,939; at exp 23: {ol_any23} of 500. "
               "Exp 30 recount (naming LoRA, exp 23's 500 points, statement-pair substring; inside-only / outside-only / both / "
               "neither): " + "; ".join(f"{arm} {'in-string' if h else 'not'} (n={n}): " + "/".join(f"{x:.3f}" for x in r)
                                         for (arm, h), (n, r) in r30.items())),
    )
    print("judge", jr)
    print("regex", rx, ol_any, ol_any23)
    print("c27", c27, "c23", c23)
    print("r30", r30)
    print("mismatch", [c for c in checks if not c["match"]])


if __name__ == "__main__":
    main()
