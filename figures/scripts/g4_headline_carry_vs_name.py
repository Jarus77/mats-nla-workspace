"""headline_carry_vs_name: on the independent public bank (exps 22 + 25), every structural state the activation holds
is carried by both readers' readouts (x) and almost never stated (y); only the language control is stated."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_common as G  # noqa: E402
import style  # noqa: E402
from style import C, plt  # noqa: E402

E22 = G.EXP / "22_public_bank" / "results"
E25 = G.EXP / "25_public_exhibits" / "results"
ROWS = ["py_string", "js_string", "latex_math", "py_arg", "json_depth", "md_heading", "md_list", "code_lang"]
N1 = N0 = 80  # selection split, 80 per value per row (exp 22 items_base)


def gather():
    sources, checks, data = [], [], {}
    score25 = G.load(E25 / "score25.json")
    obs25 = {r["feature"]: r["observed"] for r in score25["rows"]}
    sources.append(G.rel(E25 / "score25.json"))
    items = [it for it in G.load(E22 / "items_base.json") if it["split"] == "select"]
    sources.append(G.rel(E22 / "items_base.json"))
    for row in ROWS:
        sel = [it for it in items if it["feature"] == row]
        assert len(sel) == N1 + N0, (row, len(sel))
        rec = G.load(E25 / f"recon25_{row}.json")["auroc"]
        sources.append(G.rel(E25 / f"recon25_{row}.json"))
        d = {}
        for reader, key, jf in (("nla", "nla_readout", f"{row}_nla.json"), ("ol", "ol_readout", f"{row}_ol42.json")):
            a = rec[key]
            lo, hi = G.hanley_mcneil(a, N1, N0)
            j = G.load(E22 / "judged" / jf)
            sources.append(G.rel(E22 / "judged" / jf))
            flags = [bool(r["stated"]) for r in j["judged"]]
            grp = [r["item"] for r in j["judged"]]
            p, plo, phi = G.cluster_prop_ci(flags, grp)
            code = {j["values"][0]: "A", j["values"][1]: "B"}
            correct = np.mean([r["stated_value"] == code[r["value"]] for r in j["judged"]])
            d[reader] = dict(x=a, xlo=lo, xhi=hi, y=p, ylo=plo, yhi=phi, n_samples=len(flags),
                             n_items=len(set(grp)), n_stated=int(sum(flags)), stated_correct=float(correct))
            rkey = "nla_states" if reader == "nla" else "ol_states"
            checks.append(G.check(f"{row} {G.READER[reader]} stated rate (judge)", p, round(obs25[row][rkey], 4), "exp25 score25.json (= exp22 judge cell)"))
        checks.append(G.check(f"{row} Oracle Lens readout AUROC via critic", d["ol"]["x"], round(obs25[row]["ol_exhibits"], 4), "exp25 score25.json ol_exhibits"))
        data[row] = d
    # RESULTS.md table25 values (2 dp)
    rec_ol = {"py_string": 0.86, "js_string": 0.84, "latex_math": 0.87, "py_arg": 0.51, "json_depth": 0.97,
              "md_heading": 0.65, "md_list": 0.48, "code_lang": 0.98}
    rec_nla = {"py_string": 0.99, "js_string": 0.97, "latex_math": 1.00, "py_arg": 0.93, "json_depth": 0.99,
               "md_heading": 0.70, "md_list": 0.49, "code_lang": 0.98}
    for row in ROWS:
        checks.append(G.check(f"{row} Oracle Lens readout AUROC (2dp)", round(data[row]["ol"]["x"], 2), rec_ol[row], "exp25 RESULTS.md table"))
        checks.append(G.check(f"{row} NLA readout AUROC (2dp)", round(data[row]["nla"]["x"], 2), rec_nla[row], "exp25 RESULTS.md table"))
    checks.append(G.check("code language NLA stated (2dp)", round(data["code_lang"]["nla"]["y"], 2), 0.70, "exp22 RESULTS.md"))
    checks.append(G.check("Oracle Lens max stated over 8 rows (2dp)", round(max(data[r]["ol"]["y"] for r in ROWS), 2), 0.01, "exp22 RESULTS.md 'max 1%'"))
    checks.append(G.check("NLA max stated over 7 natural rows (2dp)", round(max(data[r]["nla"]["y"] for r in ROWS[:-1]), 2), 0.25, "exp22 RESULTS.md (json_depth 0.25)"))
    return data, sorted(set(sources)), checks


# group label for the cluster of readable structural rows (NLA points), drawn once with a bracket-free connector
CLUSTER = ("py_string", "js_string", "latex_math", "json_depth")


def draw(data):
    fig, ax = plt.subplots(figsize=G.FIG, layout="constrained")
    ax.set_xlim(0.40, 1.03)
    ax.set_ylim(-0.10, 0.84)
    ax.add_patch(plt.Rectangle((0.80, -0.10), 0.23, 0.44, color=C["ol_pale"], alpha=0.22, lw=0, zorder=0))
    ax.text(0.805, 0.345, "carried, not stated", color=C["ol"], fontsize=15, fontweight="bold", va="bottom")
    ax.axvline(0.5, color=C["line"], lw=1.5, ls="--", zorder=0)
    ax.text(0.497, 0.83, "chance", color=C["muted"], fontsize=15, va="top", ha="right")
    ax.axvline(0.8, color=C["line"], lw=1.5, ls="--", zorder=0)
    ax.text(0.803, 0.83, "0.80", color=C["muted"], fontsize=15, va="top")
    for reader, mk, col, pale, ms in (("ol", "D", C["ol"], C["ol_pale"], 9), ("nla", "o", C["nla"], C["nla_pale"], 10)):
        for row, d in data.items():
            md = row in G.MARKDOWN
            fc = "white" if md else col
            ec = pale if md else col
            ax.errorbar(d[reader]["x"], d[reader]["y"],
                        xerr=[[d[reader]["x"] - d[reader]["xlo"]], [d[reader]["xhi"] - d[reader]["x"]]],
                        yerr=[[d[reader]["y"] - d[reader]["ylo"]], [d[reader]["yhi"] - d[reader]["y"]]],
                        fmt="none", ecolor=pale, elinewidth=1.6, capsize=0, zorder=2, alpha=0.8)
            ax.plot(d[reader]["x"], d[reader]["y"], mk, ms=ms, mfc=fc, mec=ec, mew=2, zorder=4 if not md else 3)
    # one label for the string / math / JSON cluster of NLA points
    lx, ly = 0.952, 0.175
    ax.text(lx, ly, "strings, LaTeX math,\nJSON depth", ha="right", va="center", fontsize=15, color=C["ink"],
            linespacing=1.05, zorder=5)
    for row in CLUSTER:
        ax.plot([lx + 0.003, data[row]["nla"]["x"] - 0.006], [ly, data[row]["nla"]["y"]], color=C["muted"], lw=0.8, zorder=3)
    ca = data["py_arg"]
    ax.annotate("call argument: Oracle\nLens 0.51, NLA 0.93", (ca["ol"]["x"] + 0.004, ca["ol"]["y"] + 0.012),
                xytext=(0.53, 0.06), ha="left", va="bottom", fontsize=15, color=C["ol"], linespacing=1.05,
                arrowprops=dict(arrowstyle="-", color=C["ol"], lw=0.8))
    ax.text(ca["nla"]["x"], -0.04, "call argument", ha="center", va="top", fontsize=15, color=C["ink"])
    ax.annotate("Oracle Lens states nothing:\n0 to 0.01 on all eight rows",
                (0.843, 0.01), xytext=(0.53, 0.20), fontsize=15, color=C["ol"], linespacing=1.05, va="bottom",
                arrowprops=dict(arrowstyle="->", color=C["ol"], lw=1.2, shrinkB=6, relpos=(1.0, 0.3)))
    cl = data["code_lang"]["nla"]
    ax.annotate("code language (control): the one\nstate the NLA states, not structure",
                (cl["x"] - 0.006, cl["y"]), xytext=(0.935, 0.60), ha="right",
                fontsize=15, color=C["nla"], va="center", linespacing=1.05,
                arrowprops=dict(arrowstyle="->", color=C["nla"], lw=1.2))
    ax.text(0.41, 0.475, "Markdown rows (hollow): not linearly\npresent at layer 42, nothing to carry",
            fontsize=15, color=C["muted"], va="top", linespacing=1.05)
    ax.set_xlabel("carries it: state recovered from the readout (AUROC)")
    ax.set_ylabel("states it: share of readouts\n(blind judge)")
    h1 = ax.plot([], [], "o", ms=10, color=C["nla"], label="NLA explanation")[0]
    h2 = ax.plot([], [], "D", ms=9, color=C["ol"], label="Oracle Lens readout")[0]
    ax.legend(handles=[h1, h2], loc="upper left", bbox_to_anchor=(0.17, 1.0), handletextpad=0.2, borderaxespad=0.2)
    return fig


def main():
    data, sources, checks = gather()
    style.slide_style()
    fig = draw(data)
    style.save(
        fig, "headline_carry_vs_name",
        title="Both readers carry structural state in what they write and almost never state it",
        caption=("Each point is one row of the independent public bank, one reader. x: frozen-probe AUROC of the "
                 "critic's rebuild of the layer-42 vector from the reader's own free readout (exp 25, 160 items per row, "
                 "k = 1; bars are Hanley-McNeil 95% intervals on 80 vs 80, which ignore clustering). y: share of readouts "
                 "a blind judge says state the value (exp 22, 40 items x 8 samples on the mining split; bars are item-"
                 "bootstrap 95%, Wilson on 40 items when the rate is 0). Structural rows sit bottom-right for both readers; "
                 "the NLA states only the language control freely (0.70), Oracle Lens states nothing. Caveat: the critic is "
                 "the NLA's own (in distribution for the NLA column), and x and y are measured on different splits of the same bank."),
        section="headline", role="main", sources=sources, checks=checks,
        notes=("Carrying measure chosen: critic frozen-probe AUROC (exp 25), the measure covering the most rows of one bank "
               "for both readers including the language control; the text-bag and second-decoder checks are in "
               "s6_textbag_check and s6_two_checks. States = the judge's 'stated' verdict (any value), the record's measure; "
               "stated-correct rates per row: " + ", ".join(f"{r} NLA {data[r]['nla']['stated_correct']:.2f} / Oracle Lens {data[r]['ol']['stated_correct']:.2f}" for r in data)
               + ". Exp 22 judge prompts: see experiments/12_concept_recall/recall_lib.py (stated / exhibited). Readers: NLA "
               "actor prompt per notes/wsbench_internals.md ('You are a meticulous AI researcher ... <concept>...</concept> "
               "Please provide an explanation.'); Oracle Lens carrier 'An activation vector from layer 42 ... Produce distinct "
               "concepts that encode this activation, each as a '- ' bullet on its own line.' Per-point values: "
               + "; ".join(f"{r}: NLA x {data[r]['nla']['x']:.3f} y {data[r]['nla']['y']:.3f} ({data[r]['nla']['n_stated']}/{data[r]['nla']['n_samples']}), "
                           f"Oracle Lens x {data[r]['ol']['x']:.3f} y {data[r]['ol']['y']:.3f} ({data[r]['ol']['n_stated']}/{data[r]['ol']['n_samples']})" for r in data)),
    )
    for c in checks:
        if not c["match"]:
            print("MISMATCH", c)
    print("ok", len(checks), "checks")


if __name__ == "__main__":
    main()
