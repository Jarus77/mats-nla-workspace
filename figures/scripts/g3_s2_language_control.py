"""s2_language_control: the French vs German control row through every instrument (exps 02, 12, 18, 28)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from style import C, plt  # noqa: E402

E = style.ROOT / "experiments"
P02 = E / "02_nonj_features" / "gates.json"
P18I = E / "18_prediction_test" / "results" / "instruments18.json"
P18S = E / "18_prediction_test" / "results" / "score18.json"
P28 = {b: E / "28_stmt_curve" / "results" / f"stmt28_{b}_ij2_lang.json" for b in ("12", "18")}
JUD = {"12": E / "12_concept_recall" / "results" / "judged", "18": E / "18_prediction_test" / "results" / "judged"}
BANKS = ["12", "18"]
BANK_LABEL = {"12": "exp 12\nbank", "18": "exp 18\nbank", "02": "exp 02\nbank"}


def rel(p):
    return str(Path(p).resolve().relative_to(style.ROOT))


def judged_rates(bank, reader):
    """Same computation as g2_common.states_rate: share stated / exhibiting, item-clustered bootstrap 95%
    (Wilson on readouts when the count is 0)."""
    p = JUD[bank] / f"ij2_lang_{reader}.json"
    d = json.loads(p.read_text())
    rows = d["judged"]
    code = {d["values"][0]: "A", d["values"][1]: "B"}
    g = np.asarray([r["item"] for r in rows])
    out = {"n": len(rows), "items": len(set(g)), "src": rel(p)}
    for key, flags in [("states", [bool(r["stated"]) for r in rows]),
                       ("exhibits", [r["exhibited_value"] == code[r["value"]] for r in rows])]:
        f = np.asarray(flags, float)
        k = int(f.sum())
        if k > 0:
            lo, hi = style.bootstrap_ci(lambda a: float(a.mean()), f, groups=g)
        else:
            _, lo, hi = style.wilson(0, len(f))
        out[key] = (k / len(f), lo, hi, k)
    return out


def main():
    style.slide_style()
    g02 = json.loads(P02.read_text())["lang"]["G2_geometric_enrichment_diffmeans"]
    i18 = next(r for r in json.loads(P18I.read_text()) if r["feature"] == "ij2_lang")["instr"]
    s18 = next(r for r in json.loads(P18S.read_text())["rows"] if r["feature"] == "ij2_lang")["cells"]
    st28 = {b: json.loads(P28[b].read_text()) for b in BANKS}
    J = {(b, r): judged_rates(b, r) for b in BANKS for r in ("nla", "ol42")}

    readable = {"12": st28["12"]["readable"], "18": i18["readable"]}
    shift_fixed = {"12": st28["12"]["shift"]["fixed"], "18": i18["stmt"]}
    para = {b: (st28[b]["shift"]["para_mean"], st28[b]["shift"]["para_min"], st28[b]["shift"]["para_max"]) for b in BANKS}
    jenr = {"02": g02["r1536"], "18": i18["jenr"]}

    # rows: (label, colour, x-range, threshold, threshold label, [(lane value, lo, hi) for open, filled], fmt)
    def jr(b, rd, key):
        p, lo, hi, _ = J[(b, rd)][key]
        return (p, lo, hi)
    left = [("readable\n(probe AUROC)", C["probe"], (0.5, 1.0), 0.80, "0.80",
             [(readable["12"], None, None), (readable["18"], None, None)]),
            ("statement shift", C["nla"], (0.0, 0.7), 0.30, "0.30",
             [(shift_fixed[b], para[b][1], para[b][2]) for b in BANKS]),
            ("J-enrichment", C["jl"], (0.0, 2.0), 1.5, "1.5",
             [(jenr["02"], None, None), (jenr["18"], None, None)])]
    right = [("NLA states", C["nla"], (0, 1), None, None, [jr("12", "nla", "states"), jr("18", "nla", "states")]),
             ("Oracle Lens\nexhibits", C["ol"], (0, 1), None, None,
              [jr("12", "ol42", "exhibits"), jr("18", "ol42", "exhibits")]),
             ("Oracle Lens\nstates", C["ol"], (0, 1), None, None, [jr("12", "ol42", "states"), jr("18", "ol42", "states")])]
    fig, axes = plt.subplots(3, 2, figsize=(style.SLIDE[0] - 0.45, style.SLIDE[1] - 0.32))
    fig.subplots_adjust(hspace=1.05, wspace=0.75, left=0.2, right=0.98)
    LANE = {0: 0.45, 1: -0.45}
    for col, spec in enumerate((left, right)):
        for r, (lab, c, (x0, x1), thr, tl, lanes) in enumerate(spec):
            ax = axes[r, col]
            ax.set_xlim(x0, x1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=8)
            ax.set_xticks(np.linspace(x0, x1, 3))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            if thr is not None:
                ax.axvline(thr, color=C["muted"], ls="--", lw=1.3, zorder=0)
            for j, v in enumerate(lanes):
                m, lo, hi = v
                y = LANE[j]
                if lo is not None:
                    ax.plot([lo, hi], [y, y], color=c, lw=2.2, alpha=0.45, zorder=1)
                bar_color = c if not (lab == "J-enrichment" and m < 1.5) else C["jl_pale"]
                ax.scatter([m], [y], s=70, facecolor=bar_color if j == 1 else "white", edgecolor=bar_color,
                           lw=1.8, zorder=3, clip_on=False)
                right_side = (hi if hi is not None else m)
                if (right_side - x0) / (x1 - x0) < 0.72:
                    ax.text(right_side + 0.03 * (x1 - x0), y, f"{m:.2f}", va="center", ha="left", fontsize=15,
                            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"), zorder=4)
                else:
                    left_side = lo if lo is not None else m
                    ax.text(left_side - 0.03 * (x1 - x0), y, f"{m:.2f}", va="center", ha="right", fontsize=15,
                            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"), zorder=4)
    axes[0, 0].set_title("instruments (dashed = bar)", loc="left", fontsize=16, pad=10)
    axes[0, 1].set_title("free readouts (share of 320)", loc="left", fontsize=16, pad=10)
    fig.text(0.98, -0.04, "open = exp 12 bank (J-enrichment: exp 02), filled = exp 18 bank", ha="right",
             va="top", fontsize=15, color=C["muted"])

    ck = []

    def add(q, comp, rec, where, tol=0.005):
        ck.append({"quantity": q, "computed": round(float(comp), 4), "record": rec, "where": where,
                   "match": abs((round(float(comp), 2) if tol >= 0.005 else float(comp)) - rec) <= tol})

    add("readable, exp 18 bank (instruments18)", readable["18"], 1.00, "talk script S2 'readable 1.00'")
    add("readable, exp 12 bank (stmt28)", readable["12"], 1.00, "talk script S2 'readable 1.00'")
    add("statement shift, fixed statement, exp 18 (instruments18)", shift_fixed["18"], 0.51, "talk script S2 'statement shift 0.51'")
    add("statement shift, fixed statement, exp 18 (exp 28 recompute)", st28["18"]["shift"]["fixed"], 0.51, "talk script S2")
    add("NLA states, exp 12", J[("12", "nla")]["states"][0], 0.81, "talk script S2 '81 to 95%' / S3 table 81%")
    add("NLA states, exp 18", J[("18", "nla")]["states"][0], 0.95, "talk script S2 '81 to 95%'")
    add("NLA states, exp 18 vs score18.json", J[("18", "nla")]["states"][0], round(s18["nla_states"]["observed"], 4),
        "score18.json nla_states", tol=1e-4)
    add("Oracle Lens exhibits, exp 12", J[("12", "ol42")]["exhibits"][0], 0.56, "talk script S3 table 'exhibits it 56%'")
    add("Oracle Lens exhibits, exp 18 vs score18.json", J[("18", "ol42")]["exhibits"][0],
        round(s18["ol_exhibits"]["observed"], 4), "score18.json ol_exhibits", tol=1e-4)
    add("Oracle Lens states, exp 12", J[("12", "ol42")]["states"][0], 0.0, "talk script S3 table '0%'")
    ck.append({"quantity": "J-enriched (>= 1.5 at r = 1536), exp 18 instrument", "computed": round(jenr["18"], 4),
               "record": "J-enriched (talk script S2)", "where": "talk script S2 vs instruments18 decide.jenr = false",
               "match": jenr["18"] >= 1.5})
    ck.append({"quantity": "J-enrichment, exp 02 gate, r = 1536 (r = 384 in notes)", "computed": round(jenr["02"], 4),
               "record": "J-enriched (talk script S2)", "where": "exp 02 gates.json lang G2", "match": jenr["02"] >= 1.5})

    nla12, nla18 = J[("12", "nla")]["states"][0], J[("18", "nla")]["states"][0]
    title = "The language control is readable and moves the critic; the NLA states it, Oracle Lens only writes in it"
    caption = ("French vs German, the control row, through each instrument. Readable = linear-probe AUROC on the "
               "layer-42 activation (0.80 bar); statement shift through the NLA critic (dot = the fixed statement, "
               "whisker = min to max over 8 paraphrase pairs; 0.30 bar, since withdrawn); J-enrichment of the "
               "class-mean direction in the top-1536 J-space (1.5 bar; pale = below it); free readouts = share of 320 "
               "readouts (40 items x 8) a judge marked as stating the language or as written in it, item-clustered "
               f"bootstrap 95%. Open = exp 12 bank (exp 02 for J-enrichment), filled = exp 18 bank. The NLA states the language on {nla12:.2f} to {nla18:.2f} of readouts; Oracle Lens writes in "
               "it but never says it. Caveat: exp 18's own J-enrichment instrument gives 1.38, under the 1.5 bar.")
    notes = (f"Readable: exp 18 from instruments18.json ({readable['18']:.4f}), exp 12 bank from the exp 28 recompute "
             f"stmt28_12_ij2_lang.json ({readable['12']:.4f}); no per-item probe scores are saved, so no interval. "
             f"Statement shift: exp 18 fixed from instruments18.json ({shift_fixed['18']:.4f}; exp 28 recompute "
             f"{st28['18']['shift']['fixed']:.4f}), exp 12 fixed and both paraphrase means from stmt28_*_ij2_lang.json. "
             f"J-enrichment: exp 02 gates.json lang G2 diff-means r1536 = {g02['r1536']:.3f} (r384 = {g02['r384']:.3f}), "
             f"exp 18 instruments18.json jenr = {i18['jenr']:.3f} (r = 1536, as in modal18.py) with decide.jenr = false. "
             "Record discrepancy: the talk script calls this row J-enriched; exp 02's gate supports that, exp 18's own "
             "instrument does not (1.38 < 1.5), so exp 18 predicted the J-lens cell high only through the statement "
             "shift. Exp 12 has no J-enrichment instrument for this row, hence exp 02 in that slot. Free readouts: "
             "judged/ij2_lang_{nla,ol42}.json in exps 12 and 18; same counting as g2_common.states_rate. Oracle Lens "
             f"states: exp 12 {J[('12', 'ol42')]['states'][3]} of 320, exp 18 {J[('18', 'ol42')]['states'][3]} of 320.")
    srcs = [rel(P02), rel(P18I), rel(P18S)] + [rel(P28[b]) for b in BANKS] + [J[k]["src"] for k in J]
    style.save(fig, "s2_language_control", title=title, caption=caption, section="S2", role="main",
               sources=srcs, checks=ck, notes=notes)
    for c in ck:
        print("OK  " if c["match"] else "DIFF", c["quantity"], c["computed"], "| record", c["record"])


if __name__ == "__main__":
    main()
