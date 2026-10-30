"""bk_exp09_gain_ratio: Oracle Lens's newline continuation moves one-for-one with the model's own newline decision."""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

E09 = S.ROOT / "experiments" / "09_ol_mechanism"
sys.path.insert(0, str(E09))
import o9_analyze as A  # noqa: E402  (read-only import of its loaders)
from o9_templates import ANALOGUE  # noqa: E402

NB = 2000


def ratio_ci(groups, seed):
    rng = np.random.default_rng(seed)
    out, so, sb = [], [], []
    for _ in range(NB):
        o = b = 0.0
        no = 0
        for ol, be in groups:
            ix = rng.integers(0, len(ol), len(ol))
            o += ol[ix].sum(); b += be[ix].sum(); no += len(ix)
        out.append(o / b); so.append(o / no); sb.append(b / no)
    pc = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return pc(out), pc(so), pc(sb)


def main():
    OL = {(s, a): A.load_ol(s, a) for s in ("col", "tense") for a in ("on", "off")}
    BE = {s: A.load_behave(s) for s in ("col", "tense")}
    spec = [("col_raw", "line position, raw", [("col", "reg_raw")]),
            ("col_4r", "line position, J-space removed", [("col", "reg_jrem4")]),
            ("tense_raw", "tense, raw", [("tense", "raw")]),
            ("tense_4r", "tense, J-space removed", [("tense", "jrem4")]),
            ("randC", "random direction C", [("col", "randC"), ("tense", "randC")]),
            ("randD", "random direction D", [("col", "randD"), ("tense", "randD")])]
    analog = lambda d: ANALOGUE["col" if d.startswith("reg") else "tense" if d in ("raw", "jrem4") else d]  # noqa: E731
    res = {}
    for i, (name, lab, members) in enumerate(spec):
        groups = []
        for s, d in members:
            ch, ro = analog(d)
            p, m = A.pm(s, OL[(s, "on")][ch], d, "r108")
            bp, bm = A.pm(s, BE[s][ro], d, "r108")
            groups.append((p - m, bp - bm))
        o = np.concatenate([g[0] for g in groups]).mean()
        b = np.concatenate([g[1] for g in groups]).mean()
        rci, oci, bci = ratio_ci(groups, seed=100 + i)
        res[name] = dict(lab=lab, ol=float(o), beh=float(b), ratio=float(o / b), rci=rci, oci=oci, bci=bci)
    off = {}
    for s, ch, d, lab in (("col", "nl_cont", "reg_raw", "line position, raw"), ("col", "nl_cont", "reg_jrem4", "line position, J-removed"),
                          ("tense", "f07_tense", "raw", "tense, raw"), ("tense", "f07_tense", "jrem4", "tense, J-removed")):
        row = {}
        for ad in ("on", "off"):
            p, m = A.pm(s, OL[(s, ad)][ch], d, "q90")
            a = S.auroc(np.r_[p, m], np.r_[np.ones(len(p)), np.zeros(len(m))])
            rng = np.random.default_rng(7)
            bs = []
            for _ in range(NB):
                ix = rng.integers(0, len(p), len(p))
                bs.append(S.auroc(np.r_[p[ix], m[ix]], np.r_[np.ones(len(p)), np.zeros(len(p))]))
            row[ad] = (a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        off[lab] = row

    fig, (ax, bx) = S.plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"width_ratios": [1.15, 1]})
    lim = (-0.3, 2.9)
    ax.plot(lim, lim, color=S.C["line"], ls="--", lw=1.5, zorder=0)
    ax.text(1.62, 1.42, "one-for-one", color=S.C["muted"], fontsize=12, rotation=45, ha="center", va="top")
    style = {"col_raw": (S.C["ol"], "o"), "col_4r": (S.C["ol"], "s"), "tense_raw": (S.C["ol_pale"], "o"),
             "tense_4r": (S.C["ol_pale"], "s"), "randC": (S.C["muted"], "^"), "randD": (S.C["muted"], "v")}
    offs = {"col_raw": (-16, 6, "right"), "col_4r": (12, -22, "left"), "tense_raw": (12, 0, "left"),
            "tense_4r": (14, 4, "left"), "randC": None, "randD": None}
    for k, v in res.items():
        c, mk = style[k]
        ax.errorbar(v["beh"], v["ol"], xerr=[[v["beh"] - v["bci"][0]], [v["bci"][1] - v["beh"]]],
                    yerr=[[v["ol"] - v["oci"][0]], [v["oci"][1] - v["ol"]]], fmt=mk, ms=10, color=c,
                    ecolor=c, elinewidth=1.3, capsize=3, mec=S.C["ink"] if k.startswith("col") else c)
        if offs[k] is None:
            continue
        dx, dy, ha = offs[k]
        txt = (f"{v['lab']}\nratio {v['ratio']:.2f}" if k != "tense_4r"
               else f"{v['lab']}\n(model does not move)")
        ax.annotate(txt, (v["beh"], v["ol"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=11.5, ha=ha, va="center", color=S.C["ink"] if k.startswith("col") else S.C["muted"])
    rc, rd = res["randC"], res["randD"]
    ax.annotate(f"two random directions\nratios {rc['ratio']:.2f}, {rd['ratio']:.2f}", (rd["beh"], rd["ol"]), xytext=(18, -30),
                textcoords="offset points", fontsize=11.5, ha="left", va="center", color=S.C["muted"])
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("model's own next-token shift (nats)")
    ax.set_ylabel("Oracle Lens continuation shift (nats)")
    ax.set_title("same push, two responses", fontsize=15)

    labs = list(off)
    y = np.arange(len(labs))[::-1]
    for yi, lab in zip(y, labs):
        for ad, c, dy in (("on", S.C["ol"], 0.13), ("off", S.C["ol_pale"], -0.13)):
            a, lo, hi = off[lab][ad]
            bx.errorbar(a, yi + dy, xerr=[[a - lo], [hi - a]], fmt="o", ms=9, color=c, capsize=3,
                        label=("adapter on (Oracle Lens)" if ad == "on" else "adapter off (base Qwen, same injection)") if yi == y[0] else None)
            bx.text(hi + 0.012, yi + dy, f"{a:.2f}", va="center", fontsize=12)
    bx.axvline(0.5, color=S.C["line"], ls="--", lw=1.5, zorder=0)
    bx.set_yticks(y, labs)
    bx.set_xlim(0.35, 1.1)
    bx.set_ylim(-1.3, len(labs) - 0.45)
    bx.set_xlabel("AUROC, + push vs - push (dashed: chance)")
    bx.set_title("the response needs Oracle Lens's adapter", fontsize=15)
    bx.legend(loc="lower right", fontsize=11.5, bbox_to_anchor=(1.04, -0.01))
    fig.tight_layout(w_pad=2.5)

    rec = {"col_raw": (2.60, 2.62, 0.99), "col_4r": (2.07, 1.96, 1.06), "tense_raw": (2.07, 0.61, 3.40),
           "tense_4r": (0.62, -0.04, -15.5), "randC": (0.45, 0.27, 1.65), "randD": (0.43, 0.23, 1.89)}
    checks = []
    for k, (o, b, r) in rec.items():
        v = res[k]
        checks.append({"quantity": f"{k} gain ratio r108", "computed": round(v["ratio"], 2), "record": r,
                       "where": "09 RESULTS.md (b)", "match": abs(v["ratio"] - r) < 0.006 * max(1, abs(r))})
        checks.append({"quantity": f"{k} Oracle Lens / model shift", "computed": [round(v["ol"], 2), round(v["beh"], 2)], "record": [o, b],
                       "where": "09 RESULTS.md (b)", "match": abs(v["ol"] - o) < 0.006 and abs(v["beh"] - b) < 0.006})
    rec_off = {"line position, raw": (0.69, 0.42), "line position, J-removed": (0.63, 0.43), "tense, raw": (0.99, 0.51), "tense, J-removed": (0.65, 0.50)}
    for lab, (on, of) in rec_off.items():
        checks.append({"quantity": f"adapter on/off AUROC q90, {lab}", "computed": [round(off[lab]["on"][0], 2), round(off[lab]["off"][0], 2)],
                       "record": [on, of], "where": "09 RESULTS.md (c)",
                       "match": abs(off[lab]["on"][0] - on) < 0.006 and abs(off[lab]["off"][0] - of) < 0.006})
    S.save(fig, "bk_exp09_gain_ratio",
           title="For line position, Oracle Lens's continuation moves one-for-one with the model's own newline decision",
           caption=("Pilot (excluded from main analysis), layer 42, 80 fresh items per set (random directions pooled over 160). "
                    "Left: each activation is pushed +/- along a direction (centred, 0.108 of its norm); x is the shift in the model's "
                    "own next-token log-odds, y the shift in Oracle Lens's matching continuation (first bullet, e.g. newline vs "
                    "'the'); 95% item-bootstrap CIs. Line position sits on the identity line, raw and J-removed; tense and random "
                    "directions are amplified 1.7-3.4x, and tense J-removed moves Oracle Lens while the model does not. Right: "
                    "the same pushes at the q90 dose with the adapter off give chance (95% item bootstrap). Read as re-instatement "
                    "of the model's newline decision, not verbalised line position; the pre-registered rule was inconclusive."),
           section="backup:exp09", role="backup",
           sources=["experiments/09_ol_mechanism/results/09_ol_mechanism/ol_{on,off}_{col,tense}_chunk*.pt",
                    "experiments/09_ol_mechanism/results/09_ol_mechanism/behave_{col,tense}.pt"],
           checks=checks,
           notes=("Oracle Lens channels (o9_templates.py): nl_cont ('- \\nthe' vs '- the', and/of); tense_tok ('- was' vs '- is', were/are, "
                  "had/has); f07_tense ('- They were' vs '- They are', ...). Model readouts use the same tokens at the model's next "
                  "position. Loaded via o9_analyze.load_ol/load_behave/pm (read-only import). Random directions randC/randD are the "
                  "two chosen by o9_analyze's fixed rule. Ratio CIs: " +
                  json.dumps({k: [round(x, 2) for x in v["rci"]] for k, v in res.items()})))
    for k, v in res.items():
        print(k, round(v["ol"], 3), round(v["beh"], 3), round(v["ratio"], 3), [round(x, 2) for x in v["rci"]])
    print(off)
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
