"""s5_scorecards: per-rule hits vs the all-low baseline across exps 18, 19, 22 (computed from score*.json)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from style import C, plt  # noqa: E402

E = style.ROOT / "experiments"
SRC = {"18": "18_prediction_test/results/score18.json", "19": "19_natural_bank/results/score19.json",
       "22": "22_public_bank/results/score22.json"}
PRED = {"18": "18_prediction_test/results/predictions18.md", "19": "19_natural_bank/results/predictions19.md",
        "22": "22_public_bank/results/predictions22.md"}
CELLS = ["nla_prompted", "nla_states", "nla_names_complement", "ol_prompted", "ol_exhibits", "ol_states",
         "jlens_prompted"]
CUED = {"19": {"json_depth", "html_depth"}, "22": {"json_depth", "code_lang"}, "18": set()}  # declared lexical cue >= 0.65
PRIMARY = ["nla_states", "nla_names_complement", "ol_exhibits", "ol_states", "jlens_prompted"]

# (label, cell, which tests the rule applies to, reader colour key)
RULES = [
    ("NLA prompted high exactly when readable", "nla_prompted", {"18", "19", "22"}, "nla"),
    ("NLA states exactly when the model uses it (exp 18 rule)", "nla_states", {"18"}, "nla"),
    ("NLA states exactly when statement-sayable (revised)", "nla_states", {"19", "22"}, "nla"),
    ("NLA complement (multi-choice): never (exp 18 rule)", "nla_names_complement", {"18"}, "nla"),
    ("NLA complement (multi-choice) exactly when both values positive (revised)", "nla_names_complement", {"19", "22"}, "nla"),
    ("Oracle Lens never states", "ol_states", {"18", "19", "22"}, "ol"),
    ("Oracle Lens prompted high exactly when span-sayable", "ol_prompted", {"18", "19", "22"}, "ol"),
    ("Oracle Lens exhibits exactly when span-sayable (judged)", "ol_exhibits", {"18", "19", "22"}, "ol"),
    ("J-lens high exactly when J-enriched (uncued rows)", "jlens_prompted", {"18", "19", "22"}, "jl"),
]
RETIRED = {("NLA states exactly when the model uses it (exp 18 rule)", "19"), ("NLA states exactly when the model uses it (exp 18 rule)", "22"),
           ("NLA complement (multi-choice): never (exp 18 rule)", "19"), ("NLA complement (multi-choice): never (exp 18 rule)", "22")}


def load():
    return {k: json.loads((E / p).read_text()) for k, p in SRC.items()}


def cells_for(score, test, cell, uncued_only=True):
    out = []
    for r in score["rows"]:
        if cell == "jlens_prompted" and uncued_only and r["feature"] in CUED[test]:
            continue
        out.append(r["cells"][cell])
    return out


def tally(cs):
    return sum(c["correct"] for c in cs), sum(not c["observed_high"] for c in cs), len(cs)


def aggregates(score, test):
    allc = [r["cells"][c] for r in score["rows"] for c in CELLS]
    res = {"All 56 cells" if test != "18" else "All 70 cells": tally(allc)}
    if test != "18":
        prim = [(c, r["cells"][c]) for r in score["rows"] for c in PRIMARY
                if not (c == "jlens_prompted" and r["feature"] in CUED[test])]
        res["Primary scorecard"] = tally([x for _, x in prim])
        res["Primary without judged exhibits"] = tally([x for c, x in prim if c != "ol_exhibits"])
    return res


def check_predictions(scores):
    """Predicted cells in score json must equal the hashed predictions*.md table."""
    bad = []
    for t, p in PRED.items():
        pm = {}
        for line in (E / p).read_text().splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if line.startswith("| ") and len(c) >= 16 and c[-7] in ("H", "L"):
                pm[c[0]] = [x == "H" for x in c[-7:]]
        for r in scores[t]["rows"]:
            for i, cell in enumerate(CELLS):
                if pm[r["feature"]][i] != r["cells"][cell]["predicted_high"]:
                    bad.append((t, r["feature"], cell))
    return bad


def main():
    scores = load()
    bad = check_predictions(scores)
    tests = [("18", "Exp 18 · 10 synthetic rows"), ("19", "Exp 19 · 8 natural-text rows"),
             ("22", "Exp 22 · public rebuild, 8 rows")]
    agg_labels = ["All cells", "Primary scorecard", "Primary without judged exhibits"]
    labels = [r[0] for r in RULES] + agg_labels
    n_rows = len(labels)
    ys = {lab: n_rows - 1 - i + (0 if i < len(RULES) else -0.6) for i, lab in enumerate(labels)}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 7.6), sharey=True)
    fig.subplots_adjust(wspace=0.06)
    record = {}
    for ax, (t, title) in zip(axes, tests):
        s = scores[t]
        for lab, cell, applies, rk in RULES:
            y = ys[lab]
            if t not in applies:
                txt = "retired after exp 18" if (lab, t) in RETIRED else "written after exp 18"
                ax.text(0.5, y, txt, ha="center", va="center", fontsize=12, color=C["muted"], style="italic")
                continue
            hits, low, n = tally(cells_for(s, t, cell))
            record[(t, lab)] = (hits, low, n)
            pr, pb = hits / n, low / n
            ax.plot([pr, pr, pb, pb], [y + 0.14, y, y, y - 0.14], color=C["line"], lw=2.0, zorder=1)
            ax.scatter([pb], [y - 0.14], s=85, marker="D", facecolor="white", edgecolor=C["muted"], lw=1.8, zorder=3)
            ax.scatter([pr], [y + 0.14], s=110, color=C[rk], zorder=4)
            ax.text(1.09, y, f"{hits}/{n}", va="center", ha="left", fontsize=13, color=C["ink"])
            ax.text(1.37, y, f"{low}/{n}", va="center", ha="left", fontsize=13, color=C["muted"])
        ag = aggregates(s, t)
        for lab, key in zip(agg_labels, list(ag.keys()) + [None] * 3):
            y = ys[lab]
            if key is None:
                ax.text(0.5, y, "no primary split declared", ha="center", va="center", fontsize=12,
                        color=C["muted"], style="italic")
                continue
            hits, low, n = ag[key]
            record[(t, lab)] = (hits, low, n)
            p, lo, hi = style.wilson(hits, n)
            pb, blo, bhi = style.wilson(low, n)
            ax.plot([lo, hi], [y + 0.13, y + 0.13], color=C["ink"], lw=2, zorder=2)
            ax.plot([blo, bhi], [y - 0.13, y - 0.13], color=C["muted"], lw=2, zorder=2, alpha=0.7)
            ax.scatter([pb], [y - 0.13], s=95, marker="D", facecolor="white", edgecolor=C["muted"], lw=1.8, zorder=3)
            ax.scatter([p], [y + 0.13], s=120, color=C["ink"], zorder=4)
            ax.text(1.09, y, f"{hits}/{n}", va="center", ha="left", fontsize=13, color=C["ink"], weight="bold")
            ax.text(1.37, y, f"{low}/{n}", va="center", ha="left", fontsize=13, color=C["muted"], weight="bold")
        ax.axvline(0.80, color=C["muted"], lw=1.4, ls="--", zorder=0, alpha=0.8)
        ax.text(0.80, n_rows - 0.35, "0.80", ha="center", va="bottom", fontsize=11, color=C["muted"])
        ax.axhline(ys[agg_labels[0]] + 0.75, color=C["line"], lw=1)
        ax.set_xlim(-0.03, 1.64)
        ax.spines["bottom"].set_bounds(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(title, fontsize=15, loc="left", pad=26)
        ax.text(1.09, n_rows - 0.35, "rules", fontsize=12, color=C["ink"], va="bottom")
        ax.text(1.37, n_rows - 0.35, "all-low", fontsize=12, color=C["muted"], va="bottom")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("share of rows predicted correctly")
    axes[0].set_yticks([ys[lab] for lab in labels])
    def plain(lab):
        return lab.replace(" exactly when ", " when ").replace(" (exp 18 rule)", " (old)").replace(" (revised)", " (new)")
    axes[0].set_yticklabels([plain(lab) for lab in labels], fontsize=13.5)
    for tl in axes[0].get_yticklabels()[-3:]:
        tl.set_weight("bold")
    axes[0].set_ylim(ys[agg_labels[-1]] - 0.7, n_rows - 0.1)
    # legend
    h1 = axes[0].scatter([], [], s=110, color=C["ink"], label="pre-registered rules (colour = reader)")
    h2 = axes[0].scatter([], [], s=95, marker="D", facecolor="white", edgecolor=C["muted"], lw=1.8,
                         label="always predict 'low' (baseline)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.06), fontsize=13)
    # key annotation on exp 19 / 22 primary rows
    for ax, t in zip(axes[1:], ["19", "22"]):
        h, low, n = record[(t, "Primary scorecard")]
        if low >= h:
            ax.annotate("baseline\n" + ("beats" if low > h else "ties") + " rules", xy=(low / n - 0.04, ys["Primary scorecard"] - 0.13),
                        xytext=(0.0, ys["Primary scorecard"] - 0.13), fontsize=13, color=C["ink"], weight="bold",
                        va="center", ha="left", arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.2))

    checks = []

    def ck(q, comp, rec, where, match=None):
        checks.append({"quantity": q, "computed": comp, "record": rec, "where": where,
                       "match": (str(comp) == str(rec)) if match is None else match})

    ck("predicted cells in score*.json equal the hashed predictions*.md tables", f"{len(bad)} mismatches",
       "0 mismatches", "predictions18/19/22.md")
    for t, lab, rec in [("18", "NLA prompted high exactly when readable", "8/10"),
                        ("18", "NLA states exactly when the model uses it (exp 18 rule)", "5/10"),
                        ("18", "Oracle Lens never states", "10/10"),
                        ("18", "Oracle Lens prompted high exactly when span-sayable", "9/10"),
                        ("18", "Oracle Lens exhibits exactly when span-sayable (judged)", "6/10"),
                        ("18", "J-lens high exactly when J-enriched (uncued rows)", "2/10"),
                        ("19", "NLA prompted high exactly when readable", "5/8"), ("22", "NLA prompted high exactly when readable", "6/8"),
                        ("19", "NLA states exactly when statement-sayable (revised)", "8/8"),
                        ("22", "NLA states exactly when statement-sayable (revised)", "6/8"),
                        ("19", "NLA complement (multi-choice) exactly when both values positive (revised)", "6/8"),
                        ("22", "NLA complement (multi-choice) exactly when both values positive (revised)", "6/8"),
                        ("19", "Oracle Lens prompted high exactly when span-sayable", "6/8"),
                        ("22", "Oracle Lens prompted high exactly when span-sayable", "7/8"),
                        ("19", "Oracle Lens exhibits exactly when span-sayable (judged)", "3/8"),
                        ("22", "Oracle Lens exhibits exactly when span-sayable (judged)", "4/8"),
                        ("19", "J-lens high exactly when J-enriched (uncued rows)", "6/6"),
                        ("22", "J-lens high exactly when J-enriched (uncued rows)", "6/6"),
                        ("18", "All cells", "44/70"), ("19", "All cells", "44/56"), ("22", "All cells", "44/56"),
                        ("19", "Primary scorecard", "31/38"), ("22", "Primary scorecard", "30/38"),
                        ("19", "Primary without judged exhibits", "28/30"),
                        ("22", "Primary without judged exhibits", "26/30")]:
        h, low, n = record[(t, lab)]
        ck(f"exp {t} rule hits: {lab}", f"{h}/{n}", rec, "talk script S5 table / RESULTS.md")
    for t, lab, rec in [("19", "All cells", "45/56"), ("22", "All cells", "36/56"), ("19", "Primary scorecard", "33/38"),
                        ("22", "Primary scorecard", "30/38"), ("19", "Primary without judged exhibits", "25/30"),
                        ("22", "Primary without judged exhibits", "23/30")]:
        h, low, n = record[(t, lab)]
        ck(f"exp {t} all-low baseline: {lab}", f"{low}/{n}", rec, "BRIEF known-errors list (earlier recount)")
    # exp 18 baseline (new)
    h, low, n = record[("18", "All cells")]
    ck("exp 18 all-low baseline, all cells (rules 44/70)", f"{low}/{n}", "not in record", "new quantity", match=None)
    checks[-1]["match"] = None
    # exp 19 vs 22 agreement
    r19 = {r["feature"]: r for r in scores["19"]["rows"]}
    r22 = {r["feature"]: r for r in scores["22"]["rows"]}
    shared = [f for f in r19 if f in r22]
    agree = sum(r19[f]["cells"][c]["correct"] == r22[f]["cells"][c]["correct"] for f in shared for c in CELLS)
    agree_obs = sum(r19[f]["cells"][c]["observed_high"] == r22[f]["cells"][c]["observed_high"] for f in shared for c in CELLS)
    pos = sum(scores["19"]["rows"][i]["cells"][c]["correct"] == scores["22"]["rows"][i]["cells"][c]["correct"]
              for i in range(8) for c in CELLS)
    ck("exp 19 vs exp 22 scorecards agree (hit/miss, shared rows)", f"{agree}/{len(shared) * 7}", "50/56",
       "exp 22 RESULTS.md Reading 1; talk script S5")
    ck("exp 19 vs exp 22 agree on observed high/low (shared rows)", f"{agree_obs}/{len(shared) * 7}", "50/56", "same")
    ck("exp 19 vs exp 22 hit/miss agreement if rows paired by position (HTML nesting depth vs code language)", f"{pos}/56", "50/56",
       "attempt to reproduce the record's denominator")
    # revised rules alone vs baseline
    for t in ["19", "22"]:
        cs = cells_for(scores[t], t, "nla_states") + cells_for(scores[t], t, "nla_names_complement")
        h, low, n = tally(cs)
        ck(f"exp {t} revised NLA rules (16 cells): rules (baseline {low}/{n} is new)", f"{h}/{n}",
           "14/16" if t == "19" else "12/16", "RESULTS.md / talk script S5")

    title = "The prediction rules do not beat 'always predict low' on the primary scorecard"
    caption = ("Each row is a pre-registered rule; dots show the share of rows it predicted correctly (colour = reader) "
               "and open diamonds the share an always-'low' predictor gets on the same cells; numbers at right are hits/rows. "
               "Bold rows aggregate the cells with Wilson 95% intervals (top: rules, bottom: baseline); dashed line is the "
               "declared 0.80 bar. All-low ties the rules on exp 22's primary scorecard (30/38 vs 30/38) and beats them on "
               "exp 19's (33/38 vs 31/38); the rules' margin comes from complement naming and the prompted cells, while "
               "'NLA states exactly when statement-sayable' predicted low on every natural row and so equals the baseline. "
               "'Complement (multi-choice)' is the pre-registered cell 'NLA names the complement': a multi-candidate "
               "forced-choice argmax on the unmarked value (prereg section 3), not a regex or judge verdict.")
    notes = (f"Computed from score18/19/22.json (predicted cells verified against the hashed predictions*.md: {len(bad)} "
             "mismatches). J-lens cells: exps 19/22 score only rows with lexical cue < 0.65 as declared "
             "(JSON and HTML nesting depth excluded in 19; JSON nesting depth and code language in 22); exp 18 scores all ten rows. "
             "Primary = NLA states, NLA complement, Oracle Lens exhibits, Oracle Lens states, J-lens uncued; secondary = the two prompted cells "
             "(no primary split was declared for exp 18). Record says the exp 19 and 22 scorecards agree on 50 of 56 cells; "
             f"only 7 rows (49 cells) are shared and hit/miss agrees on {agree}/49 (observed high/low on {agree_obs}/49; "
             f"pairing rows by position gives {pos}/56): 50/56 cannot be reproduced. Exp 18's predictions18.md hashes to "
             "1edd8283...; the record (decisions D41, exp 18 RESULTS) gives 2db98042... for it, which is the sha256 of "
             "instruments18.json, so the predictions file's own hash was mislabelled (file has one commit, unchanged).")
    style.save(fig, "s5_scorecards", title=title, caption=caption, section="S5", role="backup",
               sources=list(SRC.values()) + list(PRED.values()), checks=checks, notes=notes)
    for c in checks:
        print(("OK  " if c["match"] else "DIFF"), c["quantity"], c["computed"], "| record", c["record"])


if __name__ == "__main__":
    main()
