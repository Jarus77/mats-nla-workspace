"""s5_scorecards_slide: the three aggregate scorecard rows for exps 18/19/22, rules vs always-low, Wilson 95%.

Same raw files and tallies as g3_s5_scorecards.py (imported), cut to the aggregates for the slide."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from style import C, plt  # noqa: E402
import g3_s5_scorecards as sc  # noqa: E402

TESTS = [("18", "Exp 18 · synthetic"), ("19", "Exp 19 · natural text"), ("22", "Exp 22 · public bank")]
ROWS = [("all", "all cells"), ("Primary scorecard", "primary\n(free readouts)"),
        ("Primary without judged exhibits", "primary without\njudged exhibits")]
DY = 0.17


def agg_key(ag, key):
    if key == "all":
        return next(k for k in ag if k.startswith("All"))
    return key if key in ag else None


def main():
    style.slide_style()
    scores = sc.load()
    bad = sc.check_predictions(scores)
    rec = {}
    fig, axes = plt.subplots(1, 3, figsize=(style.SLIDE[0] - 0.3, style.SLIDE[1] - 0.15), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    ys = {k: len(ROWS) - 1 - i for i, (k, _) in enumerate(ROWS)}
    for ax, (t, ttl) in zip(axes, TESTS):
        ag = sc.aggregates(scores[t], t)
        for key, _ in ROWS:
            y = ys[key]
            k = agg_key(ag, key)
            if k is None:
                ax.text(0.40, y, "not declared", ha="left", va="center", fontsize=15, zorder=3,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white" if key == "all" or y != 1 else "#F4F6F7", ec="none"),
                        color=C["muted"], style="italic")
                continue
            hits, low, n = ag[k]
            rec[(t, key)] = (hits, low, n)
            p, lo, hi = style.wilson(hits, n)
            pb, blo, bhi = style.wilson(low, n)
            ax.plot([lo, hi], [y + DY] * 2, color=C["ink"], lw=2.2, zorder=2)
            ax.plot([blo, bhi], [y - DY] * 2, color=C["muted"], lw=2.2, zorder=2, alpha=0.7)
            ax.scatter([p], [y + DY], s=90, color=C["ink"], zorder=4)
            ax.scatter([pb], [y - DY], s=80, marker="D", facecolor="white", edgecolor=C["muted"], lw=1.8, zorder=3)
            strong = key == "Primary scorecard"
            ax.text(1.03, y + DY, f"{hits}/{n}", va="center", fontsize=15, color=C["ink"],
                    weight="bold" if strong else "normal")
            ax.text(1.03, y - DY, f"{low}/{n}", va="center", fontsize=15, color=C["muted"],
                    weight="bold" if strong else "normal")
            if strong and low >= hits:
                ax.text(0.37, y - 0.40, "always-low " + ("wins" if low > hits else "ties"), va="center", ha="left",
                        fontsize=15, weight="bold", color=C["ink"])
        ax.axvline(0.80, color=C["muted"], lw=1.4, ls="--", zorder=0, alpha=0.8)
        if t == "18":
            ax.text(0.815, -0.42, "0.80 bar", ha="left", va="center", fontsize=15, color=C["muted"])
        ax.axhspan(ys["Primary scorecard"] - 0.5, ys["Primary scorecard"] + 0.5, color=C["line"], alpha=0.25, zorder=0)
        ax.set_xlim(0.35, 1.30)
        ax.spines["bottom"].set_bounds(0.4, 1.0)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0], ["0.4", "0.6", "0.8", "1"])
        ax.set_title(ttl, loc="left", fontsize=16, pad=6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlabel("share of cells predicted correctly (Wilson 95%)")
    axes[0].set_yticks([ys[k] for k, _ in ROWS], [lab for _, lab in ROWS])
    axes[0].set_ylim(-0.55, len(ROWS) - 0.45)
    h1 = axes[0].scatter([], [], s=90, color=C["ink"], label="rules")
    h2 = axes[0].scatter([], [], s=80, marker="D", facecolor="white", edgecolor=C["muted"], lw=1.8,
                         label="always 'low'")
    axes[0].legend(handles=[h1, h2], loc="upper right", bbox_to_anchor=(-0.02, -0.02), handletextpad=0.2,
                   borderaxespad=0)

    checks = [{"quantity": "predicted cells in score*.json equal the hashed predictions*.md tables",
               "computed": f"{len(bad)} mismatches", "record": "0 mismatches", "where": "predictions18/19/22.md",
               "match": len(bad) == 0}]

    def ck(t, key, which, record, where):
        h, low, n = rec[(t, key)]
        v = h if which == "rules" else low
        checks.append({"quantity": f"exp {t} {which}: {dict(ROWS)[key].replace(chr(10), ' ')}",
                       "computed": f"{v}/{n}", "record": record, "where": where, "match": f"{v}/{n}" == record})

    for t, rr in [("18", "44/70"), ("19", "44/56"), ("22", "44/56")]:
        ck(t, "all", "rules", rr, "talk script S5 table / RESULTS.md")
    for t, rr in [("19", "31/38"), ("22", "30/38")]:
        ck(t, "Primary scorecard", "rules", rr, "talk script S5 / RESULTS.md")
    for t, rr in [("19", "28/30"), ("22", "26/30")]:
        ck(t, "Primary without judged exhibits", "rules", rr, "talk script S5 / RESULTS.md")
    for t, rr in [("19", "45/56"), ("22", "36/56")]:
        ck(t, "all", "always-low", rr, "BRIEF known-errors list")
    for t, rr in [("19", "33/38"), ("22", "30/38")]:
        ck(t, "Primary scorecard", "always-low", rr, "BRIEF known-errors list ('primary 33 vs 31 and 30 vs 30')")
    for t, rr in [("19", "25/30"), ("22", "23/30")]:
        ck(t, "Primary without judged exhibits", "always-low", rr, "earlier recount (g3_s5_scorecards)")
    h, low, n = rec[("18", "all")]
    checks.append({"quantity": "exp 18 always-low, all cells (no record)", "computed": f"{low}/{n}",
                   "record": "not in record", "where": "new quantity", "match": None})
    tie_or_lose = all(rec[(t, "Primary scorecard")][1] >= rec[(t, "Primary scorecard")][0] for t in ("19", "22"))
    checks.append({"quantity": "always-low >= rules on the primary scorecard in every test that declared one (19, 22)",
                   "computed": str(tie_or_lose), "record": "True (title claim)", "where": "this figure's title",
                   "match": tie_or_lose})

    r19, r22 = rec[("19", "Primary scorecard")], rec[("22", "Primary scorecard")]
    title = "On the primary scorecard the prediction rules tie or lose to always predicting 'low'"
    caption = ("Share of scorecard cells each test got right: filled dots = the pre-registered rules, open diamonds = a "
               "predictor that always says 'low' on the same cells; bars are Wilson 95% intervals over cells, numbers at "
               "right are correct/cells, dashed line is the declared 0.80 bar. On the primary scorecard (free readouts, "
               f"shaded) always-low gets {r19[1]}/{r19[2]} vs the rules' {r19[0]}/{r19[2]} in exp 19 and ties "
               f"{r22[1]}/{r22[2]} in exp 22; exp 18 declared no primary split. Only after dropping the judged Oracle Lens "
               "exhibits cell do the rules pull ahead. Per-rule rows are in the backup s5_scorecards.")
    notes = ("Computed with g3_s5_scorecards.py's load/aggregates/check_predictions from score18/19/22.json (no "
             "re-derivation). Primary = NLA states, NLA complement (multi-choice; pre-registered as 'NLA names the complement', a multi-candidate forced-choice argmax), Oracle Lens exhibits, Oracle Lens states, J-lens on "
             "uncued rows (JSON and HTML nesting depth excluded in 19; JSON nesting depth and code language in 22); "
             "secondary = the two prompted cells. Wilson intervals treat cells as independent, which they are not "
             "(several cells per row), so they are if anything too narrow. Exp 18 rules 44/70 vs always-low "
             f"{rec[('18', 'all')][1]}/70 on all cells: there the rules lead.")
    style.save(fig, "s5_scorecards_slide", title=title, caption=caption, section="S5", role="main",
               sources=list(sc.SRC.values()) + list(sc.PRED.values()), checks=checks, notes=notes)
    for c in checks:
        print("OK  " if c["match"] else "DIFF", c["quantity"], c["computed"], "| record", c["record"])


if __name__ == "__main__":
    main()
