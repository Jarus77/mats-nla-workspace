"""bk_exp07_col_text_ceiling: natural line position is plain in the text but barely in the readers' log-odds."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

RES = S.ROOT / "experiments" / "07_template_mining" / "results"
F = "nj1_col"


def residual_auroc(lo, text_lo, y):
    X = np.c_[np.ones_like(text_lo), text_lo]
    beta, *_ = np.linalg.lstsq(X, lo, rcond=None)
    return S.auroc(lo - X @ beta, y)


def main():
    local = json.loads((RES / f"textnla_local_{F}.json").read_text())
    gem = json.loads((RES / f"textnla_gemini_{F}.partial.json").read_text())
    gen = {it["id"]: it for it in json.loads((RES / f"gen_ol_{F}.json").read_text())["items"]}
    ids, y = local["ids"], np.array(local["y"])
    cl = np.array([gen[i]["cluster"] for i in ids])
    res = {}
    for rd in ("nla", "ol"):
        sc = json.loads((RES / f"score_{rd}_{F}.json").read_text())["scores"][f"{rd}@42"]
        pos = {i: k for k, i in enumerate(sc["ids"])}
        sel = [t["id"] for t in local["templates"] if t["reader"] == rd and t["role"] == "selected"]
        act = np.mean([[sc["lo"][t][pos[i]] for i in ids] for t in sel], 0)
        txq = np.mean([local["lo"][t] for t in sel], 0)
        txg = np.nanmean(np.array([gem[t] for t in sel], float), 0)
        out = {}
        for k, v in (("act", act), ("qwen", txq), ("gemini", txg)):
            ok = np.isfinite(v)
            p = S.auroc(v[ok], y[ok])
            lo, hi = S.bootstrap_ci(S.auroc, v[ok], y[ok], groups=cl[ok])
            out[k] = (p, lo, hi)
        ok = np.isfinite(txg)
        out["after_gemini"] = residual_auroc(act[ok], txg[ok], y[ok])
        out["after_qwen"] = residual_auroc(act, txq, y)
        res[rd] = out

    fig, ax = S.plt.subplots(figsize=S.FULL)
    groups = [("nla", "NLA's three mined phrasings"), ("ol", "Oracle Lens's three mined phrasings")]
    series = [("act", "reader on the activation", None), ("qwen", "base Qwen on the text (no activation)", S.C["text_pale"]),
              ("gemini", "Gemini on the text (no activation)", S.C["text"])]
    w = 0.26
    for gi, (rd, glab) in enumerate(groups):
        for si, (k, lab, col) in enumerate(series):
            p, lo, hi = res[rd][k]
            c = col or S.C[rd]
            b = ax.bar(gi + (si - 1) * w, p, w * 0.92, color=c, label=lab if gi == 0 or k == "act" else None,
                       edgecolor="none")
            ax.errorbar(gi + (si - 1) * w, p, yerr=[[p - lo], [hi - p]], color=S.C["ink"], lw=1.4, capsize=4)
            ax.text(gi + (si - 1) * w, hi + 0.015, f"{p:.2f}", ha="center", va="bottom", fontsize=13)
        ax.text(gi - w, 0.315, f"{res[rd]['after_gemini']:.2f} with\nGemini's\nscore\nremoved",
                ha="center", va="bottom", fontsize=11.5, color="white", linespacing=1.05)
    ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_ylim(0.3, 1.04)
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylabel("AUROC, early vs late in the line (0.5 = chance)")
    S.chance_line(ax)
    h, l = ax.get_legend_handles_labels()
    # one legend entry for the two reader colours
    ax.legend([h[0], h[3], h[1], h[2]],
              ["NLA on the activation", "Oracle Lens on the activation", "base Qwen on the text (no activation)",
               "Gemini on the text (no activation)"], loc="lower center", fontsize=13, ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    checks = [
        {"quantity": "NLA mean3 AUROC on activation", "computed": round(res["nla"]["act"][0], 3), "record": 0.60, "where": "07 RESULTS.md TextNLA table", "match": abs(res["nla"]["act"][0] - 0.60) < 0.006},
        {"quantity": "Oracle Lens mean3 AUROC on activation", "computed": round(res["ol"]["act"][0], 3), "record": 0.55, "where": "07 RESULTS.md", "match": abs(res["ol"]["act"][0] - 0.55) < 0.006},
        {"quantity": "TextNLA-Qwen, NLA templates", "computed": round(res["nla"]["qwen"][0], 3), "record": 0.48, "where": "07 RESULTS.md", "match": abs(res["nla"]["qwen"][0] - 0.48) < 0.006},
        {"quantity": "TextNLA-Qwen, Oracle Lens templates", "computed": round(res["ol"]["qwen"][0], 3), "record": 0.62, "where": "07 RESULTS.md", "match": abs(res["ol"]["qwen"][0] - 0.62) < 0.006},
        {"quantity": "Gemini, NLA templates", "computed": round(res["nla"]["gemini"][0], 3), "record": 0.97, "where": "07 RESULTS.md", "match": abs(res["nla"]["gemini"][0] - 0.97) < 0.006},
        {"quantity": "Gemini, Oracle Lens templates", "computed": round(res["ol"]["gemini"][0], 3), "record": 0.97, "where": "07 RESULTS.md", "match": abs(res["ol"]["gemini"][0] - 0.97) < 0.006},
        {"quantity": "NLA after regressing out Gemini", "computed": round(res["nla"]["after_gemini"], 3), "record": 0.51, "where": "07 RESULTS.md", "match": abs(res["nla"]["after_gemini"] - 0.51) < 0.006},
        {"quantity": "Oracle Lens after regressing out Gemini", "computed": round(res["ol"]["after_gemini"], 3), "record": 0.47, "where": "07 RESULTS.md", "match": abs(res["ol"]["after_gemini"] - 0.47) < 0.006},
        {"quantity": "NLA after regressing out Qwen", "computed": round(res["nla"]["after_qwen"], 3), "record": 0.61, "where": "07 RESULTS.md", "match": abs(res["nla"]["after_qwen"] - 0.61) < 0.006},
    ]
    S.save(fig, "bk_exp07_col_text_ceiling",
           title="Line position is plain in the text, yet barely readable from the activation",
           caption=("Natural hard-wrapped prose, 160 selection items (10-30 vs 55-85 characters into the line), layer 42. "
                    "Bars: AUROC of the mean log-odds over each reader's three frozen mined phrasings, scored by the reader on "
                    "the activation, or by a text-only model given the passage up to the read token with the same phrasing pairs. "
                    "95% bootstrap CIs, 2,000 resamples clustered by passage. The readers sit near chance (CIs touch 0.5) "
                    "while Gemini reading the text scores 0.97, and the NLA's small signal vanishes once Gemini's score is "
                    "regressed out; base Qwen teacher-forced on the text is a weak ceiling. Caveat: mined templates selected "
                    "on these same items carry a best-of-~7 winner's curse."),
           section="backup:exp07", role="backup",
           sources=[f"experiments/07_template_mining/results/{x}" for x in
                    ("score_nla_nj1_col.json", "score_ol_nj1_col.json", "textnla_local_nj1_col.json",
                     "textnla_gemini_nj1_col.partial.json", "gen_ol_nj1_col.json")],
           checks=checks,
           notes=("Gemini cells are from the budget-capped partial file (7 templates; all 6 selected templates present). "
                  "Gemini value = elicited log-odds of option A, both orders averaged. TextNLA-Qwen uses the warm-start "
                  "instruction in experiments/06_resolving_decomposition/data/nla_instruction.txt (mode=prompt), NOT the "
                  "NLA actor prompt. Residual AUROC = OLS of reader LO on text LO, AUROC of residual. "
                  "NLA phrasings: 'Readout taken late/early in the span of the wrapped line.', 'Mid-sentence continuation "
                  "located near the end of a hard-wrapped line.', 'The token appears far into of the wrapped line.'; "
                  "Oracle Lens: '- deep after a newline', '- line position: late in the line', '- current line depth: deep'."))
    print(json.dumps({k: {kk: (v if isinstance(v, float) else [round(x, 3) for x in v]) for kk, v in d.items()} for k, d in res.items()}, indent=0))
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
