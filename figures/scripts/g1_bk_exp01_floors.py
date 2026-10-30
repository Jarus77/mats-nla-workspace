"""bk_exp01_floors: the published prompt-only floors are too low (exps 01 and 06 follow-up)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402
from style import C, ROOT  # noqa: E402
from g1_wsb import E01, E06, FAMILIES, FAMILY_LABEL, load_pass, rel  # noqa: E402

PUB = ROOT / "third_party" / "workspace-bench" / "evals" / "baselines" / "prompt_only.json"
REC_PUB = {"typo_mt": 0.67, "multilingual_mt": 0.42, "multihop_mt": 0.69, "basic_readout_mt": 0.59}
REC_OFF = {"typo_mt": 0.83, "multilingual_mt": 0.47, "multihop_mt": 0.84, "basic_readout_mt": 0.80}
REC_WSB = {"typo_mt": (0.85, 0.83, 0.83), "multilingual_mt": (0.58, 0.55, 0.51), "multihop_mt": (0.82, 0.79, 0.88),
           "basic_readout_mt": (0.77, 0.78, 0.80)}


def rate(passes: dict, fam: str) -> tuple[int, int]:
    v = [p for (f, _), p in passes.items() if f == fam]
    return sum(v), len(v)


def main() -> None:
    pub = json.loads(PUB.read_text())
    off = load_pass(E06, "floor_think_off")
    wsb = [load_pass(E06, f"qwen_prompt_wsb_s{i}") for i in range(3)]
    nla = load_pass(E01, "nla_L42")
    checks = []
    fig, ax = plt.subplots(figsize=style.FULL)
    w = 0.26
    for i, fam in enumerate(FAMILIES):
        pk = round(pub[fam]["rate"] * pub[fam]["n_items"])
        pn = pub[fam]["n_items"]
        p0, l0, h0 = style.wilson(pk, pn)
        k1, n1 = rate(off, fam)
        p1, l1, h1 = style.wilson(k1, n1)
        draws = [rate(d, fam)[0] / rate(d, fam)[1] for d in wsb]
        kn, nn = rate(nla, fam)
        for xi, (p, lo, hi), col, lab in ((i - w / 2, (p0, l0, h0), C["text_pale"], "published floor"),
                                          (i + w / 2, (p1, l1, h1), C["text"], "our re-run, same template, thinking off")):
            ax.bar(xi, p, w * 0.95, color=col, zorder=2, label=lab if i == 0 else None)
            ax.errorbar(xi, p, yerr=[[p - lo], [hi - p]], color=C["ink"], capsize=3, lw=1.1, zorder=3)
            ax.text(xi, hi + 0.012, f"{p:.2f}", ha="center", va="bottom", fontsize=12.5, fontweight="bold")
        ax.scatter([i + w / 2 + w * 0.7] * 3, draws, s=28, color=C["text"], edgecolor="white", zorder=4,
                   label="three earlier draws of the same arm" if i == 0 else None)
        ax.plot([i - w - 0.04, i + w + 0.2], [kn / nn] * 2, color=C["nla"], lw=2.2, zorder=5,
                label="NLA, layer 42 (reader)" if i == 0 else None)
        if fam == "multihop_mt":
            ax.text(i + w + 0.22, kn / nn, f" NLA {kn / nn:.2f}", va="center", fontsize=12, color=C["nla"])
        checks += [
            {"quantity": f"{fam} published floor", "computed": pub[fam]["rate"], "record": REC_PUB[fam],
             "where": "exp 01 RESULTS replication table", "match": round(pub[fam]["rate"], 2) == REC_PUB[fam]},
            {"quantity": f"{fam} re-run thinking off", "computed": round(p1, 2), "record": REC_OFF[fam],
             "where": "exp 06 RESULTS floor verification", "match": round(p1, 2) == REC_OFF[fam]},
            {"quantity": f"{fam} T_wsb s0/s1/s2", "computed": [round(d, 2) for d in draws], "record": list(REC_WSB[fam]),
             "where": "exp 06 RESULTS floor verification", "match": tuple(round(d, 2) for d in draws) == REC_WSB[fam]},
        ]
    diffs = [rate(off, f)[0] / rate(off, f)[1] - pub[f]["rate"] for f in FAMILIES]
    ax.set_xticks(range(len(FAMILIES)), [FAMILY_LABEL[f] for f in FAMILIES])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("prompt-only pass rate\n(text model sees only the prompt)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12.5)
    style.save(
        fig, "bk_exp01_floors",
        title="The published prompt-only floors are too low; re-run with thinking off they rise by up to 0.21",
        caption=("Prompt-only floor per WorkspaceBench family: the frozen published value against our re-run of the "
                 "authors' own template and sampling (base Qwen3.6-27B, chat rendering with thinking off, k=1, n=100 per "
                 "family, regex scorer), Wilson 95% CIs; dots are three earlier draws of the same arm; the red line is the "
                 f"NLA's own layer-42 score. Re-run minus published: {', '.join(f'{d:+.2f}' for d in diffs)}. Caveat: the "
                 "published generator and its rendering are not public; thinking-on and raw rendering come closer but do "
                 "not reproduce it either."),
        section="backup:exp01", role="backup",
        sources=[rel(PUB), rel(E06 / "judged" / "floor_think_off") + "/*/results.json",
                 *[rel(E06 / "judged" / f"qwen_prompt_wsb_s{i}") + "/*/results.json" for i in range(3)],
                 rel(E01 / "judged" / "nla_L42") + "/*/results.json"],
        checks=checks,
        notes=("Template (from prompt_only.json _source): 'Below is a prompt that was given to a language model, cut off at a "
               "specific token. The token in question is the final token of the text below: {token!r}. <prompt>{prefix}"
               "</prompt> What would a language model be thinking about on this token? Please write me a summary for this "
               "token.' Published CI recomputed as Wilson from rate x n_items."),
    )


if __name__ == "__main__":
    main()
