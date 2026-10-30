"""s1_wsbench_arms: the readout adds nothing the prompt lacks (exps 01 and 06)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import style  # noqa: E402
from style import C  # noqa: E402
from g1_wsb import E01, E06, FAMILIES, PROMPT_ONLY_ARMS, judged_path, load_pass, load_rows, rel  # noqa: E402

ARMS = [  # (arm in exp 06 judged/, label, face, edge)
    ("gem_rsP", "prompt only", C["text"], C["text"]),
    ("gem_resolve_nla", "NLA readout only", C["nla"], C["nla"]),
    ("gem_resolve_ol42", "Oracle Lens readout only", C["ol"], C["ol"]),
    ("gem_rsP_nla", "prompt + NLA readout", C["text"], C["nla"]),
    ("gem_rsP_ol42", "prompt + Oracle Lens\nreadout", C["text"], C["ol"]),
]
RECORD = {"gem_rsP": 0.89, "gem_resolve_nla": 0.60, "gem_resolve_ol42": 0.56, "gem_rsP_nla": 0.88, "gem_rsP_ol42": 0.88}


def paired_diff_ci(a: dict, b: dict) -> tuple[float, float, float]:
    keys = sorted(a)
    x = np.array([a[k] for k in keys], float) - np.array([b[k] for k in keys], float)
    lo, hi = style.bootstrap_ci(np.mean, x)
    return float(x.mean()), lo, hi


def main() -> None:
    passes = {arm: load_pass(E06, arm) for arm, *_ in ARMS}
    checks, stats = [], {}
    for arm, *_ in ARMS:
        k, n = sum(passes[arm].values()), len(passes[arm])
        stats[arm] = style.wilson(k, n) + (k, n)
        checks.append({"quantity": f"pass rate {arm}", "computed": round(k / n, 3), "record": RECORD[arm],
                       "where": "exp 06 RESULTS re-solver check; talk script S1 table", "match": round(k / n, 2) == RECORD[arm]})
    d_nla = paired_diff_ci(passes["gem_rsP_nla"], passes["gem_rsP"])
    d_ol = paired_diff_ci(passes["gem_rsP_ol42"], passes["gem_rsP"])

    # item-level: reader passes unreachable by all eight prompt-only texts
    po = {a: load_pass(E06, a) for a in PROMPT_ONLY_ARMS}
    reach = {}
    for reader, arm in (("nla", "nla_L42"), ("ol", "olens_L42")):
        rows = load_rows(E01, arm)
        hits = [k for k, r in rows.items() if r["pass"]]
        hard = [k for k in hits if not any(po[a][k] for a in PROMPT_ONLY_ARMS)]
        zh = [k for k in hard if "zh" in rows[k].get("unit_langs", {}).get("concept", [])]
        reach[reader] = (len(hits), len(hard), len(zh))
    checks += [
        {"quantity": "NLA@42 passes / unreachable by 8 prompt-only texts", "computed": f"{reach['nla'][0]} / {reach['nla'][1]}",
         "record": "187 / 0", "where": "exp 06 RESULTS sec 2; talk S1", "match": reach["nla"][:2] == (187, 0)},
        {"quantity": "Oracle Lens at layer 42 passes / unreachable", "computed": f"{reach['ol'][0]} / {reach['ol'][1]}",
         "record": "159 / 4", "where": "exp 06 RESULTS sec 2; talk S1", "match": reach["ol"][:2] == (159, 4)},
        {"quantity": "Oracle Lens unreachable items whose concept credit came from a Chinese form", "computed": reach["ol"][2],
         "record": "all 4 (Chinese code-switching)", "where": "exp 06 RESULTS sec 2", "match": reach["ol"][2] == 4},
    ]

    style.slide_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.3), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.55})
    fig.subplots_adjust(left=0.335, right=0.97, bottom=0.15, top=0.89)
    ys = [4.8, 3.8, 2.8, 1.0, 0.0]  # the two prompt+readout bars set apart below
    for y, (arm, label, face, edge) in zip(ys, ARMS):
        p, lo, hi, *_ = stats[arm]
        both = face != edge
        ax.barh(y, p, 0.72, color=C["text_pale"] if both else face, edgecolor=edge, linewidth=3 if both else 0,
                zorder=2)
        ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], color=C["ink"], capsize=4, lw=1.4, zorder=3)
        ax.text(hi + 0.02, y, f"{p:.2f}", ha="left", va="center", fontsize=15, fontweight="bold")
    ax.set_yticks(ys, [a[1] for a in ARMS])
    ax.set_ylim(-0.55, 5.35)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("pass rate (higher = better)")
    ax.set_title("A. One judge, different inputs", loc="left", x=-0.95, pad=8)
    p0 = stats["gem_rsP"][0]
    ax.axvline(p0, color=C["text"], lw=1.4, ls="--", zorder=1)
    ax.text(0.02, 1.9, "prompt + readout:\nno gain over prompt", va="center", ha="left", fontsize=15,
            color=C["ink"], linespacing=1.05, zorder=4, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    rows = [("ol", "Oracle Lens"), ("nla", "NLA")]
    for j, (r, lab) in enumerate(rows):
        n_hit, n_hard, _ = reach[r]
        bx.barh(j, n_hit - n_hard, 0.55, color=style.C[f"{r}_pale"], zorder=2)
        bx.barh(j, n_hard, 0.55, left=n_hit - n_hard, color=C[r], zorder=2)
        bx.text(n_hit + 4, j, f"{n_hard} of {n_hit}", va="center", fontsize=15, fontweight="bold", color=C[r])
    bx.set_yticks(range(len(rows)), [r[1] for r in rows])
    bx.set_xlim(0, 285)
    bx.set_xticks([0, 100, 200])
    bx.set_xlabel("reader passes (of 400)")
    bx.set_title("B. Passes no prompt-only text got", loc="left", x=-0.5, pad=8)
    bx.text(3, -0.42, "the 4: Chinese spellings\nthe regex accepts", fontsize=15, color=C["ol"], va="top",
            linespacing=1.05)
    bx.text(3, 1.38, "dark = missed by all 8\nprompt-only texts", fontsize=15, color=C["muted"], va="bottom",
            linespacing=1.05)
    bx.set_ylim(-1.15, 2.0)

    srcs = [rel(judged_path(E06, a, f)) for a, *_ in ARMS for f in FAMILIES[:1]]
    srcs = [rel(E06 / "judged" / a) + "/*/results.json" for a, *_ in ARMS] + \
           [rel(E06 / "judged" / a) + "/*/results.json" for a in PROMPT_ONLY_ARMS] + \
           [rel(E01 / "judged" / a) + "/*/results.json" for a in ("nla_L42", "olens_L42")]
    style.save(
        fig, "s1_wsbench_arms",
        title="The readout adds nothing the prompt lacks",
        caption=("A: one judge model (Gemini 3.8 Flash, one naming instruction, k=1) answers WorkspaceBench from "
                 "the prompt, a reader's layer-42 readout, or both; regex-scored, 400 items over 4 multi-token families, "
                 f"Wilson 95% CIs. Prompt + readout minus prompt: NLA {d_nla[0]:+.3f} [{d_nla[1]:+.3f}, {d_nla[2]:+.3f}], "
                 f"Oracle Lens {d_ol[0]:+.3f} [{d_ol[1]:+.3f}, {d_ol[2]:+.3f}] (paired item bootstrap). "
                 "B: of each reader's own regex passes, the number missed by all eight prompt-only texts "
                 "(Qwen x3 and Gemini x1 under the NLA register and the benchmark template); the 4 Oracle Lens items are "
                 "credited through Chinese forms of the concept. Caveat: the eight text arms are best-of-8 against k=1 readers."),
        section="S1", role="main", sources=srcs, checks=checks,
        notes=("Judge prompt (prompt arm): 'Below is a text that a language model was reading, cut off at its final "
               "token. <text>...</text> [readout arms add: 'Below is also a description, produced by an interpretability "
               "tool, of what the model was internally representing at that final token. <description>...</description>'] "
               "Work out what the model was thinking about at that token. Write a short summary (at most ~150 words) that "
               "(1) states what the text is about, (2) names the specific entity, term, word or answer the model was most "
               "likely representing or about to produce, written out in full and spelled correctly, and (3) states the "
               "language of the text.' Readout-only arms use RESOLVE_TEMPLATE in experiments/06_resolving_decomposition/"
               "common.py (same three asks, 'You do not see the text itself'). Reader passes in B come from exp 01's own "
               "judged layer-42 files; the 4 Oracle Lens items are mlh-ar-sandpaper, mlh-id-hailstone, mlh-pl-strzykawka, "
               "mlh-vi-clothesiron (concept unit hit in 'zh')."),
    )


if __name__ == "__main__":
    main()
