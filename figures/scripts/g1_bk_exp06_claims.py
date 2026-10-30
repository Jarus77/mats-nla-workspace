"""bk_exp06_claims: a text-only model's claims are more often text-true than either reader's (exp 06)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402
from style import C, ROOT  # noqa: E402

F = ROOT / "experiments" / "06_resolving_decomposition" / "data" / "claims_gemini.jsonl"
SUMMARY = ROOT / "experiments" / "06_resolving_decomposition" / "data" / "claims_summary.json"
READERS = [("NLA@42", "NLA readout", "nla"), ("OL@42", "Oracle Lens readout", "ol"),
           ("TextNLA(prompt,Qwen)", "text-only model\n(saw the prompt)", "text")]
TRUE = ["prompt_restatement", "prompt_inferable", "target_latent_consistent"]
LABELS = [("prompt_restatement", "restates the prompt"), ("prompt_inferable", "inferable from the prompt"),
          ("target_latent_consistent", "consistent with the gold answer"),
          ("unsupported_by_text", "unsupported"), ("contradicted_by_text", "contradicted by the prompt")]
REC = {"NLA@42": 0.62, "OL@42": 0.52, "TextNLA(prompt,Qwen)": 0.92}


def tvp_of(readouts: list[list[str]]) -> float:
    flat = [l for r in readouts for l in r]
    t = sum(l in TRUE for l in flat)
    c = sum(l == "contradicted_by_text" for l in flat)
    return t / (t + c) if t + c else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in F.read_text().splitlines() if l.strip()]
    summ = json.loads(SUMMARY.read_text())["readers"]
    checks, res = [], {}
    for key, *_ in READERS:
        rr = [[c["label"] for c in r["claims"]] for r in rows if r["reader"] == key]
        flat = [l for r in rr for l in r]
        idx = np.arange(len(rr))
        lo, hi = style.bootstrap_ci(lambda ix: tvp_of([rr[i] for i in ix]), idx)
        res[key] = dict(tvp=tvp_of(rr), lo=lo, hi=hi, n=len(flat), nr=len(rr),
                        dist={k: sum(l == k for l in flat) / len(flat) for k, _ in LABELS})
        checks.append({"quantity": f"TVP {key}", "computed": round(res[key]["tvp"], 3), "record": REC[key],
                       "where": "exp 06 RESULTS sec 5", "match": round(res[key]["tvp"], 2) == REC[key]})
        checks.append({"quantity": f"n claims {key}", "computed": len(flat), "record": summ[key]["n_claims"],
                       "where": "claims_summary.json", "match": len(flat) == summ[key]["n_claims"]})

    fig, (ax, bx) = plt.subplots(1, 2, figsize=style.FULL, gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.32})
    shades = {"prompt_restatement": "#4F6D7A", "prompt_inferable": "#8FA7B3", "target_latent_consistent": "#C9D3D9",
              "unsupported_by_text": "#EEEEEE", "contradicted_by_text": C["ink"]}
    y = np.arange(len(READERS))[::-1]
    for yi, (key, lab, col) in zip(y, READERS):
        left = 0.0
        for k, kl in LABELS:
            v = res[key]["dist"][k]
            ax.barh(yi, v, 0.6, left=left, color=shades[k], edgecolor="white", lw=1,
                    label=kl if yi == y[0] else None)
            if v >= 0.07:
                ax.text(left + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=11.5,
                        color="white" if k in ("contradicted_by_text", "prompt_restatement") else C["ink"])
            left += v
    ax.set_yticks(y, [r[1] for r in READERS], fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of atomic claims")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=11.5)
    ax.set_title("A. What the claims are", loc="left", fontsize=15)
    for yi, (key, lab, col) in zip(y, READERS):
        r = res[key]
        bx.errorbar(r["tvp"], yi, xerr=[[r["tvp"] - r["lo"]], [r["hi"] - r["tvp"]]], fmt="o", ms=12, color=C[col],
                    capsize=4, lw=2)
        bx.text(r["hi"] + 0.02, yi + 0.18, f"{r['tvp']:.2f}", fontsize=13, fontweight="bold", color=C[col])
    bx.set_yticks(y, [""] * len(y))
    bx.set_xlim(0.4, 1.02)
    bx.set_ylim(y[-1] - 0.5, y[0] + 0.5)
    bx.set_xlabel("text-verifiable precision\n(true / (true + contradicted))")
    bx.set_title("B. Precision", loc="left", fontsize=15)
    style.save(
        fig, "bk_exp06_claims",
        title="A text-only model's claims are more often true of the prompt than either reader's",
        caption=("100 readouts per source (25 per WorkspaceBench family, same items), split by a judge model (Gemini 3.8 Flash, "
                 "T=0) into at most 12 atomic claims each and labelled against the prompt and gold answer "
                 f"(NLA {res['NLA@42']['n']}, Oracle Lens {res['OL@42']['n']}, text-only {res['TextNLA(prompt,Qwen)']['n']} claims). "
                 "B: precision = text-true / (text-true + contradicted), 95% bootstrap CI over readouts. The NLA's "
                 "contradictions are mostly near-miss quotes of the prompt; Oracle Lens's are wrong facts plus 28% unsupported. "
                 "Caveat: the judge sees the gold answer; agreement with a second judge is kappa 0.72 (5-way), no human labels."),
        section="backup:exp06", role="backup",
        sources=["experiments/06_resolving_decomposition/data/claims_gemini.jsonl",
                 "experiments/06_resolving_decomposition/data/claims_summary.json"],
        checks=checks,
        notes=("Text-only arm = base Qwen3.6-27B given the prompt under the NLA warm-start instruction "
               "(data/nla_instruction.txt), sample s0. Codebook precedence: contradicted > restatement > target/latent > "
               "inferable > unsupported (claims.py)."),
    )


if __name__ == "__main__":
    main()
