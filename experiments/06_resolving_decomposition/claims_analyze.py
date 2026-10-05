"""Claim-level tables from data/claims_gemini.jsonl and the kimi-k3 agreement check
(data/claims_kimi.jsonl) -> data/claims_tables.md."""
import json
import random
from collections import Counter

from claims import LABELS
from common import FAMILIES, HERE

TRUE3 = {"prompt_restatement", "prompt_inferable", "target_latent_consistent"}
SHORT = {"prompt_restatement": "restate", "prompt_inferable": "inferable",
         "target_latent_consistent": "target/latent", "unsupported_by_text": "unsupported",
         "contradicted_by_text": "contradicted"}


def collapse(lab: str) -> str:
    return "true" if lab in TRUE3 else ("contra" if lab == "contradicted_by_text" else "unsup")


def tvp(readouts: list[list[str]]) -> float:
    t = sum(l in TRUE3 for r in readouts for l in r)
    c = sum(l == "contradicted_by_text" for r in readouts for l in r)
    return t / (t + c) if t + c else float("nan")


def boot(readouts, fn, b=2000, seed=0):
    rng = random.Random(seed)
    n = len(readouts)
    vs = sorted(fn([readouts[rng.randrange(n)] for _ in range(n)]) for _ in range(b))
    return vs[int(0.025 * b)], vs[int(0.975 * b) - 1]


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "data" / "claims_gemini.jsonl").read_text().splitlines()]
    readers = list(dict.fromkeys(r["reader"] for r in rows))
    md = ["## Claim labels by reader (Gemini 3.8 Flash judge; 100 readouts per reader, 25 per family)",
          "| reader | readouts | claims | claims/readout | " + " | ".join(SHORT[l] for l in LABELS)
          + " | text-verifiable precision [95% CI] | unsupported share | readouts with ≥1 contradicted |",
          "|---|---|---|---|" + "---|" * len(LABELS) + "---|---|---|"]
    fam_md = ["## Text-verifiable precision by family", "| reader | " + " | ".join(FAMILIES) + " |",
              "|---|" + "---|" * len(FAMILIES)]
    out = {}
    for rd in readers:
        rr = [[c["label"] for c in (r["claims"] or [])] for r in rows if r["reader"] == rd]
        allc = [l for r in rr for l in r]
        cnt = Counter(allc)
        lo, hi = boot(rr, tvp)
        p = tvp(rr)
        unsup = cnt["unsupported_by_text"] / len(allc)
        anyc = sum("contradicted_by_text" in r for r in rr)
        out[rd] = dict(n_claims=len(allc), tvp=p, ci=[lo, hi], unsup=unsup,
                       dist={l: cnt[l] / len(allc) for l in LABELS})
        md.append(f"| {rd} | {len(rr)} | {len(allc)} | {len(allc) / len(rr):.1f} | "
                  + " | ".join(f"{cnt[l] / len(allc):.2f}" for l in LABELS)
                  + f" | {p:.2f} [{lo:.2f},{hi:.2f}] | {unsup:.2f} | {anyc}/{len(rr)} |")
        cells = []
        for fam in FAMILIES:
            fr = [[c["label"] for c in (r["claims"] or [])] for r in rows
                  if r["reader"] == rd and r["family"] == fam]
            cells.append(f"{tvp(fr):.2f}")
        fam_md.append(f"| {rd} | " + " | ".join(cells) + " |")
    # agreement
    k = [json.loads(l) for l in (HERE / "data" / "claims_kimi.jsonl").read_text().splitlines()]
    k = [x for x in k if x["kimi"]]
    g5, k5 = [x["gemini"] for x in k], [x["kimi"] for x in k]
    g3, k3 = [collapse(x) for x in g5], [collapse(x) for x in k5]
    rng = random.Random(0)
    bs5, bs3 = [], []
    for _ in range(2000):
        idx = [rng.randrange(len(k)) for _ in k]
        bs5.append(kappa([g5[i] for i in idx], [k5[i] for i in idx]))
        bs3.append(kappa([g3[i] for i in idx], [k3[i] for i in idx]))
    bs5.sort(), bs3.sort()
    agr = ["## Judge agreement: Gemini 3.8 Flash vs kimi-k3 (Fireworks) on 40 claims, stratified by reader x Gemini label",
           "| label set | raw agreement | Cohen's κ [95% bootstrap CI] |", "|---|---|---|",
           f"| 5-way | {sum(a == b for a, b in zip(g5, k5)) / len(k):.2f} | {kappa(g5, k5):.2f} [{bs5[50]:.2f},{bs5[1949]:.2f}] |",
           f"| 3-way (text-true / contradicted / unsupported) | {sum(a == b for a, b in zip(g3, k3)) / len(k):.2f} | {kappa(g3, k3):.2f} [{bs3[50]:.2f},{bs3[1949]:.2f}] |",
           "", "Confusion (rows Gemini, cols kimi-k3):",
           "| Gemini \\ kimi | " + " | ".join(SHORT[l] for l in LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for a in LABELS:
        agr.append(f"| {SHORT[a]} | " + " | ".join(str(sum(x == a and y == b for x, y in zip(g5, k5))) for b in LABELS) + " |")
    # kimi-relabelled precision on the subset is not meaningful (stratified); report kimi's TVP shift direction
    md += ["", *fam_md, "", *agr]
    (HERE / "data" / "claims_tables.md").write_text("\n".join(md), encoding="utf-8")
    (HERE / "data" / "claims_summary.json").write_text(json.dumps(
        {"readers": out, "kappa5": kappa(g5, k5), "kappa3": kappa(g3, k3), "n_val": len(k)}, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
