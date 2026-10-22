"""10 operand gate, stage 1: CPU analysis of results/gate.json -> markdown table + labels (printed; pasted into
RESULTS.md). Pilot, exploratory; prompted by an external result seen 2026-10-12."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RANKS = ("384", "1536")


def fmt_ci(x: float, ci: list[float]) -> str:
    return f"{x:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def enr(g2: dict, key: str) -> str:
    return " / ".join(f"{g2[r][key]['median']:.2f} ({g2[r][key]['min']:.2f}-{g2[r][key]['max']:.2f})" for r in RANKS)


def slot_rows(out: dict) -> list[str]:
    rows = []
    for slot, v in out["slots"].items():
        g1, g2, g3 = v["G1"], v["G2"], v["G3"]
        rows.append(
            f"| {slot} | {g1['acc']:.2f} [{g1['acc_ci95'][0]:.2f}, {g1['acc_ci95'][1]:.2f}] | "
            f"{fmt_ci(g1['macro_auroc'], g1['macro_auroc_ci95'])} | {enr(g2, 'class_mean')} | {enr(g2, 'probe_weight')} | "
            f"{enr(g2, 'shuffled_label_class_mean')} | {g3['stat']:.3f} [{g3['null_band95'][0]:.3f}, {g3['null_band95'][1]:.3f}] | "
            f"{g3['true_label_in_top10']:.2f} | **{v['label']}** |")
    return rows


def main(path: str) -> None:
    out = json.loads(Path(path).read_text())
    band = out["random_band_iso_K100"]
    print(f"n = {out['n']}; capture {out['t_capture_s']} s; total {out['t_total_s']} s; lens {out['lens_path']}")
    print(f"tokenisation: {out['capture']}")
    print(f"next token (ours, top 5): {out['next_token']['ours_top5']}; agreement with his model_next: "
          f"{out['next_token']['agree_with_his_model_next']:.2f}")
    print(f"random isotropic band (K=100) enrichment: " + "; ".join(
        f"r={r}: {band[r]['median']:.2f} [{band[r]['p2.5']:.2f}, {band[r]['p97.5']:.2f}]" for r in RANKS))
    print(f"activation-variance enrichment in J: {out['acts_enrichment']}")
    print()
    print("| slot | G1 acc (chance 0.10) | G1 macro AUROC | G2 class-mean enrichment r*/4r* median (range) | "
          "G2 probe-weight | G2 shuffled-label ref | G3 J-Lens OvR stat [null 95%] | true digit in lens top-10 | label |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(slot_rows(out)))
    print()
    for k in ("exact_a", "exact_b", "exact_ans"):
        e = out[k]
        print(f"- {k}: {e['acc']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]")
    op = out["op"]
    print(f"- op: G1 AUROC {fmt_ci(op['G1']['auroc'], op['G1']['ci95'])} acc {op['G1']['acc']:.2f}; "
          f"G2 {', '.join(f'{r}: {op['G2'][r]['diffmeans']:.2f}' for r in RANKS)}; "
          f"G3 stat {op['G3']['stat']:.3f} null {op['G3']['null_band95']} readable={op['G3']['j_readable']}")
    for name, m in out["magnitude"].items():
        print(f"- magnitude {name}: CV Pearson r {m['cv_pearson_r']:.2f}, MAE {m['cv_mae']:.1f}; enrichment "
              f"{', '.join(f'{r}: {m['enrichment'][r]:.2f}' for r in RANKS)}")
    print()
    for slot, v in out["slots"].items():
        print(f"- {slot} lens top-10 common: {v['G3']['lens_top10_common'][:8]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "results" / "gate.json"))
