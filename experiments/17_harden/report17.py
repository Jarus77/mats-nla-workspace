"""Exp 17 report: markdown tables for the multi-candidate re-score and the vocabulary readout.

    .venv/bin/python report17.py --analysis results/analysis17_mc_*.json --out results/mc17.md
    .venv/bin/python report17.py --vocab results/vocab17_*.json --out results/vocab17.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "15_ar_steering"))
from lib15 import band  # noqa: E402
from spec17 import ROWS as _ROWS  # noqa: E402

ROWS_C = {k: v["n_cands"] for k, v in _ROWS.items()}

PEAK_Z = 3.0  # spec: lexical if the peak ratio is > 3 null-SDs above the isotropic directions


def _words(tokens: list[str]) -> set[str]:
    return {t.strip().lower() for t in tokens}


def vocab_markdown(results: list[dict]) -> str:
    lines = []
    for r in results:
        null_peaks = [v["peak_ratio"] for k, v in r["nulls"].items() if k.startswith("iso_")]
        lines += [f"## {r['feature']} (gain {r['gain']}; z against {len(null_peaks)} isotropic directions)", "",
                  "| direction | peak ratio | z | top-50 mass | top positive tokens | top negative tokens |",
                  "|---|---:|---:|---:|---|---|"]
        for d, e in r["dirs"].items():
            b = band(e["peak_ratio"], null_peaks)
            lines.append(f"| {d} | {e['peak_ratio']:.1f} | {b['z']:+.1f} | {e['top_mass']:.4f} | "
                         f"{' '.join(repr(t) for t in e['top_pos_tokens'][:10])} | "
                         f"{' '.join(repr(t) for t in e['top_neg_tokens'][:10])} |")
        s = r["dirs"].get("arstmt_raw")
        if s is not None:
            z = band(s["peak_ratio"], null_peaks)["z"]
            hit = sorted(_words(s["top_pos_tokens"][:10]) & set(r["content_words"]))
            yes = z > PEAK_Z and bool(hit)
            lines.append("")
            lines.append(f"**Statement direction lexical: {'yes' if yes else 'no'}** (peak-ratio z {z:+.1f}, "
                         f"statement words in the top-10: {hit or 'none'}).")
        lines.append("")
    return "\n".join(lines)


def _fmt(c: dict) -> str:
    return f"{c['point']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"


def mc_verdict(res: dict) -> str:
    """The spec's declared criteria, applied to the NLA's value-slot cell against OLens's."""
    row, cells = res["row"], res["cells"]
    nla = next(iter(cells.get("nla", {}).values()), None)
    if nla is None:
        return "verdict: pending (NLA cell missing)"
    ol_best = max((t for t in cells.get("ol", {}).values()), key=lambda t: t["top1"]["point"], default=None)
    if row == "tb_tablecol":
        ok = (nla["top1"]["lo"] > 2 / 6 and nla["auroc_lse"]["lo"] > 0.80
              and (ol_best is None or (nla["top1"]["point"] > ol_best["top1"]["point"]
                                       and nla["auroc_lse"]["point"] > ol_best["auroc_lse"]["point"])))
        return ("verdict: **survives** (NLA six-way top-1 CI lower bound above 0.33, log-sum-exp AUROC lower bound "
                "above 0.80, both above OLens)" if ok else
                "verdict: **two-candidate artefact** by the declared rule; the claim retreats to the spontaneous columns")
    ok = nla["top1"]["lo"] > 0.50
    return ("verdict: **survives** (NLA four-way top-1 CI lower bound above 0.50)" if ok else
            "verdict: **two-candidate artefact** by the declared rule (NLA four-way top-1 CI lower bound at or below 0.50)")


def mc_markdown(results: list[dict]) -> str:
    lines = []
    for res in results:
        c = ROWS_C.get(res["row"], 0)
        lines += [f"## {res['row']}: {res['n']} items, {len(res['excluded'])} excluded, {c} candidates", "",
                  "| reader | phrasing | top-1 (chance) | MRR | class AUROC over all candidates | Spearman(argmax, truth) | two-candidate reference | argmax counts per class |",
                  "|---|---|---|---|---|---:|---:|---|"]
        for reader in ("nla", "ol", "jlens"):
            for tid, m in res["cells"].get(reader, {}).items():
                sp = "" if m["spearman_argmax"] is None else f"{m['spearman_argmax']:+.2f}"
                ref = res["two_candidate_reference"].get(reader)
                lines.append(f"| {reader} | {tid} | {_fmt(m['top1'])} ({m['top1']['chance']:.2f}) | {_fmt(m['mrr'])} | "
                             f"{_fmt(m['auroc_lse'])} | {sp} | {'' if ref is None else f'{ref:.2f}'} | "
                             + "; ".join(f"{k}: {v}" for k, v in m.get("argmax_counts", {}).items()) + " |")
        lines += ["", mc_verdict(res), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", nargs="*", default=[])
    ap.add_argument("--vocab", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.vocab:
        md = vocab_markdown([json.load(open(p)) for p in a.vocab])
    else:
        md = mc_markdown([json.load(open(p)) for p in a.analysis])
    Path(a.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
