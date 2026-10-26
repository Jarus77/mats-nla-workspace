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


def mc_markdown(results: list[dict]) -> str:  # filled in Task 3
    raise NotImplementedError


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
