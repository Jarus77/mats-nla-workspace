"""Exp 17 analysis (CPU): multi-candidate metrics per (reader, template) from the mc_*.pt files, with the
exp 13 two-candidate AUROC of the frozen fixed pair as the reference.

    .venv/bin/python analyze17.py --rows tb_tablecol,tb_paren
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from lib17 import confusion, metrics_multi  # noqa: E402
from mats.metrics import auroc_labels  # noqa: E402
from spec17 import CLASS_SETS, N_BOOT, ROWS, SEED  # noqa: E402

EXP13 = HERE.parent / "13_new_rows" / "results"


def two_candidate_reference(row: str, exp13: Path = EXP13) -> dict[str, float]:
    """AUROC of the frozen fixed-label pair per reader from the exp 13 score files (the number under attack)."""
    out = {}
    for reader, cell in (("ol", "ol@42"), ("nla", "nla@42"), ("jlens", "jlens@42")):
        p = exp13 / f"score_{'nla' if reader == 'nla' else 'ol'}_{row}.json"
        if not p.exists():
            continue
        s = json.load(open(p))["scores"].get(cell)
        if not s:
            continue
        key = f"{reader}:{row}:fixed"
        if key in s["lo"]:
            lo = np.asarray(s["lo"][key], float)
            y = np.asarray(s["y"], int)  # 1 = values[0] in exp 13; the LO is values[0] - values[1]
            out[reader] = float(auroc_labels(lo, y))
    return out


def mc_table(row: str, vol_dir: Path, n_boot: int = N_BOOT, reference: dict | None = None) -> dict:
    import torch

    out = {"row": row, "n": None, "excluded": [], "cells": {}, "two_candidate_reference": reference or {}}
    for reader in ("ol", "nla"):
        p = Path(vol_dir) / f"mc_{reader}_{row}.pt"
        if not p.exists():
            continue
        b = torch.load(p, weights_only=False)
        out["n"], out["excluded"] = int(len(b["ids"])), list(b["excluded"])
        for tid, lp in b["lp"].items():
            r = tid.split(":")[0]  # ol / nla / jlens (the J-lens rides in the OL file)
            m = metrics_multi(lp.numpy(), b["truth"], b["y"], CLASS_SETS[row], ROWS[row]["values"],
                              np.asarray(b["clusters"]), n_boot=n_boot, seed=SEED)
            m["argmax_counts"] = confusion(lp.numpy(), b["y"], ROWS[row]["values"])
            out["cells"].setdefault(r, {})[tid] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(ROWS))
    ap.add_argument("--vol", default=str(HERE / "results" / "vol"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--xs", action="store_true", help="cross-scoring tables instead of the multi-candidate ones")
    a = ap.parse_args()
    for row in a.rows.split(","):
        if a.xs:
            res = xs_table(row, Path(a.vol), a.n_boot)
            (HERE / "results" / f"analysis17_xs_{row}.json").write_text(json.dumps(res, indent=1, default=float))
            print(row, json.dumps(res["cells"], default=float)[:400])
            continue
        res = mc_table(row, Path(a.vol), a.n_boot, two_candidate_reference(row))
        (HERE / "results" / f"analysis17_mc_{row}.json").write_text(json.dumps(res, indent=1, default=float))
        print(row, "n", res["n"], "cells", {r: list(t) for r, t in res["cells"].items()}, "ref", res["two_candidate_reference"])


if __name__ == "__main__":
    main()


def xs_table(row: str, vol_dir: Path, n_boot: int = N_BOOT) -> dict:
    """2x2 (items x phrasings) AUROC per reader with cluster-bootstrap CIs, mean over phrasings; B - A log-odds
    against label 1 = values[1]."""
    import torch

    from mats.metrics import auroc_ci

    out = {"row": row, "cells": {}}
    for reader in ("ol", "nla"):
        p = Path(vol_dir) / f"xs_{reader}_{row}.pt"
        if not p.exists():
            continue
        s = torch.load(p, weights_only=False)
        cells = {}
        for cell, lo in s["lo"].items():
            if not lo:
                continue
            on_his = cell.startswith("his") or cell.startswith("jlens_his")
            y = np.asarray(s["his_y"] if on_his else s["our_y"], int)
            cl = np.asarray(s["his_clusters"] if on_his else s["our_clusters"])
            score = np.mean([np.asarray(v, float) for v in lo.values()], axis=0)
            r = auroc_ci(score, y, cl, n_boot=n_boot, seed=SEED)
            cells[cell] = {"point": float(r.auroc), "lo": float(r.lo), "hi": float(r.hi), "n": int(r.n), "n_pairs": len(lo)}
        out["cells"][reader] = cells
    return out
