"""Cluster-bootstrap confidence intervals for the earlier experiments, from their local .pt files.

  ci15: exp 15 paired +/- mean shift per (direction, r11) and series, with a CI over items and the
        null-band z (K isotropic / J-complement nulls, NaN null rows dropped as analyze15 does);
        z_lo / z_hi are the CI endpoints divided by the null SD.
  ci16: exp 16 follows_j AUROC per reader x variant (mean phrasing and fixed label), CI over clusters.

    .venv/bin/python ci_prior.py --rows15 tb_tablecol,tb_paren --rows16 tb_tablecol,tb_paren,ij2_lang,bk_listidx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for p in (HERE, EXP / "15_ar_steering", EXP / "16_difference_term", HERE.parent.parent / "src"):
    sys.path.insert(0, str(p))

from analyze15 import _index, _series  # noqa: E402
from lib15 import band, paired_shift  # noqa: E402
from lib16 import _auroc, follow  # noqa: E402
from mats.metrics import cluster_bootstrap  # noqa: E402
from spec15 import DIR_NAMES, NULL_CLASSES  # noqa: E402
from spec16 import VARIANTS  # noqa: E402

VOL15 = EXP / "15_ar_steering" / "results" / "vol"
VOL16 = EXP / "16_difference_term" / "results" / "vol"


def _load15(row: str, vol: Path):  # noqa: ANN202
    import torch

    spec = torch.load(vol / f"spec15_{row}.pt", weights_only=False)
    parts = [torch.load(p, weights_only=False) for p in sorted(vol.glob(f"behave15_{row}_c*.pt"))]
    beh = {**parts[0], "model": {ci: m for p in parts for ci, m in p["model"].items()}}
    sc = {r: torch.load(vol / f"score15_{r}_{row}.pt", weights_only=False) for r in ("nla", "ol")}
    return spec, beh, sc


def _shift_ci(plus: np.ndarray, minus: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int) -> dict:
    d = np.asarray(plus, float) - np.asarray(minus, float)
    lo, hi = cluster_bootstrap(lambda i: float(d[i].mean()), clusters, n_boot=n_boot, seed=seed)
    return {**paired_shift(plus, minus), "lo": float(lo), "hi": float(hi)}


def ci15(row: str, vol: Path, n_boot: int = 2000, seed: int = 17) -> dict:
    spec, beh, sc = _load15(row, vol)
    conds = [tuple(c) for c in beh["conds"]]
    idx = _index(conds)
    clusters = np.array([i.rsplit("_", 1)[0] for i in spec["ids"]])
    keys = list(_series(beh, sc, 0))
    nulls = {cls: [] for cls in NULL_CLASSES}
    for cls in NULL_CLASSES:
        k = 0
        while (("null", f"{cls}_{k}", "r11", 1) in idx):
            P = _series(beh, sc, idx[("null", f"{cls}_{k}", "r11", 1)])
            M = _series(beh, sc, idx[("null", f"{cls}_{k}", "r11", -1)])
            nulls[cls].append((P, M))
            k += 1
    cells = {}
    for d in DIR_NAMES:
        P = _series(beh, sc, idx[("real", d, "r11", 1)])
        M = _series(beh, sc, idx[("real", d, "r11", -1)])
        cell = {}
        for key in keys:
            c = _shift_ci(P[key], M[key], clusters, n_boot, seed)
            for cls in NULL_CLASSES:
                ns = [paired_shift(p[key], m[key])["mean_shift"] for p, m in nulls[cls]
                      if not (np.isnan(p[key]).all() or np.isnan(m[key]).all())]
                b = band(c["mean_shift"], ns)
                sd = b["null_sd"]
                c[f"z_{cls}"] = b["z"]
                c[f"k_{cls}"] = b["k"]
                c[f"z_lo_{cls}"] = float((c["lo"] - b["null_mean"]) / sd) if sd and sd > 0 else float("nan")
                c[f"z_hi_{cls}"] = float((c["hi"] - b["null_mean"]) / sd) if sd and sd > 0 else float("nan")
            cell[key] = c
        cells[f"{d}:r11"] = cell
    return {"feature": row, "n": int(len(clusters)), "n_boot": n_boot, "cells": cells}


def ci16(row: str, vol: Path, n_boot: int = 2000, seed: int = 17) -> dict:
    import torch

    s1 = torch.load(vol / f"s1_{row}.pt", weights_only=False)
    s2p = vol / f"s2_{row}.pt"
    ol_lo = torch.load(s2p, weights_only=False)["ol_lo"] if s2p.exists() else None
    y, j, clusters = np.asarray(s1["y"]), np.asarray(s1["j"]), np.asarray(s1["clusters"])
    out = {"feature": row, "n": int(len(y)), "n_boot": n_boot, "readers": {}}
    for name, lo in (("nla", s1["nla_lo"]), ("ol", ol_lo)):
        if lo is None:
            continue
        out["readers"][name] = {}
        for v in VARIANTS:
            if v not in lo:
                continue
            arr = lo[v].numpy()
            res = {}
            for kind, s in (("mean", arr.mean(0)), ("fixed", arr[0])):
                point = follow(s, y, j)["follows_j"]
                yj = y[j]  # the partner's label per row; resampling rows keeps each row's own partner label
                lo_, hi_ = cluster_bootstrap(lambda i, s=s: _auroc(s[i], yj[i]), clusters, n_boot=n_boot, seed=seed)
                res[kind] = {"point": float(point), "lo": float(lo_), "hi": float(hi_)}
            out["readers"][name][v] = res
    return out


def markdown(r15: list[dict], r16: list[dict]) -> str:
    lines = ["## Exp 15: paired shift at r11 with 95% cluster-bootstrap CI and the isotropic-null z", "",
             "| row | direction | series | shift | CI | z (iso) | z of CI ends |", "|---|---|---|---:|---|---:|---|"]
    series = {"tb_tablecol": ["model_lo_newline", "nla_mean", "ol_mean", "jlens"],
              "tb_paren": ["model_lo_close", "nla_mean", "ol_mean", "jlens"]}
    for r in r15:
        for d in DIR_NAMES:
            for key in series.get(r["feature"], ["nla_mean", "ol_mean"]):
                c = r["cells"][f"{d}:r11"][key]
                lines.append(f"| {r['feature']} | {d} | {key} | {c['mean_shift']:+.3f} | [{c['lo']:+.3f}, {c['hi']:+.3f}] | "
                             f"{c['z_iso']:+.1f} | [{c['z_lo_iso']:+.1f}, {c['z_hi_iso']:+.1f}] |")
    lines += ["", "## Exp 16: follows_j AUROC (mean phrasing) with 95% cluster-bootstrap CI", "",
              "| row | reader | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---|" * len(VARIANTS)]
    for r in r16:
        for name, by_v in r["readers"].items():
            cells = [f"{by_v[v]['mean']['point']:.2f} [{by_v[v]['mean']['lo']:.2f}, {by_v[v]['mean']['hi']:.2f}]"
                     if v in by_v else "" for v in VARIANTS]
            lines.append(f"| {r['feature']} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows15", default="tb_tablecol,tb_paren")
    ap.add_argument("--rows16", default="tb_tablecol,tb_paren,ij2_lang,bk_listidx")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    r15 = [ci15(r, VOL15, a.n_boot) for r in a.rows15.split(",") if r]
    r16 = [ci16(r, VOL16, a.n_boot) for r in a.rows16.split(",") if r]
    for r in r15:
        (out / f"ci15_{r['feature']}.json").write_text(json.dumps(r, indent=1, default=float))
    for r in r16:
        (out / f"ci16_{r['feature']}.json").write_text(json.dumps(r, indent=1, default=float))
    (out / "ci_prior.md").write_text(markdown(r15, r16))
    print((out / "ci_prior.md").read_text())


if __name__ == "__main__":
    main()
