"""Exp 15 analysis (pure CPU): for each real direction x dose, the paired +/- shift of every
model-side metric and every reader's prompted log-odds, placed against the K isotropic and K
J-complement null contrasts at the same dose norm (r11). q90 cells report the shift only."""
from __future__ import annotations

from typing import Any

import numpy as np

from lib15 import contrast_vs_nulls, paired_shift
from spec15 import DIR_NAMES, NULL_CLASSES, ROWS


def _index(conds: list[tuple]) -> dict[tuple, int]:
    return {tuple(c): i for i, c in enumerate(conds)}


def _series(beh: dict, sc: dict, ci: int) -> dict[str, np.ndarray]:
    m = beh["model"][ci]
    out = {"model_metric": m["metric"].numpy(), "model_chars_to_close": m["chars_to_close"].numpy(),
           "model_lo_close": m["lo_close"].numpy(), "model_lo_newline": m["lo_nl"].numpy(),
           "jlens": beh["jlens"][ci].numpy()}
    for r in ("nla", "ol"):
        lo = sc[r]["lo"][ci].numpy()  # [T, N]
        out[f"{r}_mean"] = lo.mean(0)
        out[f"{r}_fixed"] = lo[0]
        for t in range(1, lo.shape[0]):
            out[f"{r}_t{t}"] = lo[t]
    return out


def analyze_feature(feature: str, spec: dict, beh: dict, sc: dict) -> dict[str, Any]:
    conds = [tuple(c) for c in beh["conds"]]
    idx = _index(conds)
    keys = list(_series(beh, sc, 0))
    cells: dict[str, Any] = {}
    null_pm = {cls: [(_series(beh, sc, idx[("null", f"{cls}_{k}", "r11", 1)]),
                      _series(beh, sc, idx[("null", f"{cls}_{k}", "r11", -1)]))
                     for k in range(sum(1 for c in conds if c[0] == "null" and c[1].startswith(cls) and c[3] == 1))]
               for cls in NULL_CLASSES}
    for d in DIR_NAMES:
        for q in ("r11", "q90"):
            P = _series(beh, sc, idx[("real", d, q, 1)])
            M = _series(beh, sc, idx[("real", d, q, -1)])
            cell = {}
            for k in keys:
                if q == "r11":
                    cell[k] = {cls: contrast_vs_nulls(P[k], M[k], [p[k] for p, _ in null_pm[cls] if not np.isnan(p[k]).all()],
                                                      [m[k] for _, m in null_pm[cls] if not np.isnan(m[k]).all()])
                               for cls in NULL_CLASSES}  # NaN rows = nulls a trimmed reader run skipped
                else:
                    cell[k] = {"real": paired_shift(P[k], M[k])}
            cells[f"{d}:{q}"] = cell
    zero = _series(beh, sc, idx[("zero", None, "r11", 0)])
    clean = beh["clean_nohook"]
    sanity = {"zero_vs_nohook_metric_mean_absdiff": float(np.abs(zero["model_metric"] - clean["metric"].numpy()).mean()),
              "zero_metric_mean": float(zero["model_metric"].mean()),
              "zero_found_rate": float(beh["model"][idx[("zero", None, "r11", 0)]]["found"].float().mean()),
              "zero_by_class": {v: float(zero["model_metric"][spec["y"] == i].mean())
                                for i, v in enumerate(ROWS[feature]["values"])}}
    return {"feature": feature, "metric": beh["metric"], "expect_sign": ROWS[feature]["expect_sign"],
            "geometry": spec["geometry"], "sanity": sanity, "cells": cells, "pairs": {r: sc[r]["pairs"] for r in sc}}


def table_rows(res: dict) -> list[dict]:
    """One row per direction at r11: model metric and reader shifts with z vs the J-complement band."""
    rows = []
    for d in DIR_NAMES:
        c = res["cells"][f"{d}:r11"]
        row = {"direction": d}
        for k in ("model_metric", "model_lo_close", "model_lo_newline", "nla_mean", "nla_fixed", "ol_mean", "ol_fixed", "jlens"):
            r = c[k]["nonj4"]
            row[k] = {"shift": round(r["real"]["mean_shift"], 3), "win": round(r["real"]["paired_win"], 2),
                      "z_nonj": round(r["band"]["z"], 1), "z_iso": round(c[k]["iso"]["band"]["z"], 1)}
        rows.append(row)
    return rows
