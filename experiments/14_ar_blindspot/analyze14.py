"""Exp 14 analysis for one row: how much of the concept each reconstruction keeps.

For every (ruler, layer, input) we report: raw FVE and retrieval@1 (does the ruler reconstruct
anything), frozen-probe AUROC on the reconstruction and on the error term (does it keep the
concept), a retrained probe on the reconstruction (is the concept linearly present at all), the
class-mean gap retained along the true direction, and the statement shifts (does feeding the ruler
"first column" vs "fifth column" move the reconstruction along the concept at all).
"""
from __future__ import annotations

import numpy as np
import torch

from ar_lib import (frozen_probe_retention, fve_raw, gap_retention, retrained_probe_auroc,
                    retrieval_at_1, statement_shift)
from spec14 import K_GENS, NLA_LAYER, OL_AR_LAYERS, ROWS


def _labels(feature: str, items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    v0 = ROWS[feature]["values"][0]
    y = np.array([int(it["value"] == v0) for it in items])
    groups = np.array([it.get("cluster", it["id"]) for it in items])
    return y, groups


def _block(true: torch.Tensor, rec: torch.Tensor, y: np.ndarray, g: np.ndarray) -> dict:
    out = {"fve": fve_raw(rec, true), "ret1": retrieval_at_1(rec, true),
           "probe": frozen_probe_retention(true, rec, y, g),
           "probe_retrained_on_recon": retrained_probe_auroc(rec, y, g),
           "gap": gap_retention(true, rec, y)}
    return out


def _mean_k(rec: torch.Tensor) -> torch.Tensor:
    return rec.float().mean(1) if rec.dim() == 3 else rec.float()


def analyze_row(feature: str, items: list[dict], acts: dict[int, torch.Tensor], out_dir: str) -> dict:
    y, g = _labels(feature, items)
    res = {"feature": feature, "n": len(items), "role": ROWS[feature]["role"], "nla": {}, "ol": {}}

    nb = torch.load(f"{out_dir}/recon14_nla_{feature}.pt", weights_only=False)
    true42 = acts[NLA_LAYER]
    res["nla"] = {"layer": NLA_LAYER, "nla_expl": _block(true42, _mean_k(nb["recon"]), y, g),
                  "nla_expl_sample0": _block(true42, nb["recon"][:, 0], y, g),
                  "stmt": {sid: statement_shift(true42, y, nb["stmt_recon"][i, 0], nb["stmt_recon"][i, 1])
                           for i, sid in enumerate(nb["stmt_ids"])}}

    ob = torch.load(f"{out_dir}/recon14_olar_{feature}.pt", weights_only=False)
    res["ol"] = {"variant": ob["variant"], "calib": ob["calib"], "layers": {}}
    for L in OL_AR_LAYERS:
        d, true = ob["layers"][L], acts[L]
        row = {k: _block(true, _mean_k(d[k]), y, g) for k in ("span32", "span128", "ol_bullets", "nla_expl")}
        row["ol_bullets_sample0"] = _block(true, d["ol_bullets"][:, 0], y, g)
        row["stmt"] = {sid: statement_shift(true, y, r[0], r[1]) for sid, r in d["stmt"].items()}
        row["probe_true_cv"] = row["span32"]["probe"]["true"]
        res["ol"]["layers"][str(L)] = row
    return res


def table_rows(res: dict) -> list[dict]:
    """Flat rows for the summary table: one per (ruler, layer, input)."""
    rows = []
    n = res["nla"]
    rows.append({"feature": res["feature"], "ruler": "NLA-AR", "layer": n["layer"], "input": "nla_expl",
                 **_flat(n["nla_expl"])})
    for L, d in res["ol"]["layers"].items():
        for k in ("span32", "span128", "ol_bullets", "nla_expl"):
            rows.append({"feature": res["feature"], "ruler": "OL-AR", "layer": int(L), "input": k, **_flat(d[k])})
    return rows


def _flat(b: dict) -> dict:
    return {"fve": round(b["fve"], 3), "ret1": round(b["ret1"], 2), "probe_true": round(b["probe"]["true"], 3),
            "probe_recon": round(b["probe"]["recon"], 3), "probe_err": round(b["probe"]["err"], 3),
            "probe_retrained": round(b["probe_retrained_on_recon"], 3), "gap_ratio": round(b["gap"]["ratio"], 3),
            "gap_cos": round(b["gap"]["cosine"], 3)}
