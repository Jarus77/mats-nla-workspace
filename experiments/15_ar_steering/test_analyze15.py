"""Synthetic end-to-end check of analyze15 on fake behave/score blobs shaped like the Modal outputs."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze15 as A  # noqa: E402
from lib15 import conditions  # noqa: E402
from spec15 import DIR_NAMES  # noqa: E402


def _fake(n=40, seed=0):
    rng = np.random.default_rng(seed)
    conds = conditions()
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    spec = {"y": y, "geometry": {"n": n}}
    model, jl, lo = {}, {}, {"nla": {}, "ol": {}}
    for ci, c in enumerate(conds):
        kind, name, q, s = c
        eff = s * (2.0 if (kind == "real" and name.startswith("cm")) else 0.0)
        base = rng.normal(size=n)
        model[ci] = {"metric": torch.tensor(5 + base - eff, dtype=torch.float32), "found": torch.ones(n, dtype=torch.bool),
                     "lo_close": torch.tensor(base + eff, dtype=torch.float32), "lo_nl": torch.tensor(base, dtype=torch.float32),
                     "chars_to_close": torch.tensor(10 + base, dtype=torch.float32), "texts": ["x"] * n}
        jl[ci] = torch.tensor(base * 0.1, dtype=torch.float32)
        for r in lo:
            lo[r][ci] = torch.tensor(np.stack([base + eff, base + 0.5 * eff]), dtype=torch.float32)
    beh = {"conds": conds, "model": model, "jlens": jl, "metric": "cells_to_newline",
           "clean_nohook": {"metric": model[0]["metric"].clone()}}
    sc = {r: {"conds": conds, "pairs": [["fixed", "a", "b"], ["t1", "c", "d"]], "lo": lo[r]} for r in lo}
    return spec, beh, sc


def test_analyze_feature_detects_planted_effect_only_on_cm_directions():
    spec, beh, sc = _fake()
    res = A.analyze_feature("tb_tablecol", spec, beh, sc)
    assert set(res["cells"]) == {f"{d}:{q}" for d in DIR_NAMES for q in ("r11", "q90")}
    cm = res["cells"]["cm_jrem4:r11"]
    assert cm["model_metric"]["nonj4"]["real"]["mean_shift"] < -3
    assert cm["model_metric"]["nonj4"]["band"]["z"] < -5
    assert cm["nla_mean"]["iso"]["band"]["z"] > 5
    ar = res["cells"]["arstmt_raw:r11"]
    assert abs(ar["model_metric"]["nonj4"]["band"]["z"]) < 3
    assert "real" in res["cells"]["cm_raw:q90"]["nla_fixed"] and "band" not in res["cells"]["cm_raw:q90"]["nla_fixed"]
    assert res["sanity"]["zero_vs_nohook_metric_mean_absdiff"] == 0.0
    rows = A.table_rows(res)
    assert len(rows) == len(DIR_NAMES) and rows[0]["direction"] == "cm_raw"
    assert all(k in rows[0] for k in ("model_metric", "nla_mean", "ol_mean", "jlens"))
