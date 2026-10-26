import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import ci_prior as C  # noqa: E402

DIRS = ("cm_raw", "cm_jrem4", "arexpl_raw", "arexpl_jrem4", "arspan_raw", "arspan_jrem4", "arstmt_raw", "arstmt_jrem4")


def _exp15_files(tmp_path: Path, row: str = "tb_tablecol", n: int = 8, k: int = 3, nan_null: bool = True) -> None:
    conds = [("zero", None, "r11", 0)]
    for d in DIRS:
        for q in ("r11", "q90"):
            conds += [("real", d, q, 1), ("real", d, q, -1)]
    for i in range(k):
        for cls in ("iso", "nonj4"):
            conds += [("null", f"{cls}_{i}", "r11", 1), ("null", f"{cls}_{i}", "r11", -1)]
    rng = np.random.default_rng(0)
    ids = [f"tbl_{i // 2}_c{i % 2}" for i in range(n)]
    torch.save({"ids": ids, "y": np.array([i % 2 for i in range(n)])}, tmp_path / f"spec15_{row}.pt")
    model = {}
    for ci, c in enumerate(conds):
        shift = 2.0 if (c[0] == "real" and c[3] == 1) else 0.0
        model[ci] = {"lo_close": torch.zeros(n), "lo_nl": torch.tensor(rng.normal(shift, 0.1, n)),
                     "metric": torch.zeros(n), "found": torch.ones(n, dtype=torch.bool),
                     "chars_to_close": torch.zeros(n), "texts": [""] * n}
    jl = {ci: torch.zeros(n) for ci in range(len(conds))}
    half = len(conds) // 2
    for ch, rng_ in ((0, range(0, half)), (1, range(half, len(conds)))):
        torch.save({"conds": conds, "chunk": ch, "n_chunks": 2, "model": {ci: model[ci] for ci in rng_},
                    "metric": "cells_to_newline", "clean_nohook": {"lo_nl": torch.zeros(n), "metric": torch.zeros(n)},
                    "jlens": jl, "wall_s": 1.0}, tmp_path / f"behave15_{row}_c{ch}.pt")
    for r in ("nla", "ol"):
        lo = {}
        for ci, c in enumerate(conds):
            if nan_null and c[0] == "null" and c[1] == "iso_2":
                lo[ci] = torch.full((4, n), float("nan"))
            else:
                shift = 3.0 if (c[0] == "real" and c[3] == 1) else 0.0
                lo[ci] = torch.tensor(rng.normal(shift, 0.1, (4, n)))
        torch.save({"conds": conds, "pairs": [["p0", "a", "b"]] * 4, "lo": lo}, tmp_path / f"score15_{r}_{row}.pt")


def test_ci15_reports_shift_ci_and_z_and_drops_nan_null_rows(tmp_path):
    _exp15_files(tmp_path)
    out = C.ci15("tb_tablecol", tmp_path, n_boot=50, seed=0)
    cell = out["cells"]["cm_raw:r11"]["nla_mean"]
    assert cell["lo"] <= cell["mean_shift"] <= cell["hi"]
    assert np.isfinite(cell["z_iso"]) and cell["k_iso"] == 2  # the NaN null was dropped
    assert out["cells"]["cm_raw:r11"]["model_lo_newline"]["mean_shift"] > 1.5


def test_ci16_follows_j_ci_contains_point(tmp_path):
    n = 8
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    j = np.array([4, 5, 6, 7, 0, 1, 2, 3])
    clusters = [f"c{i // 2}" for i in range(n)]
    lo_partner = torch.tensor(np.stack([y[j].astype(float)] * 4))  # readout tracks the partner's label
    torch.save({"ids": [str(i) for i in range(n)], "y": y, "j": j, "clusters": clusters,
                "nla_lo": {"swap_res": lo_partner, "h": torch.tensor(np.stack([y.astype(float)] * 4))},
                "pairs": [["p", "a", "b"]] * 4}, tmp_path / "s1_tb_tablecol.pt")
    torch.save({"ids": [str(i) for i in range(n)], "ol_lo": {"swap_res": lo_partner, "h": lo_partner},
                "pairs": [["p", "a", "b"]] * 4}, tmp_path / "s2_tb_tablecol.pt")
    out = C.ci16("tb_tablecol", tmp_path, n_boot=50, seed=0)
    r = out["readers"]["nla"]["swap_res"]["mean"]
    assert r["point"] == 1.0 and r["lo"] == 1.0 and r["hi"] == 1.0  # a perfect readout is perfect in every resample
    assert out["readers"]["nla"]["h"]["mean"]["point"] == 0.0
    assert out["readers"]["ol"]["swap_res"]["fixed"]["point"] == 1.0
