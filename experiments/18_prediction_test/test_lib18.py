import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib18 as L  # noqa: E402


def test_fill_uses_fillers():
    t = {"template": "The text is written in {v}", "fillers": {"French": "French", "German": "German"}}
    assert L.fill(t, "German") == "The text is written in German"


def test_set_logodds_is_logit_of_set_mass():
    lp = torch.log(torch.tensor([[0.25, 0.25, 0.5]]))
    lo = L.set_logodds(lp, [0, 1])          # mass 0.5 -> log-odds 0
    assert torch.allclose(lo, torch.zeros(1), atol=1e-5)


def test_centred_push_has_equal_norm_both_sides_and_removes_own_projection():
    torch.manual_seed(0)
    h = torch.randn(5, 8)
    v = L.unit(torch.randn(8))
    p, m = L.centred_push(h, v, 0.11, +1), L.centred_push(h, v, 0.11, -1)
    assert torch.allclose(p.norm(dim=1), m.norm(dim=1), atol=1e-5)
    assert torch.allclose(p @ v, 0.11 * h.norm(dim=1), atol=1e-5)
    assert torch.allclose(m @ v, -0.11 * h.norm(dim=1), atol=1e-5)


def test_iso_nulls_are_unit_and_reproducible():
    a, b = L.iso_nulls(8, 3, seed=1), L.iso_nulls(8, 3, seed=1)
    assert torch.allclose(a, b) and torch.allclose(a.norm(dim=1), torch.ones(3))


def test_band_z():
    b = L.band(5.0, [0.0, 1.0, -1.0, 0.5, -0.5])
    assert b["z"] > 5 and b["k"] == 5


def test_decide_and_predict_follow_the_prereg_rules():
    instr = {"readable": 0.92, "span": 0.61, "stmt": 0.05, "used_z": 7.2, "jenr": 0.9}
    flags = {"constrains_next": True, "short_value": True}
    d = L.decide(instr)
    assert d == {"readable": True, "span": True, "stmt": False, "used": True, "jenr": False}
    p = L.predict(d, flags)
    assert p == {"nla_prompted": True, "nla_states": True, "nla_names_complement": False,
                 "ol_prompted": True, "ol_exhibits": True, "ol_states": False, "jlens_prompted": False}
    d2 = L.decide({"readable": 0.95, "span": 0.2, "stmt": 0.8, "used_z": 0.4, "jenr": 2.3})
    p2 = L.predict(d2, {"constrains_next": False, "short_value": True})
    assert p2["nla_states"] is False and p2["ol_prompted"] is False and p2["jlens_prompted"] is True
    assert L.decide({"readable": 0.95, "span": 0.6, "stmt": 0.1, "used_z": float("nan"), "jenr": 1.0})["used"] is False


def test_assemble_builds_one_row_per_feature_with_predictions(tmp_path):
    import json

    import instruments18 as I

    (tmp_path / "instr118_json_depth.json").write_text(json.dumps({"feature": "json_depth", "readable": 0.9, "jenr": 1.1, "used_z": 4.0, "used_shift": 0.3}))
    (tmp_path / "instr218_json_depth.json").write_text(json.dumps({"feature": "json_depth", "span": 0.7, "stmt": 0.1}))
    rows = I.assemble(tmp_path, ["json_depth", "md_heading"])
    assert [r["feature"] for r in rows] == ["json_depth", "md_heading"]
    r = rows[0]
    assert r["decide"]["readable"] and r["decide"]["span"] and not r["decide"]["stmt"] and r["decide"]["used"]
    assert r["predict"]["nla_states"] is True and r["predict"]["ol_prompted"] is True
    assert rows[1]["status"] == "instruments missing" and rows[1]["predict"] is None
    md = I.markdown(rows)
    assert "json_depth" in md and "md_heading" in md
