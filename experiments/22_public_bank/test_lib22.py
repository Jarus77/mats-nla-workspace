import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib22 as L  # noqa: E402


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
    instr = {"readable": 0.92, "span": 0.85, "stmt": 0.05, "used_z": 7.2, "jenr": 0.9}
    flags = {"constrains_next": True, "short_value": True}
    d = L.decide(instr)
    assert d == {"readable": True, "span": True, "stmt": False, "used": True, "jenr": False}
    p = L.predict(d, flags)
    assert p == {"nla_prompted": True, "nla_states": False, "nla_names_complement": False,
                 "ol_prompted": True, "ol_exhibits": True, "ol_states": False, "jlens_prompted": False}
    d2 = L.decide({"readable": 0.95, "span": 0.6, "stmt": 0.8, "used_z": 0.4, "jenr": 2.3})
    p2 = L.predict(d2, {"constrains_next": False, "short_value": True})
    assert p2["nla_states"] is True and p2["ol_prompted"] is False and p2["jlens_prompted"] is True
    assert L.decide({"readable": 0.95, "span": 0.6, "stmt": 0.1, "used_z": float("nan"), "jenr": 1.0})["used"] is False


def test_assemble_builds_one_row_per_feature_with_predictions(tmp_path):
    import json

    import instruments22 as I

    (tmp_path / "instr122_json_depth.json").write_text(json.dumps({"feature": "json_depth", "readable": 0.9, "jenr": 1.1, "used_z": 4.0, "used_shift": 0.3}))
    (tmp_path / "instr222_json_depth.json").write_text(json.dumps({"feature": "json_depth", "span": 0.7, "stmt": 0.1}))
    rows = I.assemble(tmp_path, ["json_depth", "md_heading"])          # no vol: the span probe is unavailable
    assert [r["feature"] for r in rows] == ["json_depth", "md_heading"]
    r = rows[0]
    assert r["instr"]["span"] is None and r["instr"]["span_gap"] == 0.7
    assert r["decide"]["readable"] and not r["decide"]["span"] and not r["decide"]["stmt"] and r["decide"]["used"]
    assert r["predict"]["nla_states"] is False and r["predict"]["ol_prompted"] is False
    assert rows[1]["status"] == "instruments missing" and rows[1]["predict"] is None
    md = I.markdown(rows)
    assert "json_depth" in md and "md_heading" in md


def test_decide_used_is_two_sided():
    base = {"readable": 0.9, "span": 0.6, "stmt": 0.1, "jenr": 1.0}
    assert L.decide({**base, "used_z": -12.2})["used"] is True
    assert L.decide({**base, "used_z": 2.6})["used"] is False


def test_frozen_probe_auroc_is_high_when_recon_keeps_the_separating_direction():
    torch.manual_seed(0)
    n, d = 40, 6
    y = np.array([0, 1] * (n // 2))
    X = torch.randn(n, d) + 3.0 * torch.as_tensor(y)[:, None] * torch.eye(d)[0]
    groups = np.arange(n)
    good = X * 0.3                         # shrunk but same direction
    bad = torch.randn(n, d)                # no class information
    assert L.frozen_probe_auroc(X, good, y, groups) > 0.85
    assert abs(L.frozen_probe_auroc(X, bad, y, groups) - 0.5) < 0.25


def test_score_cells_and_accuracy():
    import score22 as S

    pred = {"nla_prompted": True, "nla_states": True, "nla_names_complement": False, "ol_prompted": False,
            "ol_exhibits": False, "ol_states": False, "jlens_prompted": False}
    obs = {"nla_prompted": 0.97, "nla_states": 0.31, "nla_names_complement": 0.05, "ol_prompted": 0.74,
           "ol_exhibits": 0.0, "ol_states": 0.0, "jlens_prompted": 0.52}
    c = S.score_row(pred, obs)
    assert c["nla_prompted"]["correct"] and c["nla_states"]["correct"] and not c["ol_prompted"]["correct"]
    assert c["ol_prompted"]["observed_high"] is True
    acc = S.accuracy([c, c])
    assert acc["n"] == 14 and acc["correct"] == 12 and 0.5 < acc["lo"] < acc["point"] < acc["hi"] <= 1.0
    conf = S.confusion([c])
    assert conf["ol"]["pred_low_obs_high"] == 1 and conf["nla"]["pred_high_obs_high"] == 2


def test_complement_rate_counts_unmarked_items_where_the_unmarked_candidate_wins():
    import score22 as S

    lo = [2.0, 1.0, -0.5, 0.3]           # lp(marked) - lp(unmarked)
    y = [1, 1, 0, 0]                     # 1 = marked class
    assert S.complement_rate(lo, y) == 0.5   # of the two unmarked items, one has lo < 0


def test_lexical_cue_auroc_detects_a_token_cue_and_not_noise():
    rng = np.random.default_rng(0)
    n = 120
    y = np.array([0, 1] * (n // 2))
    ids_cued = [list(rng.integers(100, 400, 32)) + ([7] if y[i] else [9]) for i in range(n)]
    ids_noise = [list(rng.integers(100, 400, 33)) for i in range(n)]
    groups = np.arange(n)
    assert L.lexical_cue_auroc(ids_cued, y, groups) > 0.9
    assert abs(L.lexical_cue_auroc(ids_noise, y, groups) - 0.5) < 0.15


def test_predict_uses_revised_nla_rules():
    d = {"readable": True, "span": True, "stmt": False, "used": True, "jenr": False}
    p = L.predict(d, {"constrains_next": True, "short_value": True, "both_positive": True})
    assert p["nla_states"] is False and p["nla_names_complement"] is True
    p2 = L.predict({**d, "stmt": True}, {"constrains_next": False, "short_value": True, "both_positive": False})
    assert p2["nla_states"] is True and p2["nla_names_complement"] is False
