import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))  # mats package, as run_local.py does
import lib17 as L  # noqa: E402
import spec17 as S  # noqa: E402


def test_candidates_fill_in_slot_order():
    t = {"id": "ol:mc:fixed", "template": "- column {v}", "slots": ["one", "two"]}
    assert L.candidates(t) == ["- column one", "- column two"]


def test_candidates_returns_word_lists_for_jlens():
    t = {"id": "jlens:mc:fixed", "template": None, "slots": [["first", "one"], ["second", "two"]]}
    assert L.candidates(t) == [["first", "one"], ["second", "two"]]


def test_true_index_table_column_and_drift():
    assert L.true_index("tb_tablecol", {"meta": {"col": 5}}) == 4
    assert L.true_index("tb_tablecol", {"meta": {"col": 1}}) == 0
    with pytest.raises(ValueError):
        L.true_index("tb_tablecol", {"meta": {"col": 2}})
    assert L.true_index("tb_paren", {"value": "inside"}) == 0
    assert L.true_index("tb_paren", {"value": "outside"}) == 1


def test_paren_item_filter():
    assert L.paren_item_ok("He said (as before) that")
    assert not L.paren_item_ok('He said "as before (that')
    assert not L.paren_item_ok("He said [as before (that")


def test_metrics_multi_perfect_and_chance():
    lp = np.full((6, 6), -5.0)
    truth = np.array([0, 0, 0, 3, 4, 5])
    y = np.array([0, 0, 0, 1, 1, 1])
    lp[np.arange(6), truth] = 0.0
    m = L.metrics_multi(lp, truth, y, S.CLASS_SETS["tb_tablecol"], S.ROWS["tb_tablecol"]["values"],
                        np.arange(6), n_boot=50, seed=0)
    assert m["top1"]["point"] == 1.0 and m["top1"]["chance"] == pytest.approx(1 / 6)
    assert m["auroc_lse"]["point"] == 1.0 and m["mrr"]["point"] == 1.0
    assert m["n"] == 6
    flat = np.zeros((6, 6))
    m0 = L.metrics_multi(flat, truth, y, S.CLASS_SETS["tb_tablecol"], S.ROWS["tb_tablecol"]["values"],
                         np.arange(6), n_boot=50, seed=0)
    assert m0["auroc_lse"]["point"] == pytest.approx(0.5)


def test_metrics_multi_keeps_template_axis_out():
    # one template at a time: a [N, C] matrix, never a stacked [T, N, C]
    with pytest.raises(ValueError):
        L.metrics_multi(np.zeros((2, 3, 4)), np.zeros(3, int), np.zeros(3, int), {"a": [0], "b": [1]}, ["a", "b"],
                        np.arange(3), n_boot=5, seed=0)


def test_lse_pair_lo_matches_two_candidate_when_classes_are_singletons():
    lp = np.array([[0.0, -1.0], [-2.0, 0.0]])
    out = L.lse_pair_lo(lp, {"inside": [0], "outside": [1]}, ["inside", "outside"])
    assert np.allclose(out, lp[:, 1] - lp[:, 0])


def test_spec_candidate_counts_match_rows():
    for row, per_reader in S.MC_TEMPLATES.items():
        for reader, temps in per_reader.items():
            for t in temps:
                assert len(t["slots"]) == S.ROWS[row]["n_cands"], (row, reader, t["id"])
