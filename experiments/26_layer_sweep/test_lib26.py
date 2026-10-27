"""Tests for lib26 (pure helpers of the layer sweep). Written before lib26.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib26 as L  # noqa: E402


# ---- sweep_layers ------------------------------------------------------------------------------------------------

def test_sweep_layers_is_twelve_ascending_ints_with_42_in_place():
    ls = L.sweep_layers()
    assert len(ls) == 12 and all(isinstance(x, int) for x in ls)
    assert ls == sorted(ls) and len(set(ls)) == 12
    assert 42 in ls and ls[0] == 20 and ls[-1] == 60
    assert ls.index(42) == ls.index(40) + 1


def test_sweep_layers_matches_the_olens_module():
    from mats.readers.olens import SWEEP_LAYERS

    assert tuple(L.sweep_layers()) == tuple(SWEEP_LAYERS)


# ---- names_rate / NAME_RE ----------------------------------------------------------------------------------------

def test_name_re_covers_every_row():
    assert set(L.NAME_RE) == set(L.ROWS)


@pytest.mark.parametrize("row, stated", [
    ("py_string", "- inside a string literal\n- Python code"),
    ("js_string", "- the cursor is within an unclosed string"),
    ("json_depth", "- nested inside a curly brace block"),
    ("latex_math", "- currently in math mode in LaTeX"),
    ("md_heading", "- this is a markdown heading"),
    ("md_list", "- a nested list item in markdown"),
    ("py_arg", "- the second argument of a call"),
    ("code_lang", "- Python source code"),
])
def test_names_rate_fires_on_a_stated_example(row, stated):
    assert L.names_state(stated, row) is True
    assert L.names_rate([stated, "- nothing structural here"], row) == pytest.approx(0.5)


@pytest.mark.parametrize("row", ["py_string", "js_string", "json_depth", "latex_math", "md_heading", "md_list", "py_arg"])
def test_names_rate_does_not_fire_on_a_content_word(row):
    assert L.names_state("- open", row) is False
    assert L.names_state("- the file is open for reading, level 2", row) is False
    assert L.names_rate(["- open", "- level", None, ""], row) == 0.0


def test_names_rate_on_empty_input_is_nan():
    assert np.isnan(L.names_rate([], "py_string"))


def test_the_exp23_rows_use_exp23s_strict_regexes():
    from lib23 import STATE_RE

    for row, state in L.STATE_OF.items():
        assert L.NAME_RE[row].pattern == STATE_RE[state].pattern


# ---- select_items ------------------------------------------------------------------------------------------------

def test_select_items_keeps_order_and_marks_the_first_value():
    items = [{"feature": "f", "id": "a", "split": "mine", "value": "in", "cluster": "c0"},
             {"feature": "f", "id": "b", "split": "select", "value": "out", "cluster": "c1"},
             {"feature": "f", "id": "c", "split": "select", "value": "in", "cluster": "c2"}]
    sel, y, groups = L.select_items(items, marked="in")
    assert [it["id"] for it in sel] == ["b", "c"] and y.tolist() == [0, 1] and groups.tolist() == ["c1", "c2"]
    assert items[1] is sel[0]                                   # no copies, no mutation


# ---- prompted scoring (exp 22's convention) ----------------------------------------------------------------------

def test_prompted_candidates_interleave_marked_and_unmarked_per_template():
    temps = [{"id": "t0", "template": "- {v} string", "fillers": {"inside": "inside", "outside": "outside"}},
             {"id": "t1", "template": "- {v}", "fillers": {"inside": '""', "outside": "()"}}]
    assert L.prompted_candidates(temps, ["inside", "outside"]) == ["- inside string", "- outside string", '- ""', "- ()"]


def test_prompted_logodds_splits_columns_by_template():
    import torch

    lp = torch.tensor([[0.0, -1.0, 2.0, 0.5], [1.0, 1.0, -3.0, 0.0]])    # [N=2, T*2=4]
    lo = L.prompted_logodds(lp)                                           # [T, N]
    assert lo.shape == (2, 2)
    assert lo.tolist() == [[1.0, 0.0], [1.5, -3.0]]


def test_prompted_auroc_is_max_of_mean_of_selected_and_fixed():
    y = [1, 1, 0, 0]
    lo = {"m0": [3.0, 2.0, 1.0, 0.0], "m1": [0.0, 1.0, 2.0, 3.0], "m2": [1.0, 0.0, 2.0, 3.0], "fixed": [0.0, 0.0, 1.0, 1.0]}
    out = L.prompted_auroc(lo, ["m0", "m1", "m2"], "fixed", y)
    # mean of selected = [4/3, 1, 5/3, 2]: both negatives above both positives except... AUROC 0.0; fixed: 0.0
    assert out["mean_of_selected"] == pytest.approx(0.0) and out["fixed_label"] == pytest.approx(0.0)
    assert out["auroc"] == pytest.approx(0.0)
    lo2 = {**lo, "fixed": [1.0, 1.0, 0.0, 0.0]}
    out2 = L.prompted_auroc(lo2, ["m0", "m1", "m2"], "fixed", y)
    assert out2["fixed_label"] == pytest.approx(1.0) and out2["auroc"] == pytest.approx(1.0)


def test_ol_templates_for_row_reads_three_selected_plus_fixed():
    frozen = {"templates": {"ol": {"r": {
        "selected": [{"id": "ol:r:m0", "template": "- {v}", "fillers": {"a": "x", "b": "y"}, "selection": {}},
                     {"id": "ol:r:m1", "template": "- {v}!", "fillers": {"a": "x", "b": "y"}},
                     {"id": "ol:r:m2", "template": "- {v}?", "fillers": {"a": "x", "b": "y"}}],
        "fixed_label": {"id": "ol:r:fixed", "template": "- {v} thing", "fillers": {"a": "a", "b": "b"}, "candidates": {}},
        "mean_of_selected": {}}}}}
    temps, selected_ids, fixed_id = L.ol_templates(frozen, "r")
    assert [t["id"] for t in temps] == ["ol:r:m0", "ol:r:m1", "ol:r:m2", "ol:r:fixed"]
    assert selected_ids == ["ol:r:m0", "ol:r:m1", "ol:r:m2"] and fixed_id == "ol:r:fixed"
    assert all(set(t) == {"id", "template", "fillers"} for t in temps)


# ---- merge ------------------------------------------------------------------------------------------------------

def test_merge_keeps_stage1_keys_adds_stage2_keys_and_does_not_mutate():
    a = {"feature": "r", "n": 2, "layers": [20, 42], "readable": {"20": 0.5, "42": 0.9}, "ol_prompted": {"20": 0.5, "42": 0.8}}
    b = {"carries": {"20": 0.5, "42": 0.7}, "names": {"20": 0.0, "42": 0.01}}
    a0, b0 = dict(a), dict(b)
    m = L.merge(a, b)
    assert a == a0 and b == b0 and m is not a
    assert set(m) == {"feature", "n", "layers", "readable", "ol_prompted", "carries", "names"}
    assert m["readable"] == a["readable"] and m["carries"] == b["carries"]


def test_merge_second_wins_on_a_shared_key():
    assert L.merge({"x": 1, "y": 2}, {"y": 3})["y"] == 3


# ---- table --------------------------------------------------------------------------------------------------------

def test_table_lists_layers_ascending_with_42_in_place_and_two_decimals():
    rows = [{"feature": "py_string", "layers": [60, 20, 42, 40, 44], "readable": {"60": 0.9, "20": 0.5, "42": 0.912, "40": 0.8, "44": 0.85}},
            {"feature": "md_list", "layers": [20, 40, 42, 44, 60], "readable": {20: 0.5, 40: 0.6, 42: 0.55, 44: 0.5, 60: 0.6}}]
    md = L.table(rows, "readable")
    lines = md.splitlines()
    assert lines[0] == "| row | 20 | 40 | 42 | 44 | 60 |"
    assert lines[1].startswith("|---|")
    assert lines[2] == "| py_string | 0.50 | 0.80 | 0.91 | 0.85 | 0.90 |"
    assert lines[3] == "| md_list | 0.50 | 0.60 | 0.55 | 0.50 | 0.60 |"


def test_table_leaves_a_missing_cell_blank():
    rows = [{"feature": "r", "layers": [20, 24], "names": {"20": 0.0}}]
    assert L.table(rows, "names").splitlines()[2] == "| r | 0.00 |  |"
