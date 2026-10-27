import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXP22 = HERE.parent / "22_public_bank"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib25 as L  # noqa: E402


def _row(feature: str, cells: dict, observed: dict | None = None) -> dict:
    return {"feature": feature, "observed": observed or {c: v["observed"] for c, v in cells.items()}, "cells": cells}


def _cell(pred: bool, obs: float, thresh: float) -> dict:
    high = obs >= thresh
    return {"predicted_high": pred, "observed": obs, "observed_high": high, "correct": high == pred}


# ---- swap_exhibits ---------------------------------------------------------------------------------------------

def test_swap_replaces_only_the_exhibits_cell_and_does_not_mutate_the_input():
    cells = {"ol_exhibits": _cell(True, 0.0, 0.30), "ol_states": _cell(False, 0.0, 0.05), "nla_prompted": _cell(True, 0.9, 0.85)}
    rows = [_row("py_string", cells)]
    before = json.dumps(rows, sort_keys=True)
    new = L.swap_exhibits(rows, {"py_string": {"auroc": {"ol_readout": 0.91}}}, line=0.70)
    assert json.dumps(rows, sort_keys=True) == before                      # input untouched
    ex = new[0]["cells"]["ol_exhibits"]
    assert ex == {"predicted_high": True, "observed": 0.91, "observed_high": True, "correct": True}
    assert new[0]["observed"]["ol_exhibits"] == 0.91
    for c in ("ol_states", "nla_prompted"):
        assert new[0]["cells"][c] == rows[0]["cells"][c]
        assert new[0]["observed"][c] == rows[0]["observed"][c]
    assert new[0]["feature"] == "py_string"


def test_swap_uses_the_line_and_the_row_prediction():
    rows = [_row("md_list", {"ol_exhibits": _cell(False, 0.25, 0.30)}),
            _row("py_arg", {"ol_exhibits": _cell(True, 0.03, 0.30)})]
    recon = {"md_list": {"auroc": {"ol_readout": 0.62}}, "py_arg": {"auroc": {"ol_readout": 0.70}}}
    new = L.swap_exhibits(rows, recon, line=0.70)
    assert new[0]["cells"]["ol_exhibits"]["observed_high"] is False and new[0]["cells"]["ol_exhibits"]["correct"] is True
    assert new[1]["cells"]["ol_exhibits"]["observed_high"] is True and new[1]["cells"]["ol_exhibits"]["correct"] is True
    assert L.swap_exhibits(rows, recon, line=0.75)[1]["cells"]["ol_exhibits"]["correct"] is False


def test_swap_fails_loudly_when_a_row_has_no_reconstruction():
    rows = [_row("py_string", {"ol_exhibits": _cell(True, 0.0, 0.30)})]
    with pytest.raises(KeyError, match="py_string"):
        L.swap_exhibits(rows, {}, line=0.70)


# ---- accuracy_split ---------------------------------------------------------------------------------------------

def test_accuracy_split_reproduces_a_hand_computed_case():
    rows = [
        _row("a", {"p": _cell(True, 0.9, 0.5), "s": _cell(True, 0.1, 0.5), "j": _cell(False, 0.1, 0.5)}),   # p ok, s MISS, j ok
        _row("b", {"p": _cell(False, 0.9, 0.5), "s": _cell(False, 0.1, 0.5), "j": _cell(True, 0.9, 0.5)}),  # p MISS, s ok, j ok (excluded)
    ]
    out = L.accuracy_split(rows, primary_cells=("p", "j"), secondary_cells=("s",), exclude={"j": {"b"}})
    assert (out["primary"]["correct"], out["primary"]["n"]) == (2, 3)
    assert (out["secondary"]["correct"], out["secondary"]["n"]) == (1, 2)
    assert (out["all"]["correct"], out["all"]["n"]) == (4, 6)                 # "all" scores every cell, no exclusion
    assert out["primary"]["point"] == pytest.approx(2 / 3)
    assert out["primary"]["lo"] < 2 / 3 < out["primary"]["hi"]
    assert 0.0 <= out["primary"]["lo"] and out["primary"]["hi"] <= 1.0


def test_accuracy_split_skips_unscored_cells():
    rows = [_row("a", {"p": {"predicted_high": True, "observed": None, "observed_high": None, "correct": None},
                       "s": _cell(True, 0.9, 0.5)})]
    out = L.accuracy_split(rows, primary_cells=("p",), secondary_cells=("s",), exclude={})
    assert out["primary"]["n"] == 0 and out["secondary"]["n"] == 1 and out["all"]["n"] == 1


# ---- the exp 22 numbers, from its own score22.json ------------------------------------------------------------

def test_score22_json_reproduces_the_reported_exp22_scorecard():
    rows = json.loads((EXP22 / "results" / "score22.json").read_text())["rows"]
    out = L.accuracy_split(rows, L.PRIMARY_CELLS, L.SECONDARY_CELLS, L.EXCLUDE)
    assert (out["primary"]["correct"], out["primary"]["n"]) == (30, 38)
    assert (out["secondary"]["correct"], out["secondary"]["n"]) == (13, 16)
    assert (out["all"]["correct"], out["all"]["n"]) == (44, 56)
    no_ex = L.accuracy_split(rows, tuple(c for c in L.PRIMARY_CELLS if c != "ol_exhibits"), L.SECONDARY_CELLS, L.EXCLUDE)
    assert (no_ex["primary"]["correct"], no_ex["primary"]["n"]) == (26, 30)
    no_cl = L.accuracy_split([r for r in rows if r["feature"] != "code_lang"], L.PRIMARY_CELLS, L.SECONDARY_CELLS, L.EXCLUDE)
    assert (no_cl["primary"]["correct"], no_cl["primary"]["n"]) == (27, 34)
    assert (no_cl["secondary"]["correct"], no_cl["secondary"]["n"]) == (11, 14)


# ---- the selection split and the text helpers used on the GPU ---------------------------------------------------

def test_select_split_marks_the_first_value_and_keeps_item_order():
    items = [{"feature": "f", "split": "mine", "value": "in", "cluster": "c0"},
             {"feature": "f", "split": "select", "value": "out", "cluster": "c1"},
             {"feature": "f", "split": "select", "value": "in", "cluster": "c2"}]
    idx, y, groups = L.select_split(items, marked="in")
    assert idx == [1, 2] and y.tolist() == [0, 1] and groups.tolist() == ["c1", "c2"]


def test_shuffle_keeps_the_multiset_and_changes_the_order():
    ids = list(range(32))
    s = L.shuffle_tokens(ids, random.Random(25))
    assert sorted(s) == ids and s != ids


def test_explanation_text_falls_back_to_raw():
    assert L.explanation_text("<explanation>Inside a string literal</explanation>") == "Inside a string literal"
    assert L.explanation_text("no tags here at all") == "no tags here at all"
    assert L.explanation_text("") == "(empty)"


def test_per_row_lines_pair_judge_and_reconstructor_values():
    old = [_row("py_string", {"ol_exhibits": _cell(True, 0.0, 0.30)})]
    new = L.swap_exhibits(old, {"py_string": {"auroc": {"ol_readout": 0.91}}}, line=0.70)
    lines = L.per_row_lines(old, new)
    assert lines == ["py_string: judge 0.00 -> recon 0.91 (predicted high, ok)"]
