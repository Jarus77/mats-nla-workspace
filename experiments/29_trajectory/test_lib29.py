"""Tests for lib29 (pure helpers of the training-trajectory experiment). Written before lib29.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib29 as L  # noqa: E402


# ---- checkpoint_order / iter_step / discover_checkpoints -----------------------------------------------------------

def test_checkpoint_order_puts_textnla_then_av_base_then_iters_numerically():
    names = ["iter_000400", "av_base", "iter_000100", "textnla", "iter_000300", "iter_001000"]
    assert L.checkpoint_order(names) == ["textnla", "av_base", "iter_000100", "iter_000300", "iter_000400", "iter_001000"]


def test_checkpoint_order_sorts_by_step_not_lexically_and_dedups():
    assert L.checkpoint_order(["iter_000900", "iter_001000", "iter_000900"]) == ["iter_000900", "iter_001000"]


def test_checkpoint_order_rejects_unknown_names():
    with pytest.raises(ValueError):
        L.checkpoint_order(["av_base", "iter_0004x0"])
    with pytest.raises(ValueError):
        L.checkpoint_order(["final"])


def test_iter_step():
    assert L.iter_step("iter_000400") == 400
    assert L.iter_step("av_base") == 0
    assert L.iter_step("textnla") is None


def test_discover_checkpoints_from_adapter_dir_names():
    assert L.discover_checkpoints(["iter_000400", "iter_000300"]) == ["av_base", "iter_000300", "iter_000400"]
    assert L.discover_checkpoints([]) == ["av_base"]
    assert L.discover_checkpoints(["iter_000300", "notes.txt"]) == ["av_base", "iter_000300"]


def test_adapter_path():
    assert L.adapter_path("iter_000300") == "av_rl_adapters/iter_000300"
    assert L.adapter_path("av_base") is None
    with pytest.raises(ValueError):
        L.adapter_path("textnla")


# ---- text stripping ---------------------------------------------------------------------------------------------

def test_strip_analysis_keeps_the_payload_and_tolerates_an_unclosed_tag():
    assert L.strip_analysis("<analysis>\n- a\n- b\n</analysis> trailing") == "- a\n- b"
    assert L.strip_analysis("<analysis>\n- a") == "- a"
    assert L.strip_analysis("no tags at all") == "no tags at all"


def test_strip_expl_tail_is_modal22s_rule():
    assert L.strip_expl_tail("<explanation>\nx\n</explanation></br>garbage") == "<explanation>\nx\n</explanation>"
    assert L.strip_expl_tail("  plain  ") == "plain"


def test_readout_text_dispatches_on_checkpoint():
    assert L.readout_text("<explanation>\nhello\n</explanation>", "iter_000400") == "hello"
    assert L.readout_text("<explanation>\nhello\n</explanation>", "av_base") == "hello"
    assert L.readout_text("<analysis>\nhello\n</analysis>", "textnla") == "hello"
    assert L.readout_text("", "iter_000400") == "(empty)"


# ---- rate_table / merge -------------------------------------------------------------------------------------------

def test_rate_table_columns_follow_checkpoint_order_and_blank_missing_cells():
    vals = {"iter_000400": {"py_string": 0.125, "code_lang": 0.75}, "av_base": {"py_string": 0.0}, "textnla": {"code_lang": 0.5}}
    md = L.rate_table(vals, rows=["py_string", "code_lang"])
    lines = md.splitlines()
    assert lines[0] == "| row | textnla | av_base | iter_000400 |"
    assert lines[2] == "| py_string |  | 0.00 | 0.12 |" or lines[2] == "| py_string |  | 0.00 | 0.13 |"
    assert lines[3] == "| code_lang | 0.50 |  | 0.75 |"


def test_rate_table_respects_an_explicit_checkpoint_list():
    vals = {"iter_000400": {"a": 1.0}, "av_base": {"a": 0.0}}
    md = L.rate_table(vals, rows=["a"], ckpts=["av_base"])
    assert md.splitlines()[0] == "| row | av_base |"


def test_merge_returns_a_new_dict_and_mutates_neither_input():
    a, b = {"x": 1, "y": 1}, {"y": 2}
    before = (json.dumps(a), json.dumps(b))
    m = L.merge(a, b)
    assert m == {"x": 1, "y": 2}
    assert (json.dumps(a), json.dumps(b)) == before
    assert m is not a and m is not b


def test_merge_nested_merges_per_checkpoint():
    a = {"av_base": {"py_string": 0.1}}
    b = {"av_base": {"code_lang": 0.9}, "iter_000400": {"py_string": 0.2}}
    m = L.merge_nested(a, b)
    assert m == {"av_base": {"py_string": 0.1, "code_lang": 0.9}, "iter_000400": {"py_string": 0.2}}
    assert a == {"av_base": {"py_string": 0.1}}


# ---- textbag_auroc ------------------------------------------------------------------------------------------------

def _synthetic_bag(n_per_class: int = 40, seed: int = 0):
    """Token-id lists where class 1 carries a marker token (900) and class 0 does not, among random noise; 10 groups."""
    rng = np.random.default_rng(seed)
    ids, y, groups = [], [], []
    for c in (0, 1):
        for i in range(n_per_class):
            toks = list(rng.integers(100, 120, size=12))
            if c == 1:
                toks.append(900)
            ids.append([int(t) for t in toks])
            y.append(c)
            groups.append(f"g{i % 10}")
    return ids, np.array(y), np.array(groups)


def test_textbag_auroc_separates_a_marker_token_and_is_at_chance_on_shuffled_labels():
    ids, y, groups = _synthetic_bag()
    hi = L.textbag_auroc(ids, y, groups)
    assert hi > 0.9
    rng = np.random.default_rng(1)
    lo = L.textbag_auroc(ids, rng.permutation(y), groups)
    assert abs(lo - 0.5) < 0.2


def test_textbag_auroc_uses_whole_text_not_a_window():
    """The marker sits at position 0 of 60 tokens: a last-48-token window would miss it."""
    rng = np.random.default_rng(2)
    ids, y, groups = [], [], []
    for c in (0, 1):
        for i in range(40):
            toks = [int(t) for t in rng.integers(100, 160, size=60)]
            if c == 1:
                toks[0] = 900
            ids.append(toks)
            y.append(c)
            groups.append(f"g{i % 10}")
    assert L.textbag_auroc(ids, np.array(y), np.array(groups)) > 0.9


def test_word_tokens_is_a_deterministic_fallback_tokeniser():
    a = L.word_tokens("Inside a string literal; inside!")
    assert a == L.word_tokens("Inside a string literal; inside!")
    assert len(a) == 7 and a[0] == a[5]        # "inside" twice, case-folded; ";" and "!" kept as tokens


# ---- judge rows and cost -------------------------------------------------------------------------------------------

def test_judge_rows_flatten_mining_samples_with_values_and_cap_samples():
    gen = {"feature": "py_string", "ckpt": "av_base", "sel": {}, "mine": {"i1": ["a", "b", "c"], "i2": ["d", "e", "f"]}}
    value_of = {"i1": "inside", "i2": "outside"}
    rows = L.judge_rows(gen, value_of, samples=2)
    assert [(r["item"], r["value"], r["k"], r["text"]) for r in rows] == [
        ("i1", "inside", 0, "a"), ("i1", "inside", 1, "b"), ("i2", "outside", 0, "d"), ("i2", "outside", 1, "e")]
    assert len(L.judge_rows(gen, value_of, samples=8)) == 6


def test_judge_rows_drop_items_without_a_value():
    gen = {"mine": {"i1": ["a"], "ghost": ["z"]}}
    assert [r["item"] for r in L.judge_rows(gen, {"i1": "inside"})] == ["i1"]


def test_judge_cost_estimate_scales_with_readouts():
    assert L.judge_cost_estimate(4096) == pytest.approx(1.17, abs=0.01)     # exp 22: $1.17 for 4,096 readouts
    assert L.judge_cost_estimate(0) == 0.0
    assert L.judge_cost_estimate(2560) == pytest.approx(1.17 * 2560 / 4096, rel=1e-6)


def test_samples_for_budget_halves_when_over_cap():
    assert L.samples_for_budget(n_items=320, per_ckpt_cap=15.0) == 8
    assert L.samples_for_budget(n_items=320, per_ckpt_cap=0.5) == 4
    assert L.samples_for_budget(n_items=320, per_ckpt_cap=0.0) == 4


def test_stated_rate_from_judged_rows():
    judged = [{"stated": True}, {"stated": False}, {"stated": True}, {"stated": False}]
    assert L.stated_rate(judged) == 0.5
    assert L.stated_rate([]) is None


# ---- predictions ----------------------------------------------------------------------------------------------------

def _summary():
    ck = ["textnla", "av_base", "iter_000300", "iter_000400"]
    rows = ["py_string", "js_string", "latex_math", "py_arg", "json_depth", "md_heading", "md_list", "code_lang"]
    states = {c: {r: (0.1 if r != "code_lang" else v) for r in rows} for c, v in zip(ck, (0.3, 0.4, 0.5, 0.6))}
    carries = {c: {r: v for r in rows} for c, v in zip(ck, (0.80, 0.82, 0.86, 0.90))}
    textbag = {c: {r: v for r in rows} for c, v in zip(ck, (0.90, 0.91, 0.92, 0.93))}
    return {"states": states, "carries": carries, "textbag": textbag}


def test_check_predictions_all_hold_on_the_designed_summary():
    out = L.check_predictions(_summary(), final="iter_000400")
    assert out == {"p1_structural_states_flat": True, "p2_code_lang_states_rise": True,
                   "p3_carrying_rises_and_textnla_near_base": True, "p4_textbag_stable_on_strings": True}


def test_check_predictions_fail_individually():
    s = _summary()
    s["states"]["av_base"]["py_string"] = 0.4                       # p1 fails
    s["states"]["iter_000400"]["code_lang"] = 0.45                  # rise 0.05 < 0.15: p2 fails
    s["carries"]["textnla"]["py_string"] = 0.50                     # textnla far from base: p3 fails
    s["textbag"]["iter_000400"]["js_string"] = 0.70                 # change 0.21 on a string row: p4 fails
    out = L.check_predictions(s, final="iter_000400")
    assert out == {"p1_structural_states_flat": False, "p2_code_lang_states_rise": False,
                   "p3_carrying_rises_and_textnla_near_base": False, "p4_textbag_stable_on_strings": False}


def test_check_predictions_tolerates_missing_cells():
    s = _summary()
    del s["states"]["textnla"]
    s["carries"] = {}
    out = L.check_predictions(s, final="iter_000400")
    assert out["p1_structural_states_flat"] is True
    assert out["p3_carrying_rises_and_textnla_near_base"] is None


def test_mean_delta():
    vals = {"a": {"r1": 0.5, "r2": 0.7}, "b": {"r1": 0.6, "r2": 0.9}}
    assert L.mean_delta(vals, "a", "b", ["r1", "r2"]) == pytest.approx(0.15)
    assert L.mean_delta(vals, "a", "b", ["r3"]) is None


def test_constants_match_the_design():
    assert L.STRUCTURAL_ROWS == ("py_string", "js_string", "latex_math", "py_arg", "json_depth", "md_heading", "md_list")
    assert L.STRING_ROWS == ("py_string", "js_string", "latex_math")
    assert L.READABLE_ROWS == ("py_string", "js_string", "latex_math", "py_arg", "json_depth")
    assert L.CONTROL_ROW == "code_lang"
    assert L.K_SEL == 1 and L.K_MINE == 8
    assert L.OUT_FILE("gen", "av_base", "py_string") == "gen29_av_base_py_string.json"
