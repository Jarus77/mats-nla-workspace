"""Tests for lib30 (pure helpers of exp 30, no model). Written before lib30.py (RED first)."""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib30 as L  # noqa: E402
from mats.readers import nla as NL  # noqa: E402


# ---- append_statement / training_target -------------------------------------------------------------------------

def test_append_statement_appends_as_a_final_sentence_when_there_is_no_closing_tag():
    out = L.append_statement("First sentence.\n\nSecond sentence.", "The current position is inside a string literal")
    assert out == "First sentence.\n\nSecond sentence.\n\nThe current position is inside a string literal."


def test_append_statement_inserts_before_the_closing_tag_when_present():
    raw = "<explanation>\nBody text.\n</explanation>"
    out = L.append_statement(raw, "The text is Python code")
    assert out == "<explanation>\nBody text.\n\nThe text is Python code.\n</explanation>"
    assert out.count("</explanation>") == 1


def test_append_statement_does_not_double_the_period_and_strips_the_explanation():
    assert L.append_statement("Body.  ", "A statement.") == "Body.\n\nA statement."


def test_training_target_has_the_expected_format_for_both_arms():
    expl, stmt = "Python docstring register, parameters listed.", "The current position is inside a string literal"
    assert L.training_target(expl, stmt) == expl + "\n\n" + stmt + ".\n</explanation>"
    assert L.training_target(expl, None) == expl + "\n</explanation>"
    assert L.training_target(expl, stmt) == L.append_statement(expl, stmt) + L.CLOSE


def test_training_target_refuses_an_empty_explanation():
    with pytest.raises(ValueError):
        L.training_target("(empty)", "x")
    with pytest.raises(ValueError):
        L.training_target("   ", None)


# ---- grouped_split -----------------------------------------------------------------------------------------------

def _items(n_clusters: int, per: int) -> list[dict]:
    return [{"id": f"i{c}_{k}", "cluster": f"c{c}", "value": "a" if k % 2 else "b"} for c in range(n_clusters) for k in range(per)]


def test_grouped_split_never_lets_a_cluster_straddle_and_covers_every_item():
    items = _items(25, 4)
    train, ev = L.grouped_split(items, 0.8, seed=30)
    assert len(train) + len(ev) == len(items)
    assert {it["id"] for it in train}.isdisjoint({it["id"] for it in ev})
    assert {it["cluster"] for it in train}.isdisjoint({it["cluster"] for it in ev})
    assert 0.7 <= len(train) / len(items) <= 0.9
    assert len(ev) > 0


def test_grouped_split_is_deterministic_in_the_seed_and_does_not_mutate_its_input():
    items = _items(40, 2)
    before = json.dumps(items)
    a = L.grouped_split(items, 0.8, seed=30)
    b = L.grouped_split(items, 0.8, seed=30)
    c = L.grouped_split(items, 0.8, seed=31)
    assert json.dumps(items) == before
    assert [i["id"] for i in a[0]] == [i["id"] for i in b[0]]
    assert [i["id"] for i in a[0]] != [i["id"] for i in c[0]]


def test_grouped_split_with_singleton_clusters_gives_the_requested_fraction():
    items = _items(100, 1)
    train, ev = L.grouped_split(items, 0.8, seed=0)
    assert len(train) == 80 and len(ev) == 20


# ---- statement_accuracy ------------------------------------------------------------------------------------------

PAIR = {"inside": "The current position is inside a string literal",
        "outside": "The current position is outside a string literal"}


def test_statement_accuracy_counts_correct_wrong_and_none():
    items = [{"id": "a", "value": "inside"}, {"id": "b", "value": "outside"}, {"id": "c", "value": "inside"},
             {"id": "d", "value": "outside"}]
    readouts = {"a": "Docstring.\n\nThe current position is inside a string literal.",   # correct
                "b": "Docstring. The current position is inside a string literal.",      # wrong (complement)
                "c": "Plain prose with no statement at all.",                             # none
                "d": "the current position is OUTSIDE a string literal"}                  # correct, case-insensitive
    r = L.statement_accuracy(readouts, items, PAIR)
    assert (r["n"], r["correct"], r["wrong"], r["none"]) == (4, 2, 1, 1)
    assert r["accuracy"] == pytest.approx(0.5) and r["naming_rate"] == pytest.approx(0.75)
    assert r["per_item"] == {"a": "correct", "b": "wrong", "c": "none", "d": "correct"}


def test_statement_accuracy_treats_both_statements_present_as_wrong_and_missing_readouts_as_none():
    items = [{"id": "a", "value": "inside"}, {"id": "b", "value": "inside"}]
    readouts = {"a": PAIR["inside"] + ". " + PAIR["outside"] + "."}
    r = L.statement_accuracy(readouts, items, PAIR)
    assert r["per_item"] == {"a": "wrong", "b": "none"}
    assert r["accuracy"] == 0.0


def test_statement_accuracy_ignores_whitespace_differences():
    items = [{"id": "a", "value": "deep"}]
    pair = {"deep": "The current value sits three levels deep in the JSON object", "shallow": "The current value sits at the top level in the JSON object"}
    r = L.statement_accuracy({"a": "The current value sits  three levels\ndeep in the JSON object."}, items, pair)
    assert r["per_item"]["a"] == "correct"


# ---- fve ---------------------------------------------------------------------------------------------------------

def test_fve_reproduces_the_nla_definition_on_a_toy():
    g = torch.Generator().manual_seed(0)
    gold = torch.randn(6, 16, generator=g) * 3
    pred = gold + 0.5 * torch.randn(6, 16, generator=g)
    per_row, base = L.fve(pred, gold)
    ref = NL.fve(pred, gold, NL.predict_mean_baseline(gold))
    assert torch.allclose(per_row, ref, atol=1e-6)
    assert base == pytest.approx(NL.predict_mean_baseline(gold))
    assert torch.allclose(L.fve(gold, gold)[0], torch.ones(6), atol=1e-6)       # perfect rebuild
    noise = torch.randn(6, 16, generator=g)
    assert float(L.fve(noise, gold)[0].mean()) < 0.0            # an unrelated rebuild scores below zero


def test_fve_accepts_a_fixed_baseline_so_arms_are_comparable():
    gold = torch.randn(4, 8)
    per_row, base = L.fve(gold, gold, baseline=2.0)
    assert base == 2.0 and torch.allclose(per_row, torch.ones(4))


# ---- paths / rows / p5 -------------------------------------------------------------------------------------------

def test_trained_adapter_path_is_absolute_so_load_nla_root_join_ignores_the_snapshot_root():
    p = L.trained_adapter_path("/results/30_naming_lora", "S")
    assert p == "/results/30_naming_lora/adapter_S/iter_000400"
    assert str(Path("/weights/snap") / p) == p                      # the pathlib property load_nla relies on
    assert Path(p).name == "iter_000400"                            # the adapter name load_nla derives


def test_row_sets_are_the_six_training_rows_and_the_two_transfer_rows():
    assert set(L.TRAIN_ROWS) == {"py_string", "py_arg", "json_depth", "md_heading", "md_list", "code_lang"}
    assert L.TRANSFER_ROWS == ("js_string", "latex_math")
    assert set(L.TRAIN_ROWS).isdisjoint(L.TRANSFER_ROWS) and len(L.TRAIN_ROWS) + len(L.TRANSFER_ROWS) == 8


def test_within_item_fve_diff_is_naming_minus_non_naming_averaged_over_items_with_both():
    samples = {"a": [(True, 0.8), (False, 0.7), (False, 0.6)], "b": [(True, 0.5), (True, 0.5), (False, 0.6)],
               "c": [(False, 0.9), (False, 0.8)]}                   # c has no naming sample: excluded
    r = L.within_item_fve_diff(samples)
    assert r["n_items"] == 2 and r["mean_diff"] == pytest.approx(((0.8 - 0.65) + (0.5 - 0.6)) / 2)
    assert L.within_item_fve_diff({"c": [(False, 0.9)]}) == {"n_items": 0, "mean_diff": None}


def test_pass_line_helpers():
    assert L.within(0.76, 0.75, 0.02) and not L.within(0.80, 0.75, 0.02)
    assert L.within(float("nan"), 0.75, 0.02) is False


# ---- data30.build_data -------------------------------------------------------------------------------------------

def _toy_bank() -> tuple[list[dict], dict, dict]:
    sys.path.append(str(HERE.parent / "22_public_bank"))
    from feature_spec22 import FEATURES, FIXED

    rng = random.Random(0)
    items, expl = [], {}
    for row, spec in FEATURES.items():
        expl[row] = {}
        for c in range(10):
            for k, v in enumerate(spec["values"]):
                iid = f"{row}_{c}_{k}"
                items.append({"id": iid, "feature": row, "cluster": f"{row}_c{c}", "value": v, "split": "select" if c else "mine"})
                expl[row][iid] = "(empty)" if (row == "py_string" and c == 1 and k == 0) else f"Explanation {rng.random():.3f}."
    return items, expl, FIXED["nla"]


def test_build_data_splits_by_cluster_targets_the_true_value_and_skips_empty_explanations():
    import data30 as D

    items, expl, fixed = _toy_bank()
    d = D.build_data(items, expl, fixed, seed=30, frac=0.8)
    assert set(d["rows"]) == set(L.ALL_ROWS)
    for row, r in d["rows"].items():
        assert set(r["train_ids"]).isdisjoint(r["eval_ids"]) and r["n_eval_items"] > 0
        by_id = {it["id"]: it for it in items}
        assert {by_id[i]["cluster"] for i in r["train_ids"]}.isdisjoint({by_id[i]["cluster"] for i in r["eval_ids"]})
        assert all(by_id[i]["split"] == "select" for i in r["train_ids"] + r["eval_ids"])
    assert {t["row"] for t in d["train"]} == set(L.TRAIN_ROWS)
    assert {e["row"] for e in d["eval"]} == set(L.ALL_ROWS)
    assert d["skipped_empty"] == ["py_string_1_0"] or d["skipped_empty"] == []    # cluster 1 may land in eval
    for t in d["train"]:
        assert t["statement"] == fixed[t["row"]]["template"].replace("{v}", fixed[t["row"]]["fillers"][t["value"]])
        assert t["target_S"] == t["explanation"] + "\n\n" + t["statement"] + ".\n</explanation>"
        assert t["target_C"] == t["explanation"] + "\n</explanation>"
    assert d["gate_ids"] and all(e["row"] in L.TRAIN_ROWS for e in d["eval"] if e["id"] in set(d["gate_ids"]))
    assert d["n_train"] == len(d["train"]) and d["n_eval"] == len(d["eval"])


def test_freeze_writes_the_file_and_a_matching_sha256(tmp_path: Path):
    import hashlib

    import data30 as D

    items, expl, fixed = _toy_bank()
    info = D.freeze(D.build_data(items, expl, fixed), tmp_path)
    blob = (tmp_path / "data30.json").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == info["sha256"]
    assert (tmp_path / "data30.sha256").read_text().split()[0] == info["sha256"]
    assert json.loads(blob)["seed"] == 30


def test_statement_said_by_state_splits_by_true_value():
    from lib30 import statement_said_by_state
    pair = {"inside": "It is inside", "outside": "It is outside"}
    items = [{"id": "a", "value": "inside"}, {"id": "b", "value": "inside"}, {"id": "c", "value": "inside"},
             {"id": "d", "value": "outside"}, {"id": "e", "value": "outside"}]
    texts = {"a": "x. It is inside.", "b": "it  is OUTSIDE", "c": "It is inside. It is outside.", "d": None, "e": "It is outside"}
    by = statement_said_by_state(texts, items, pair)
    assert by["inside"]["n"] == 3 and by["outside"]["n"] == 2
    assert by["inside"]["said"] == {"inside": 1 / 3, "outside": 1 / 3, "both": 1 / 3, "nothing": 0.0}
    assert by["outside"]["said"] == {"inside": 0.0, "outside": 0.5, "both": 0.0, "nothing": 0.5}
