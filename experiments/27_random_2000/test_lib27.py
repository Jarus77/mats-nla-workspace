import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib27 as L  # noqa: E402


def _items(n_pos: int, n_neg: int) -> list[dict]:
    pos = [{"id": f"p{i}", "meta": {"labels": {"in_string": True}}} for i in range(n_pos)]
    neg = [{"id": f"n{i}", "meta": {"labels": {"in_string": False}}} for i in range(n_neg)]
    return pos + neg


def _texts(items: list[dict]) -> dict:
    return {r: {it["id"]: f"{r} readout for {it['id']}" for it in items} for r in ("ol", "nla")}


def test_select_judge_items_keeps_all_positives_and_exactly_n_neg_negatives():
    items = _items(3, 10)
    sel = L.select_judge_items(items, _texts(items), "in_string", seed=27, n_neg=3)
    assert len(sel) == 2 * (3 + 3)
    for reader in ("ol", "nla"):
        rows = [s for s in sel if s[0] == reader]
        assert sorted(s[1] for s in rows if s[3]) == ["p0", "p1", "p2"]
        assert len([s for s in rows if not s[3]]) == 3
        assert all(s[2] == f"{reader} readout for {s[1]}" for s in rows)


def test_select_judge_items_negatives_match_across_readers_and_seed_is_deterministic():
    items = _items(2, 20)
    a = L.select_judge_items(items, _texts(items), "in_string", seed=27, n_neg=5)
    b = L.select_judge_items(items, _texts(items), "in_string", seed=27, n_neg=5)
    c = L.select_judge_items(items, _texts(items), "in_string", seed=28, n_neg=5)
    assert a == b
    neg_ol = sorted(s[1] for s in a if s[0] == "ol" and not s[3])
    neg_nla = sorted(s[1] for s in a if s[0] == "nla" and not s[3])
    assert neg_ol == neg_nla
    assert sorted(s[1] for s in c if s[0] == "ol" and not s[3]) != neg_ol


def test_select_judge_items_default_n_neg_is_number_of_positives_and_too_many_raises():
    items = _items(4, 9)
    sel = L.select_judge_items(items, _texts(items), "in_string", seed=1)
    assert len([s for s in sel if s[0] == "ol" and not s[3]]) == 4
    with pytest.raises(ValueError):
        L.select_judge_items(items, _texts(items), "in_string", seed=1, n_neg=10)


def test_select_judge_items_treats_positive_depth_as_true():
    items = [{"id": "a", "meta": {"labels": {"paren_depth": 2}}}, {"id": "b", "meta": {"labels": {"paren_depth": 0}}}]
    sel = L.select_judge_items(items, _texts(items), "paren_depth", seed=0, n_neg=1)
    assert {(s[1], s[3]) for s in sel if s[0] == "ol"} == {("a", True), ("b", False)}


def test_wilson_interval_brackets_the_point_and_handles_edges():
    lo, hi = L.wilson(5, 10)
    assert lo < 0.5 < hi and 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    lo0, hi0 = L.wilson(0, 20)
    assert lo0 == 0.0 and 0.16 < hi0 < 0.17
    lo1, hi1 = L.wilson(20, 20)
    assert hi1 == 1.0 and 0.83 < lo1 < 0.84
    assert all(math.isnan(v) for v in L.wilson(0, 0))


@pytest.mark.parametrize("text,expected", [
    ("Yes.", True), ("no", False), ("Answer: yes", True), ("Answer: No\nEvidence: none", False),
    ("YES\nevidence: \"inside an unclosed string\"", True), ("**No**", False), ('{"answer": "yes"}', True),
    ("Verdict: no. The text never says so.", False), ("", None), (None, None), ("maybe", None),
])
def test_parse_verdict(text, expected):
    assert L.parse_verdict(text) is expected


def test_parse_verdict_uses_the_first_line_not_a_later_mention():
    assert L.parse_verdict("no\nevidence: it says 'yes, a string' but about the topic") is False


def _row(reader, i, label, verdict, regex=False, source="e27"):
    return {"reader": reader, "id": i, "label": label, "verdict": verdict, "regex": regex, "source": source}


def test_judge_rates_conditional_on_label_per_reader():
    rows = [_row("ol", "a", True, True), _row("ol", "b", True, False), _row("ol", "c", False, False), _row("ol", "d", False, False),
            _row("nla", "a", True, True, regex=True), _row("nla", "b", True, True), _row("nla", "c", False, True), _row("nla", "d", False, None)]
    r = L.judge_rates(rows)
    assert r["ol"]["n_pos"] == 2 and r["ol"]["n_neg"] == 2
    assert r["ol"]["yes_pos"] == 1 and r["ol"]["yes_neg"] == 0
    assert r["ol"]["rate_pos"] == 0.5 and r["ol"]["rate_neg"] == 0.0
    assert r["ol"]["regex_rate_pos"] == 0.0
    assert r["nla"]["rate_pos"] == 1.0 and r["nla"]["regex_rate_pos"] == 0.5
    assert r["nla"]["n_unparsed"] == 1 and r["nla"]["n_neg"] == 1 and r["nla"]["rate_neg"] == 1.0
    assert r["ol"]["ci_pos"] == list(L.wilson(1, 2)) and r["ol"]["ci_neg"] == list(L.wilson(0, 2))


def test_judge_rates_empty_class_gives_none_rate():
    r = L.judge_rates([_row("ol", "a", True, True)])
    assert r["ol"]["rate_neg"] is None and r["ol"]["n_neg"] == 0


def test_judge_prompt_contains_only_the_readout_and_asks_for_yes_or_no_first():
    p = L.judge_prompt("the cursor sits inside an unclosed string literal")
    assert "the cursor sits inside an unclosed string literal" in p
    assert "yes" in p.lower() and "no" in p.lower()
    assert "string literal" in p.lower()


def test_judge_prompt_truncates_long_readouts():
    p = L.judge_prompt("x" * 5000)
    assert len(p) < 5000 + 1500


def test_cache_key_and_jsonl_roundtrip(tmp_path: Path):
    assert L.cache_key("e27", "ol", "rnd_py_3") == "e27/ol/rnd_py_3"
    assert L.cache_key("e23", "ol", "rnd_py_3") != L.cache_key("e27", "ol", "rnd_py_3")
    p = tmp_path / "j.jsonl"
    assert L.load_cache(p) == {}
    rows = [_row("ol", "a", True, True), _row("nla", "a", True, None)]
    for r in rows:
        L.append_cache(p, {**r, "key": L.cache_key(r["source"], r["reader"], r["id"])})
    c = L.load_cache(p)
    assert set(c) == {"e27/ol/a", "e27/nla/a"} and c["e27/ol/a"]["verdict"] is True
    assert all(json.loads(l) for l in p.read_text().splitlines())


def test_render_judge_table_lists_both_readers_and_sources():
    rates = {"e27": {"ol": {"n_pos": 10, "n_neg": 10, "yes_pos": 1, "yes_neg": 0, "rate_pos": 0.1, "rate_neg": 0.0,
                            "ci_pos": [0.0, 0.4], "ci_neg": [0.0, 0.3], "regex_rate_pos": 0.0, "regex_rate_neg": 0.0, "n_unparsed": 0}}}
    md = L.render_judge_table(rates)
    assert "| e27 | ol | 10 / 10 | 0.10 [0.00, 0.40] | 0.00 [0.00, 0.30] | 0.00 | 0.00 | 0 |" in md


def test_render_strict_side_by_side_puts_exp23_next_to_exp27():
    s27 = {"in_string": {"ol": {"n_true": 500, "n_false": 1500, "named_when_true": 0.0, "named_when_false": 0.0},
                         "nla": {"n_true": 500, "n_false": 1500, "named_when_true": 0.04, "named_when_false": 0.01}}}
    s23 = {"in_string": {"ol": {"n_true": 134, "n_false": 366, "named_when_true": 0.0, "named_when_false": 0.0},
                         "nla": {"n_true": 134, "n_false": 366, "named_when_true": 0.03, "named_when_false": 0.01}}}
    md = L.render_strict_table(s27, s23)
    assert "| in_string | 500 / 1500 | 0.00 [0.00, 0.01] | 0.00 | 0.04 [0.03, 0.06] | 0.01 | 0.00 / 0.03 (134) |" in md
    md_alone = L.render_strict_table(s27, None)
    assert "in_string" in md_alone and "(134)" not in md_alone
