import random
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import items_build18 as IB  # noqa: E402
from feature_spec18 import FEATURES  # noqa: E402

SYNTH = ["json_depth", "latex_math", "py_string", "call_arg", "md_heading", "csv_col", "html_depth"]


@pytest.mark.parametrize("feature", SYNTH)
def test_synthetic_rows_are_balanced_and_well_formed(feature):
    items = IB.BUILDERS[feature](random.Random(0), 40)
    assert len(items) == 40
    vals = FEATURES[feature]["values"]
    assert sum(it["value"] == vals[0] for it in items) == 20
    for it in items:
        assert it["feature"] == feature and it["value"] in vals and it["text"] and it["cluster"]
        assert it["text"] == it["text"].rstrip()  # read position is the last token
    assert len({it["id"] for it in items}) == 40 and len({it["cluster"] for it in items}) == 40


def test_json_depth_reads_at_end_of_a_string_value():
    for it in IB.build_json_depth(random.Random(1), 10):
        assert it["text"].endswith('"'), it["text"]
        assert it["text"].count("{") - it["text"].count("}") == it["meta"]["depth"]
        assert (it["meta"]["depth"] == 3) == (it["value"] == "deep")


def test_latex_math_inside_has_odd_dollar_count():
    for it in IB.build_latex_math(random.Random(2), 10):
        odd = it["text"].count("$") % 2 == 1
        assert odd == (it["value"] == "inside")


def test_py_string_inside_has_open_quote():
    for it in IB.build_py_string(random.Random(3), 10):
        q = it["meta"]["quote"]
        assert (it["text"].count(q) % 2 == 1) == (it["value"] == "inside")


def test_call_arg_index_matches_commas_in_open_call():
    for it in IB.build_call_arg(random.Random(4), 10):
        inside = it["text"][it["text"].rfind("(") + 1:]
        assert inside.count(",") == {"first": 0, "third": 2}[it["value"]]
        assert ")" not in inside


def test_md_heading_level_matches_hashes():
    for it in IB.build_md_heading(random.Random(5), 10):
        last = it["text"].splitlines()[-1]
        assert last.startswith({"h1": "# ", "h3": "### "}[it["value"]])
        assert 4 <= len(last.split()) - 1 <= 8


def test_csv_col_field_index():
    for it in IB.build_csv_col(random.Random(6), 10):
        last = it["text"].splitlines()[-1]
        assert last.count(",") == {"first": 0, "fourth": 3}[it["value"]]


def test_html_depth_counts_open_tags():
    for it in IB.build_html_depth(random.Random(7), 10):
        opened = len(re.findall(r"<([a-z]+)>", it["text"]))
        closed = len(re.findall(r"</([a-z]+)>", it["text"]))
        assert opened - closed == {"shallow": 1, "deep": 3}[it["value"]]


def test_stance_word_uses_paragraphs_and_reads_15_to_30_words_in():
    paras = [" ".join(f"word{i}" for i in range(60)) + "." for _ in range(30)]
    items = IB.build_stance_word(paras, random.Random(8))
    assert len(items) == 30 and sum(it["value"] == "unfortunately" for it in items) == 15
    for it in items:
        head = it["text"].split(",", 1)[0]
        assert head in ("Unfortunately", "Fortunately")
        assert 15 <= len(it["text"].split(",", 1)[1].split()) <= 30
        assert it["cluster"] == it["meta"]["para"]


@pytest.mark.parametrize("feature", SYNTH)
def test_synthetic_rows_are_length_matched_between_values(feature):
    import statistics as st

    items = IB.BUILDERS[feature](random.Random(11), 200)
    vals = FEATURES[feature]["values"]
    med = [st.median(len(it["text"]) for it in items if it["value"] == v) for v in vals]
    assert max(med) <= 1.15 * min(med), (feature, med)


def test_length_band_keeps_only_items_inside_the_token_band():
    items = [{"ids": [0] * n, "value": "a"} for n in (5, 40, 100, 300)]
    assert [len(it["ids"]) for it in IB.length_band(items, 30, 250)] == [40, 100]
