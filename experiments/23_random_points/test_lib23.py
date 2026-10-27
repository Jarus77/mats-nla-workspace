import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib23 as L  # noqa: E402


def test_string_state_from_quote_parity():
    assert L.labels('x = "abc', "py")["in_string"] is True
    assert L.labels('x = "abc" + y', "py")["in_string"] is False
    assert L.labels("s = 'it", "js")["in_string"] is True


def test_paren_and_bracket_depth():
    assert L.labels("f(a, g(b", "py")["paren_depth"] == 2
    assert L.labels("f(a) + b", "py")["paren_depth"] == 0
    assert L.labels('{"a": {"b": "c', "json")["brace_depth"] == 2


def test_math_state_in_tex():
    assert L.labels("we have $x + y", "tex")["in_math"] is True
    assert L.labels("we have $x$ and then", "tex")["in_math"] is False


def test_markdown_list_and_heading():
    lab = L.labels("# Title\n\n- one\n  - two", "md")
    assert lab["list_depth"] == 2 and lab["heading_level"] == 0
    assert L.labels("## Sec", "md")["heading_level"] == 2


def test_candidate_positions_end_on_word_chars():
    pts = L.candidate_points("alpha beta\ngamma delta", min_ctx=0)
    assert pts and all("alpha beta\ngamma delta"[p - 1].isalnum() for p in pts)


def test_names_state_is_state_specific_and_ignores_content_words():
    assert L.names_state("the cursor is inside a string literal that has not closed", "in_string") is True
    assert L.names_state("an open-source library for math", "in_string") is False
    assert L.names_state("an open-source library for math", "in_math") is False
    assert L.names_state("still in math mode after the opening $", "in_math") is True
    assert L.names_state("continuing the same argument list inside the call", "paren_depth") is True
    assert L.names_state("a nested bullet item under the second list item", "list_depth") is True
    assert L.names_state("the heading of section 2", "heading_level") is True


def test_naming_by_state_gives_conditional_rates():
    items = [{"id": "a", "meta": {"labels": {"in_string": True}}}, {"id": "b", "meta": {"labels": {"in_string": True}}},
             {"id": "c", "meta": {"labels": {"in_string": False}}}, {"id": "d", "meta": {"labels": {"in_string": False}}}]
    texts = {"a": "inside a string literal", "b": "plain prose", "c": "plain prose", "d": "more prose"}
    r = L.naming_by_state(texts, items, "in_string")
    assert r == {"n_true": 2, "n_false": 2, "named_when_true": 0.5, "named_when_false": 0.0}
