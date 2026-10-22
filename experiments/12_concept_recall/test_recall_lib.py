import json

from recall_lib import Readout, aggregate, build_prompt, chunks, parse_answers


def _r(item, value, k=0, text="t"):
    return Readout("f", "nla", item, value, k, text)


def test_parse_answers_filters_malformed():
    raw = json.dumps({"answers": [
        {"i": 0, "stated": True, "stated_value": "A", "exhibited_value": "neither"},
        {"i": 5, "stated": True, "stated_value": "A", "exhibited_value": "neither"},  # out of range
        {"i": 1, "stated": False, "stated_value": "maybe", "exhibited_value": "neither"},  # bad enum
        {"i": 2, "stated": False, "stated_value": "none", "exhibited_value": "B"},
    ]})
    out = parse_answers(raw, n=3)
    assert set(out) == {0, 2}
    assert out[0]["stated"] is True and out[2]["exhibited_value"] == "B"


def test_parse_answers_bad_json():
    assert parse_answers("not json", 3) == {}
    assert parse_answers(None, 3) == {}


def test_build_prompt_numbers_and_flattens_newlines():
    p = build_prompt("desc", ["x", "y"], [_r("i1", "x", text="a\nb"), _r("i2", "y", text="c")])
    assert '0. a / b' in p and '1. c' in p and 'A = "x"' in p and 'B = "y"' in p


def test_chunks():
    rows = [_r(f"i{i}", "x") for i in range(45)]
    cs = chunks(rows, 20)
    assert [len(c) for c in cs] == [20, 20, 5]


def test_aggregate_rates():
    judged = [
        {"item": "i1", "value": "x", "stated": True, "stated_value": "A", "exhibited_value": "neither"},
        {"item": "i1", "value": "x", "stated": True, "stated_value": "B", "exhibited_value": "neither"},
        {"item": "i2", "value": "y", "stated": False, "stated_value": "none", "exhibited_value": "B"},
        {"item": "i2", "value": "y", "stated": True, "stated_value": "unclear", "exhibited_value": "A"},
    ]
    a = aggregate(judged, ["x", "y"])
    assert a["n_samples"] == 4 and a["n_items"] == 2
    assert a["stated_rate"] == 0.75
    assert a["stated_correct_rate"] == 0.25
    assert a["stated_wrong_rate"] == 0.25
    assert a["value_acc_given_committed"] == 0.5
    assert a["any_stated_per_item"] == 1.0
    assert a["any_correct_per_item"] == 0.5
    assert a["exhibited_rate"] == 0.5 and a["exhibited_correct_rate"] == 0.25


def test_aggregate_item_denominator_includes_unstated_items():
    judged = [
        {"item": "i1", "value": "x", "stated": True, "stated_value": "A", "exhibited_value": "neither"},
        {"item": "i2", "value": "y", "stated": False, "stated_value": "none", "exhibited_value": "neither"},
        {"item": "i3", "value": "y", "stated": False, "stated_value": "none", "exhibited_value": "neither"},
    ]
    a = aggregate(judged, ["x", "y"])
    assert a["n_items"] == 3
    assert a["any_stated_per_item"] == 1 / 3 and a["any_correct_per_item"] == 1 / 3


def test_aggregate_empty():
    a = aggregate([], ["x", "y"])
    assert a["stated_rate"] is None and a["n_items"] == 0
