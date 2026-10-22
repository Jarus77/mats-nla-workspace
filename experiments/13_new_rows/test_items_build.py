import random
import re

import items_build as IB
from feature_spec import BINS, FEATURES, QA_WORDS_IN, SENT_WORDS_IN

PARA = ("The river rises in the northern hills and flows south for many miles. It passes three towns "
        "before reaching the plain (where the valley widens considerably) and then it slows. Farmers "
        "have used its water for centuries, and the old mills still stand along the banks. In the "
        "nineteenth century a canal was cut beside it to carry coal. The canal closed in 1952 when "
        "the railway took the traffic. Today the towpath is a walking route, and the locks are kept "
        "by volunteers who meet every Sunday morning near the bridge.")


def _rng():
    return random.Random(IB.SEED)


def _check(items, feature):
    vals = FEATURES[feature]["values"]
    assert items, feature
    for it in items:
        assert it["feature"] == feature and it["value"] in vals
        assert it["text"] and it["text"][-1].isalnum(), it["text"][-30:]
        assert set(it) >= {"id", "feature", "cluster", "value", "text", "meta"}


def test_word_ends_only_after_words_followed_by_nonalnum():
    assert IB.word_ends("ab cd, ef") == [2, 5, 9]
    assert IB.word_ends("ab1 cd") == [6]  # 'ab' is followed by a digit


def test_sentcount_values_and_matched_words_in():
    items = IB.build_sentcount([PARA] * 4, _rng())
    _check(items, "tb_sentcount")
    for it in items:
        lo, hi = BINS["tb_sentcount"][it["value"]]
        assert lo <= it["meta"]["sentence"] <= hi
        assert SENT_WORDS_IN[0] <= it["meta"]["words_in"] <= SENT_WORDS_IN[1]


def test_paren_inside_and_outside():
    items = IB.build_paren([PARA] * 4, _rng())
    _check(items, "tb_paren")
    for it in items:
        t = it["text"]
        if it["value"] == "inside":
            assert t.count("(") == t.count(")") + 1
        else:
            assert t.count("(") == t.count(")") and ")" in t


def test_wordsleft_counts():
    items = IB.build_wordsleft([PARA] * 4, _rng())
    _check(items, "tb_wordsleft")
    for it in items:
        body = it["text"].split("\n\n", 1)[1]
        assert IB.n_words(body) == it["meta"]["words_so_far"]
        lo, hi = BINS["tb_wordsleft"][it["value"]]
        assert lo <= it["meta"]["limit"] - IB.n_words(body) <= hi
        assert f"exactly {it['meta']['limit']} words" in it["text"]


def test_qa_same_body_two_layouts():
    items = IB.build_qa([PARA] * 4, _rng())
    _check(items, "fmt_qa")
    for it in items:
        if it["value"] == "qa":
            assert it["text"].startswith("Question: ") and "\nAnswer: " in it["text"]
        else:
            assert not it["text"].startswith("Question")
        assert QA_WORDS_IN[0] <= it["meta"]["words"] <= QA_WORDS_IN[1]


def test_tablecol_cut_in_right_column():
    items = IB.build_tablecol(_rng(), 10)
    _check(items, "tb_tablecol")
    for it in items:
        last = it["text"].split("\n")[-1]
        assert last.count("|") == it["meta"]["col"]  # '| a | b' -> 2 pipes for col 2
        lo, hi = BINS["tb_tablecol"][it["value"]]
        assert lo <= it["meta"]["col"] <= hi
        assert it["text"].count("\n") == it["meta"]["row"] + 1  # header + separator + rows


def test_codeline_line_number_and_flat_indent():
    items = IB.build_codeline(_rng(), 10)
    _check(items, "tb_codeline")
    for it in items:
        lines = it["text"].split("\n")
        assert len(lines) - 1 == it["meta"]["line"]
        assert all(ln.startswith("    ") and not ln.startswith("     ") for ln in lines[1:])
        lo, hi = BINS["tb_codeline"][it["value"]]
        assert lo <= it["meta"]["line"] <= hi


def test_listdepth_depth_of_last_line():
    items = IB.build_listdepth(_rng(), 10)
    _check(items, "fmt_listdepth")
    for it in items:
        last = it["text"].split("\n")[-1]
        indent = len(last) - len(last.lstrip(" "))
        assert indent // 2 + 1 == it["meta"]["depth"]
        assert re.match(r" *- \S", last)


def test_assign_splits_counts():
    items = [{"value": v, "id": f"{v}{i}"} for v in ("a", "b") for i in range(30)]
    out = IB.assign_splits(items, _rng(), 5, 10)
    assert sum(o["split"] == "mine" for o in out) == 10 and sum(o["split"] == "select" for o in out) == 20
