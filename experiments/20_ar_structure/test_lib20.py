import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib20 as L  # noqa: E402


def test_windows_take_the_last_w_tokens_and_cap_at_length():
    ids = list(range(50))
    w = L.windows(ids, (4, 32, 128))
    assert w[4] == [46, 47, 48, 49] and w[32] == list(range(18, 50)) and w[128] == ids


def test_shuffle_keeps_the_multiset_and_changes_the_order():
    ids = list(range(32))
    s = L.shuffle_tokens(ids, random.Random(0))
    assert sorted(s) == ids and s != ids


def test_explanation_text_falls_back_to_raw():
    assert L.explanation_text("<explanation>Inside a string literal</explanation>") == "Inside a string literal"
    assert L.explanation_text("no tags here at all") == "no tags here at all"
    assert L.explanation_text("") == "(empty)"


def test_table_lists_conditions_in_declared_order():
    rows = [{"feature": "py_string", "auroc": {"span_32": 0.9, "span_4": 0.6, "bag_32": 0.5, "ol_readout": 0.55, "nla_readout": 0.6}}]
    md = L.table(rows)
    assert md.splitlines()[0].startswith("| row |") and "py_string" in md and "0.90" in md
