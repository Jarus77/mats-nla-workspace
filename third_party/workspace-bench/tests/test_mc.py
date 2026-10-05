import hashlib
import random

from wsbench.mc import CANNOT, classify, join_samples, listing, seed_int, seeded_shuffle


def test_seed_int_matches_source_formula():
    assert seed_int("x") == int.from_bytes(hashlib.sha256(b"x").digest()[:8], "big")


def test_seeded_shuffle_is_deterministic_and_equals_index_shuffle():
    opts = ["a", "b", "c", "d", "e"]
    order = list(range(5))
    random.Random(seed_int("k")).shuffle(order)
    assert seeded_shuffle(opts, "k") == [opts[j] for j in order]
    assert seeded_shuffle(opts, "k") == seeded_shuffle(opts, "k")
    assert seeded_shuffle(opts, "k") != seeded_shuffle(opts, "k2")
    assert opts == ["a", "b", "c", "d", "e"]  # copy, not in place


def test_listing():
    assert listing(["x", CANNOT]) == "  1. x\n  2. cannot tell from the readout"


def test_classify():
    assert classify(None, 2, 6) == "invalid"
    assert classify("2", 2, 6) == "invalid"
    assert classify(0, 2, 6) == "invalid"
    assert classify(7, 2, 6) == "invalid"
    assert classify(True, 1, 6) == "invalid"
    assert classify(6, 2, 6) == "cannot_tell"
    assert classify(2, 2, 6) == "gold"
    assert classify(3, 2, 6) == "distractor"
    assert classify(3, 2, 6, contrast_pos=3) == "contrast"


def test_join_samples():
    assert join_samples(["a", "  ", "", "b"]) == "a\nb"
    assert join_samples(["  "]) == ""
