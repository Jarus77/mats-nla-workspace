import math

import numpy as np
import pytest

from lib31 import (auroc, delete_tokens, derangement, is_content, low_half_mask, ngram_overlap, ngrams, norm_piece,
                   opposite_partners, paired_cluster_bootstrap, strip_tags, token_overlap, words)


def test_norm_piece_strips_and_keeps_whitespace_tokens():
    assert norm_piece(" Cannot") == "cannot"
    assert norm_piece("\n") == "\n"
    assert norm_piece("  ") == "  "


def test_is_content():
    assert is_content("abc") and is_content("3") and is_content("类型")
    assert not is_content('"') and not is_content("__") and not is_content("\n")


def test_token_overlap_counts_content_only():
    assert token_overlap(["cannot", "cast", '"', "x"], {"cannot", '"'}) == pytest.approx(1 / 3)
    assert token_overlap(['"', ")"], {'"'}) is None


def test_words_and_ngrams():
    assert words("Raise ValueError(`position_ids` must") == ["raise", "valueerror", "position_ids", "must"]
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert ngrams(["a"], 2) == []


def test_ngram_overlap():
    r = words("must be provided when used")
    w = words("ids must be provided w")
    assert ngram_overlap(r, w, 2) == pytest.approx(2 / 4)
    assert ngram_overlap(r, w, 3) == pytest.approx(1 / 3)
    assert ngram_overlap(["one"], w, 2) is None


def test_derangement_has_no_fixed_points_and_is_deterministic():
    for n in (2, 3, 10, 160):
        p = derangement(n, 31)
        assert sorted(p) == list(range(n)) and all(i != j for i, j in enumerate(p))
    assert derangement(50, 31) == derangement(50, 31)
    with pytest.raises(ValueError):
        derangement(1, 0)


def test_opposite_partners():
    y = [1, 0, 1, 0, 0]
    p = opposite_partners(y, 31)
    assert all(y[i] != y[j] for i, j in enumerate(p))
    with pytest.raises(ValueError):
        opposite_partners([1, 1], 0)


def test_delete_tokens():
    norm = {1: "cannot", 2: '"', 3: "cast", 4: "x"}.get
    assert delete_tokens([1, 2, 3, 4, 2], norm, {"cannot", '"'}) == [3, 4]
    assert delete_tokens([], norm, {"a"}) == []


def test_strip_tags():
    assert strip_tags("<explanation>\nhi</explanation>") == "\nhi"
    assert strip_tags(None) == ""


def test_auroc_matches_definition_with_ties():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5
    assert math.isnan(auroc([1, 2], [1, 1]))
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=200), rng.integers(0, 2, 200)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert auroc(s, y) == pytest.approx(brute)


def test_paired_bootstrap_is_paired_and_brackets():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    g = np.repeat(np.arange(60), 5)
    ci = paired_cluster_bootstrap({"m": lambda i: x[i].mean(), "zero": lambda i: x[i].mean() - x[i].mean()}, g,
                                  n_boot=300)
    lo, hi = ci["m"]
    assert lo < x.mean() < hi
    assert ci["zero"] == (0.0, 0.0)


def test_low_half_mask():
    m = low_half_mask([0.1, None, 0.5, 0.9])
    assert m.tolist() == [True, True, False, False]
