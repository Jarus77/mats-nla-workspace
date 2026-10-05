"""The jlens_concept_pr scoring numerics, ported unchanged from the source repo's
``tests/test_concept_pr.py`` (plus a ``token_types`` smoke test for the inlined ``is_cjk``)."""

import math

import pytest

from wsbench.evals.jlens_concept_pr.concept_pr import (
    ItemScore,
    _stems,
    bootstrap_ci,
    expected_max_grade,
    is_content_token,
    lexical_support,
    match_keys,
    score_item,
)
from wsbench.evals.jlens_concept_pr.token_types import CONTENT_TYPES, TokenType, classify_token


def test_classify_token_rules() -> None:
    assert classify_token("line\n") == TokenType.NEWLINE
    assert classify_token("  ") == TokenType.SPACE
    assert classify_token(".") == TokenType.TERMINATOR
    assert classify_token("。") == TokenType.TERMINATOR
    assert classify_token(",") == TokenType.COMMA
    assert classify_token(":") == TokenType.COLON
    assert classify_token(" (") == TokenType.OTHER_PUNCT
    assert classify_token("**") == TokenType.SYMBOL
    assert classify_token("访问") == TokenType.CJK
    assert classify_token(" Mr.") == TokenType.WORD
    assert {TokenType.WORD, TokenType.CJK} == CONTENT_TYPES


def test_is_content_token() -> None:
    assert is_content_token(" access")
    assert is_content_token("访问")
    assert is_content_token("42")
    for punct in [";", "();", ";}", "\n", " ", ":", "**", ",", ".", " =", "~"]:
        assert not is_content_token(punct), punct
    # emoji carry meaning (✅ = "correct"); a lone variation selector does not
    for emoji in [" ✅", "✅", " ✔", " ✓", "🎉", " 😊"]:
        assert is_content_token(emoji), emoji
    assert not is_content_token("️")


def test_expected_max_grade_small_set_is_plain_max() -> None:
    assert expected_max_grade([0, 0.5, 0], m=10) == 0.5
    assert expected_max_grade([], m=10) == 0.0
    assert expected_max_grade([1.0] * 10, m=10) == 1.0


def test_expected_max_grade_hypergeometric_known_value() -> None:
    # C=20 concepts, exactly one graded "in", m=10: P(hit) = 1 - C(19,10)/C(20,10) = 0.5
    grades = [1.0] + [0.0] * 19
    assert expected_max_grade(grades, m=10) == pytest.approx(0.5)
    # C=20, one "in" and one "partial": P_in = 0.5; P(>=partial) = 1 - C(18,10)/C(20,10)
    # = 1 - 43758/184756 = 0.76316...; E = 0.5 + 0.5*(0.76316 - 0.5) = 0.63158
    grades = [1.0, 0.5] + [0.0] * 18
    assert expected_max_grade(grades, m=10) == pytest.approx(0.631578947, abs=1e-6)
    # all partial: E = 0.5 exactly
    assert expected_max_grade([0.5] * 30, m=10) == pytest.approx(0.5)


def test_score_item_precision_and_recall() -> None:
    # 2 content tokens x 3 concepts
    grades = [
        [1.0, 0.0, 0.0],  # token 0 supported by concept 0
        [0.0, 0.5, 0.0],  # token 1 partially by concept 1
    ]
    s = score_item(grades, n_concepts=3, n_tokens_total=4, m=10)
    assert isinstance(s, ItemScore)
    assert s.precision == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert s.recall_at_m == pytest.approx((1.0 + 0.5) / 2)  # C=3 <= m -> plain max
    assert s.raw_recall == pytest.approx((1.0 + 0.5) / 2)
    assert (s.n_concepts, s.n_content_tokens, s.n_tokens) == (3, 2, 4)
    # explicit support (the Stage P grades) is the precision instrument
    s = score_item(grades, n_concepts=3, n_tokens_total=4, m=10, support=[0.0, 0.0, 1.0])
    assert s.precision == pytest.approx(1 / 3)


def test_score_item_no_content_tokens_gives_nan_everything() -> None:
    # all-punctuation J-lens top-k: no information about precision OR recall
    s = score_item([], n_concepts=3, n_tokens_total=4)
    assert math.isnan(s.precision)
    assert math.isnan(s.recall_at_m) and math.isnan(s.raw_recall)


def test_score_item_no_concepts() -> None:
    s = score_item([[], []], n_concepts=0, n_tokens_total=2)
    assert s.precision == 0.0
    assert s.recall_at_m == 0.0 and s.raw_recall == 0.0


def test_bootstrap_ci_brackets_mean_and_skips_nan() -> None:
    vals = [0.2, 0.4, 0.6, 0.8, float("nan")]
    lo, hi = bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo <= 0.5 <= hi
    assert lo == pytest.approx(bootstrap_ci(vals, n_boot=500, seed=1)[0])  # seeded
    assert all(math.isnan(x) for x in bootstrap_ci([0.3]))


def test_lexical_support_shares_a_word_with_a_token() -> None:
    # stem match: the token's plural meets the concept's singular and vice versa
    assert lexical_support("traditional materials", [" Materials"]) == 1.0
    assert lexical_support("approval rates", [" rate", " credit"]) == 1.0
    # CJK runs match as substrings (no word boundaries to lean on)
    assert lexical_support("花卉 (flowers)", ["花卉"]) == 1.0
    # thematic neighbours that share no word are NOT support
    assert lexical_support("war trauma", [" reading", " books", " texts"]) == 0.0
    assert lexical_support("nested abbreviations", ["P", "S", "R"]) == 0.0
    # a concept with no matchable key never scores
    assert lexical_support("...", [" anything"]) == 0.0


def test_lexical_support_ignores_short_words_that_would_match_everything() -> None:
    assert "the" not in match_keys("the end of the line")
    assert lexical_support("the a of", [" the", " a"]) == 0.0


def test_stopwords_are_filtered_on_the_raw_word_not_the_stem() -> None:
    """ "news" stems to "new", a stopword — filtering after stemming deleted it silently."""
    assert "new" in match_keys("breaking news")
    assert lexical_support("breaking news", [" news"]) == 1.0
    for word in ("others", "themed", "ones", "noted", "owned"):
        assert match_keys(word), word
    # a raw stopword still carries no support
    assert lexical_support("the other thing", [" other", " the"]) == 0.0


def test_stemmer_does_not_manufacture_false_positives() -> None:
    """-ed/-ing stripping is length-gated: "stared" must not become "star"."""
    assert lexical_support("movie star", [" stared", " glance"]) == 0.0
    assert lexical_support("a hat", [" hated", " loved"]) == 0.0
    assert lexical_support("a car", [" cared"]) == 0.0
    assert lexical_support("a rat", [" rated"]) == 0.0
    # long words still get the benefit
    assert lexical_support("scanning documents", [" scanned"]) == 1.0


def test_stemmer_keeps_ambiguous_plural_forms_meeting() -> None:
    assert lexical_support("size range 100", [" Sizes"]) == 1.0  # size+s, not box+es
    assert lexical_support("a box", [" boxes"]) == 1.0  # box+es
    assert lexical_support("virus load", [" viruses"]) == 1.0  # -s not stripped after "us"
    assert lexical_support("class schedule", [" classes"]) == 1.0


def test_cjk_containment_needs_two_characters() -> None:
    """A single CJK character is the analogue of a stopword — 的 is in a third of all concepts."""
    assert lexical_support("中国的传统文化", ["的"]) == 0.0
    assert lexical_support("中国的传统文化", ["传统"]) == 1.0
    assert lexical_support("花卉 (flowers)", ["花卉"]) == 1.0


def test_stems_returns_a_set_and_the_rules_compose() -> None:
    """ "teachings" must reach "teach" the same way "teaching" does."""
    assert isinstance(_stems("teachings"), set)
    assert "teach" in _stems("teachings") and "teach" in _stems("teaching")
    assert lexical_support("teaching fundamentals", [" teachings"]) == 1.0
    # the raw word is always a candidate, so an exact token always matches
    for word in ("virus", "stared", "the", "size"):
        assert word in _stems(word)
    # -s is not stripped after us/is/as: "spring" must not become "spr"
    assert _stems("spring") == {"spring"}
    assert lexical_support("virus load", [" virus"]) == 1.0


def test_single_character_cjk_is_not_a_key_on_either_path() -> None:
    """A lone CJK character is a morpheme, not a word — gate the exact path too."""
    assert match_keys("的") == set()
    # exact-run intersection, not just containment
    assert lexical_support("中国 的 文化", ["的"]) == 0.0
    assert lexical_support("中国 的 文化", ["文化"]) == 1.0
