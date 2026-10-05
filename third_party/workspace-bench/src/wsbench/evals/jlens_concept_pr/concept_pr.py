"""Concept precision / recall@m of a verbalizer's text against the J-lens top-k.

Copied whole from the source repo's ``src/global_workspace/judges/concept_pr.py`` (only the
``token_types`` import is rewired); the scoring numerics are unchanged.

Pure math over a judge's grade grid ``grades[t][c]`` (content J-lens token t, concept c;
values 0 / 0.5 / 1 = out / partial / in), plus the deterministic precision criterion the grid does
NOT supply. Spec: docs/superpowers/specs/2026-09-12-jlens-concept-pr-eval-design.md §6 (its
precision definition was amended 2026-09-13 — see the audit doc below).

- precision   = mean_c support(c)            (length-free). ``support`` is supplied by the caller:
                the headline instrument is the LLM Stage P grade (claude-sonnet-5);
                ``score_pr.py --precision lexical`` swaps in ``lexical_support`` (deterministic:
                does the concept share a word with some token?). With NO support grades it falls
                back to mean_c max_t g(c, t), which overstated precision under the lenient Flash
                judge and must not be used as the headline. See
                docs/project/experiments/ola/jlens_pr_judge_audit.md.
- recall@m    = mean_t E[max_{c in S} g(c, t)] over a uniform m-subset S of the concepts,
                computed EXACTLY via the hypergeometric tail (no sampling)
- raw recall  = mean_t max_c g(c, t)

Punctuation J-lens tokens are not content (``is_content_token``) and are excluded from the
recall denominator before this module sees them.
"""

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# CONTENT_TYPES = {WORD, CJK}, defined once in token_types (not redefined here)
from wsbench.evals.jlens_concept_pr.token_types import CONTENT_TYPES, classify_token

__all__ = [
    "CONTENT_TYPES",
    "ItemScore",
    "bootstrap_ci",
    "expected_max_grade",
    "has_emoji",
    "is_content_token",
    "lexical_support",
    "match_keys",
    "score_item",
]

# --- deterministic precision instrument -------------------------------------------------------
# A concept counts as supported when it shares a WORD with some content token. Words are matched
# on a crude stem so "materials"/"material" and "rates"/"rate" meet; CJK runs match by containment
# (no word boundaries to lean on). Against 120 hand labels it fires on ONE concept a human marked
# unsupported (1/120), and its shuffled-foil floor is 0.0009-0.0036 versus 0.13-0.16 for the
# max-over-tokens LLM grid it replaces. It is a deliberately CONSERVATIVE lower bound: it recovers
# 38 % (NLA) / 55 % (s3d) of the concepts a human calls supported, missing translations and
# synonyms, so absolute levels read low and the arm gap is the part to trust.
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
# function words carry no support: they are frequent in both concepts and readouts, so matching
# on them would be the same lexical-coincidence effect this instrument exists to avoid
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "from",
        "this",
        "these",
        "those",
        "into",
        "onto",
        "over",
        "under",
        "between",
        "about",
        "are",
        "was",
        "were",
        "will",
        "would",
        "can",
        "could",
        "should",
        "have",
        "has",
        "had",
        "not",
        "but",
        "its",
        "their",
        "your",
        "our",
        "his",
        "her",
        "they",
        "them",
        "then",
        "than",
        "there",
        "here",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "what",
        "how",
        "why",
        "all",
        "any",
        "some",
        "each",
        "more",
        "most",
        "other",
        "another",
        "such",
        "very",
        "just",
        "also",
        "only",
        "own",
        "same",
        "too",
        "now",
        "new",
        "one",
        "two",
        "both",
    }
)


def _stems(word: str) -> set[str]:
    """Candidate stems of ``word``, always including ``word`` itself.

    A SET, not one string, because English is ambiguous and a wrong single stem is a silent false
    positive (or, as often, a silent miss) in ``lexical_support``:

    * ``-s`` is not stripped after ``ss``/``us``/``is``/``as`` — otherwise "virus" becomes "viru"
      while "viruses" becomes "virus", and the two forms stop meeting.
    * ``-es`` after a sibilant is genuinely ambiguous: "boxes" is box+es but "sizes" is size+s.
      Both readings are emitted, so "sizes" meets "size" AND "boxes" meets "box".
    * The plural and participle rules COMPOSE, so "teachings" reaches "teach" the same way
      "teaching" does. Applying at most one rule left the plural stranded at "teaching".
    * ``-ed``/``-ing`` are stripped only when at least 5 characters survive. Blind stripping gives
      "stared" -> "star", "hated" -> "hat", "cared" -> "car", "rated" -> "rat", each of which
      credits an unrelated concept. Long words ("scanning" -> "scann") keep the benefit. The gate
      is a trade, not a free win: it also drops 40 legitimate inflection matches at L44 and 130
      over all 12 arm-layers ("flawed"/"flaw", "closing"/"closed") out of 5 083 / 19 648 hits,
      which is the conservative side of a lower bound.
    """
    plural = {word}
    if word.endswith("ies") and len(word) > 4:
        plural.add(word[:-3] + "y")
    elif word.endswith("es") and len(word) > 4 and word[-3] in "sxzho":
        plural |= {word[:-2], word[:-1]}
    elif word.endswith("s") and len(word) > 3 and word[-2:] not in ("ss", "us", "is", "as"):
        plural.add(word[:-1])
    out = set(plural)
    for cand in plural:
        if cand.endswith("ing") and len(cand) >= 8:
            out.add(cand[:-3])
        elif cand.endswith("ed") and len(cand) >= 7:
            out.add(cand[:-2])
    return out


def match_keys(text: str) -> set[str]:
    """Stemmed content words (>=3 letters, non-stopword) and CJK runs (>=2 chars) of ``text``.

    The stopword filter runs on the RAW word, never on the stem: "news" stems to "new", which is a
    stopword, so filtering after stemming silently deleted 604 word occurrences in this corpus
    ("news", "others", "themed", "ones", "noted", "owned", ...) and emptied 23 concepts outright.
    """
    low = text.lower()
    keys = {k for w in _WORD.findall(low) if w not in _STOP for k in _stems(w)}
    # A single CJK character is a morpheme, not a word — the analogue of the >=3-letter floor on
    # Latin words, and of a stopword: 的 alone sits inside a third of all Chinese concepts here.
    return keys | {r for r in _CJK.findall(low) if len(r) >= 2}


def lexical_support(concept: str, tokens: Sequence[str]) -> float:
    """1.0 when some token SHARES A WORD with ``concept`` (or a CJK run), else 0.0.

    Latin-script words must match on the stem exactly. A substring rule was tried and dropped: it
    credited "club" inside "nightclub" and — worse — "finite" inside "infinite", and it doubled
    the shuffled-foil floor while adding only ~10 % more true positives.
    """
    ckeys = match_keys(concept)
    if not ckeys:
        return 0.0
    cjk_ckeys = [c for c in ckeys if _CJK.fullmatch(c)]
    for tok in tokens:
        tkeys = match_keys(tok)
        if tkeys & ckeys:
            return 1.0
        # CJK has no word boundaries, so a run matches when either contains the other.
        # ``match_keys`` already dropped single characters, so both sides are 2+ here.
        for key in tkeys:
            if _CJK.fullmatch(key) and any(key in c or c in key for c in cjk_ckeys):
                return 1.0
    return 0.0


def has_emoji(tok: str) -> bool:
    """True if any char is a pictograph / emoji (Unicode category So, or the emoji-modifier and
    variation-selector planes attached to one). ``*``, ``=``, ``~`` are Po/Sm, not So."""
    return any(unicodedata.category(ch) == "So" for ch in tok)


def is_content_token(tok: str) -> bool:
    """True for a J-lens token that can carry a concept: WORD, CJK, or an emoji (✅ ✔ ✓ carry
    "correct / done" as surely as ``YES`` does — Camila 2026-09-13, chat-lmsys-0014). False for
    punctuation and markdown/math symbols."""
    if classify_token(tok) in CONTENT_TYPES:
        return True
    return has_emoji(tok)


def _p_hit(n_total: int, n_good: int, m: int) -> float:
    """P(a uniform m-subset of n_total contains at least one of the n_good)."""
    if n_good <= 0:
        return 0.0
    if n_total - n_good < m:
        return 1.0
    return 1.0 - math.comb(n_total - n_good, m) / math.comb(n_total, m)


def expected_max_grade(grades: Sequence[float], m: int) -> float:
    """E[max grade over a uniform m-subset]; plain max when there are <= m grades."""
    c = len(grades)
    if c == 0:
        return 0.0
    if c <= m:
        return float(max(grades))
    n_in = sum(1 for g in grades if g >= 1.0)
    n_partial = sum(1 for g in grades if 0.0 < g < 1.0)
    p_in = _p_hit(c, n_in, m)
    p_ge_partial = _p_hit(c, n_in + n_partial, m)
    return p_in + 0.5 * (p_ge_partial - p_in)


@dataclass(frozen=True)
class ItemScore:
    precision: float
    recall_at_m: float
    raw_recall: float
    n_concepts: int
    n_content_tokens: int
    n_tokens: int


def score_item(
    grades: Sequence[Sequence[float]],
    *,
    n_concepts: int,
    n_tokens_total: int,
    m: int = 10,
    support: Sequence[float] | None = None,
) -> ItemScore:
    """Score one (item, layer, arm) cell from its grade grid over CONTENT tokens.

    ``support`` (one grade per concept — the Stage P grade by default in ``score_pr.py``,
    ``lexical_support`` with ``--precision lexical``) is the precision instrument; the grid
    supplies recall either way.
    """
    n_t = len(grades)
    for row in grades:
        if len(row) != n_concepts:
            raise ValueError(f"grade row has {len(row)} entries, expected {n_concepts}")
    if support is not None and len(support) != n_concepts:
        raise ValueError(f"support has {len(support)} grades, expected {n_concepts}")
    if n_t == 0:
        # an item whose J-lens top-k is all punctuation carries no information about either
        # metric: NaN on both axes (dropped from the means), never a 0 that biases them
        return ItemScore(math.nan, math.nan, math.nan, n_concepts, 0, n_tokens_total)
    if n_concepts == 0:
        precision = 0.0
    elif support is not None:
        precision = float(np.mean(support))
    else:
        precision = float(
            np.mean([max(grades[t][c] for t in range(n_t)) for c in range(n_concepts)])
        )
    recall_at_m = float(np.mean([expected_max_grade(row, m) for row in grades]))
    raw_recall = float(np.mean([max(row) if row else 0.0 for row in grades]))
    return ItemScore(precision, recall_at_m, raw_recall, n_concepts, n_t, n_tokens_total)


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Seeded percentile bootstrap (2.5, 97.5) of the mean over non-NaN values."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))
