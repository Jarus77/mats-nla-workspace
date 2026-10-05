"""Coarse type of an input token, for the meta-token punctuation experiment.

Copied whole from the source repo's ``src/global_workspace/token_types.py`` (jlens_concept_pr needs
``classify_token`` / ``CONTENT_TYPES``); only the ``is_cjk`` import is inlined.

The interpretive meta-tokens are claimed to surface in the J-lens at *punctuation* positions — and,
per the drummer suite, at sentence terminators / newlines much more than at mid-clause commas. To
test that we label every input-token *position* by the kind of token sitting there and ask where the
meta-tokens fire.

Decoded tokens (real characters, not BPE byte pieces) are passed in — e.g. the output of
``ModelBackend.token_strs`` — so a CJK period 。 arrives as the character, not mojibake.

Classification rules (order matters):

1. A token that *contains* a line break is :attr:`TokenType.NEWLINE` — the drummer signature is a
   line-ending phenomenon, so a ``"line\\n"`` token counts as a boundary regardless of its word.
2. A whitespace-only token is :attr:`TokenType.SPACE`.
3. A token whose stripped form is *entirely* non-alphanumeric (pure punctuation/symbol) gets a
   fine-grained boundary label — terminator (ASCII ``. ! ?`` plus the CJK/fullwidth full stop,
   exclamation, question mark and ellipsis), comma (``, ;`` plus fullwidth/ideographic forms),
   colon (ASCII and fullwidth), else ``other_punct``. Requiring purity keeps abbreviations like
   ``"Mr."`` out of the terminator class.
4. Otherwise it is content: :attr:`TokenType.CJK` if it holds a Chinese character, else
   :attr:`TokenType.WORD`.
"""

import unicodedata
from enum import StrEnum

# ``is_cjk`` and ``_CJK_RANGES`` are inlined from the source repo's ``global_workspace/glossary.py``
# (L28, L87-90) so this module has no dependency outside wsbench.
# CJK Unified Ideographs + Ext-A + compatibility ideographs; kana/hangul are intentionally skipped
# (the finding we care about is Chinese hanzi).
_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))


def is_cjk(ch: str) -> bool:
    """True if ``ch`` is a Chinese character we gloss."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


# Sentence-final terminators. CJK/fullwidth forms are built from code points (literal glyphs trip
# ruff RUF001): U+3002 full stop, U+FF01 fullwidth !, U+FF1F fullwidth ?, U+2026 ellipsis.
_TERMINATORS = frozenset(".!?") | {chr(c) for c in (0x3002, 0xFF01, 0xFF1F, 0x2026)}
# Mid-clause separators (the suite's "comma dampens the gate" class): U+FF0C fullwidth comma,
# U+3001 ideographic comma, U+FF1B fullwidth semicolon.
_COMMAS = frozenset(",;") | {chr(c) for c in (0xFF0C, 0x3001, 0xFF1B)}
# Colons: U+FF1A fullwidth colon.
_COLONS = frozenset(":") | {chr(0xFF1A)}
# Markdown / code / math symbols that are NOT prose punctuation (``**`` bold, ``_``, ``~``, ``=``).
# A token of only these (or Unicode symbol-category chars) is a SYMBOL, not punctuation.
_MARKUP = frozenset("*_~=+<>|\\^`#@%&$")


def _is_symbolic(ch: str) -> bool:
    """A markup/math symbol char (markdown ``*``/``_``, operators) rather than prose punctuation."""
    return ch in _MARKUP or unicodedata.category(ch).startswith("S")


class TokenType(StrEnum):
    """Coarse type of an input token at a residual-stream position (JSON-serialisable str enum)."""

    NEWLINE = "newline"  # token contains a line break (\n or \r)
    TERMINATOR = "terminator"  # sentence-final punctuation (ASCII . ! ? and CJK/fullwidth forms)
    COMMA = "comma"  # mid-clause separator (ASCII , ; and CJK/fullwidth forms)
    COLON = "colon"  # ASCII or fullwidth colon
    OTHER_PUNCT = "other_punct"  # other prose punctuation (quotes, brackets, dashes, slashes)
    SYMBOL = "symbol"  # markdown/code/math symbols (* _ ~ = ** …) — NOT prose punctuation
    SPACE = "space"  # whitespace only (no line break)
    CJK = "cjk"  # holds at least one Chinese character
    WORD = "word"  # alphanumeric content (Latin words, digits)


# The sentence-boundary class the headline hypothesis (C1) is stated over.
BOUNDARY_TYPES = frozenset({TokenType.NEWLINE, TokenType.TERMINATOR})
# Everything Unicode would call punctuation (the broad "punctuation" claim).
PUNCT_TYPES = frozenset(
    {
        TokenType.NEWLINE,
        TokenType.TERMINATOR,
        TokenType.COMMA,
        TokenType.COLON,
        TokenType.OTHER_PUNCT,
    }
)
# Content tokens (the C2 "content word" comparison group; CJK kept separate for the controls).
CONTENT_TYPES = frozenset({TokenType.WORD, TokenType.CJK})


def classify_token(tok: str) -> TokenType:
    """Map a decoded input token to its :class:`TokenType` (see module docstring for the rules)."""
    if "\n" in tok or "\r" in tok:
        return TokenType.NEWLINE
    s = tok.strip()
    if s == "":
        return TokenType.SPACE
    # Pure-punctuation/symbol token: assign the fine-grained boundary label. CJK characters are
    # alphanumeric to ``str.isalnum``, so this branch never swallows Chinese content.
    if all(not ch.isalnum() for ch in s):
        if any(ch in _TERMINATORS for ch in s):
            return TokenType.TERMINATOR
        if any(ch in _COMMAS for ch in s):
            return TokenType.COMMA
        if any(ch in _COLONS for ch in s):
            return TokenType.COLON
        if all(_is_symbolic(ch) for ch in s):
            return TokenType.SYMBOL
        return TokenType.OTHER_PUNCT
    # Mixed / content token.
    if any(is_cjk(ch) for ch in s):
        return TokenType.CJK
    return TokenType.WORD
