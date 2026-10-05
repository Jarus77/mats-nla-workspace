"""FIXED candidate templates for the col pilot, written BEFORE any pilot data was scored.

Not mined (07's mined templates are not ready and must not be informed by steering responses).
Convention: a = the "+" value (more characters since the last newline / line nearly full),
b = the "-" value. LO = score(a) - score(b), so a reader that tracks +Delta*u has LO(+) > LO(-).

Families (lead, Fable round 4):
  count_qual  PRIMARY. Qualitative "long line" (the count; non-J content).
  count_num   SECONDARY. Numerals for the same count.
  decision    "a line break / newline is coming" (the decision col feeds; J content by
              definition). Reported separately: a reader could read either.

NLA: explanation register, scored right after "<explanation>\\n".
OL: native "- " bullets (off its usual continuation-style distribution; see 03 finding 5).
J-Lens: max cosine score over single-token variants of each word list (difference a - b).
        The decision family for the J-Lens is the newline/EOL leak set (pilot_lib.eol_token_ids),
        scored unpaired.
"""

NLA_PAIRS: dict[str, list[tuple[str, str]]] = {
    "count_qual": [
        ("Near the end of a long line of text", "At the start of a line of text"),
        ("A long line of text", "A short line of text"),
    ],
    "count_num": [
        ("The current line is about 60 characters long", "The current line is about 20 characters long"),
    ],
    "decision": [
        ("A line break is coming up soon", "A line break is still far away"),
        ("The next token is likely a newline", "The next token is likely another word"),
    ],
}

OL_PAIRS: dict[str, list[tuple[str, str]]] = {
    "count_qual": [
        ("- near the end of a long line", "- at the start of a line"),
        ("- a long line of text", "- a short line of text"),
    ],
    "count_num": [
        ("- about 60 characters", "- about 20 characters"),
    ],
    "decision": [
        ("- line break coming up", "- line break still far away"),
        ("- next token is a newline", "- next token is another word"),
    ],
}

JLENS_WORDS: dict[str, list[tuple[list[str], list[str]]]] = {
    "count_qual": [
        (["long", "longer"], ["short", "shorter"]),
        (["end", "ending"], ["start", "beginning"]),
    ],
    "count_num": [
        (["sixty", "seventy", "eighty"], ["ten", "fifteen", "twenty"]),
    ],
}

FAMILIES = ("count_qual", "count_num", "decision")
PRIMARY_FAMILY = "count_qual"


def flat_pairs(pairs: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str, str]]:
    """[(family, a, b)] in a fixed order (index = template id t0, t1, ...)."""
    return [(fam, a, b) for fam in FAMILIES for a, b in pairs.get(fam, [])]


# Regex for "does a free readout mention line length / characters / wrapping" (same scope as
# 07's nj1_col mention regex; written independently of any pilot readout).
MENTION_RE = (r"\b(characters?|chars|line[- ]length|line[- ]width|column|wrap\w*|line[- ]?breaks?|"
              r"newlines?|width|long line|short line|end of (the |a )?line|start of (the |a )?line)\b")
# BUG FIX (after first look at NATURAL readouts, before any steering comparison): "character(s)"
# matched fictional characters ("a character named Libra") in most NLA hits. The corrected regex
# keeps only count/line senses of "character". Both rates are reported.
MENTION_RE_STRICT = (r"(\b\d+\s*(-\s*)?(characters?|chars)\b|\bcharacters? (count|long|per line|wide)\b|"
                     r"\b(number of|count of|many|few) characters\b|\bchars\b|\bline[- ]length|\bline[- ]width|"
                     r"\bcolumn (position|width|\d)|\b(line|word|text)[- ]wrap\w*|\bwrap(ped|s)? (at|to) \d|\bline[- ]?breaks?|\bnewlines?\b|\blong line|\bshort line|"
                     r"\bend of (the |a )?line|\bstart of (the |a )?line|\bline is (long|short|full)|"
                     r"\bhard[- ]wrapped|\bfixed[- ]width)")
