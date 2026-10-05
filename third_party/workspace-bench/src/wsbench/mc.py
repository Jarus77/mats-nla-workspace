"""Helpers every multiple-choice family shares: seeds, shuffles, listings, pick labels."""

import hashlib
import random
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

CANNOT = "cannot tell from the readout"


def seed_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def seeded_shuffle[T](items: Sequence[T], seed: str) -> list[T]:
    out = list(items)
    random.Random(seed_int(seed)).shuffle(out)
    return out


def listing(options: Sequence[str]) -> str:
    return "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(options))


def classify(choice: Any, gold_pos: int, n_shown: int, *, contrast_pos: int | None = None) -> str:
    """``invalid`` | ``cannot_tell`` (CANNOT is always the last line) | ``gold`` | ``contrast`` |
    ``distractor``."""
    if isinstance(choice, bool) or not isinstance(choice, int) or not (1 <= choice <= n_shown):
        return "invalid"
    if choice == n_shown:
        return "cannot_tell"
    if choice == gold_pos:
        return "gold"
    if contrast_pos is not None and choice == contrast_pos:
        return "contrast"
    return "distractor"


def join_samples(samples: Sequence[str]) -> str:
    return "\n".join(s for s in samples if s.strip())


def fold(s: str) -> str:
    """Accent-, case- and curly-quote-insensitive form for verbatim-quote checks."""
    s = unicodedata.normalize("NFKD", s)
    for a, b in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"')):
        s = s.replace(a, b)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def letter_index(choice: Any, options: Sequence[str], letters: str = "ABCDEF") -> int | None:
    """The option index a judge's ``choice`` names: a lone letter (``"B"``, ``"b."``) or the
    letter followed by the text of that same option (``"B. blue ladder"``). Anything else, a
    letter past the list, or a letter with another option's text is None (``"cannot tell"`` is
    not option C)."""
    m = re.fullmatch(r"([A-Za-z])(?:\s*[.):-]?\s*(.*))?", str(choice or "").strip(), re.S)
    if not m:
        return None
    idx = letters.find(m.group(1).upper())
    if idx < 0 or idx >= len(options):
        return None
    rest = (m.group(2) or "").strip().rstrip(".,;:")
    return idx if not rest or fold(rest) == fold(options[idx]) else None
