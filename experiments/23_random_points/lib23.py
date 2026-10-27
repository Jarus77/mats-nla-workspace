"""Exp 23: structural state labels at RANDOM read points in random public documents (no selection by state), so that
naming and carrying can be measured as base rates rather than on contrasted banks. Labels are computed from the text
prefix alone, the same way the exp 19 generators define them."""
from __future__ import annotations

import re

MENTION_ANY = re.compile(r"string literal|quot|parenthes|bracket|brace|nest|depth|indent|column|heading|header|"
                         r"math|equation|list item|bullet|inside|outside|open|closed|level", re.IGNORECASE)


def _unescaped_count(s: str, ch: str) -> int:
    return len(re.findall(r"(?<!\\)" + re.escape(ch), s))


def labels(prefix: str, kind: str) -> dict:
    """State labels at the end of `prefix` for a file of `kind` in {py, js, json, tex, md}."""
    line = prefix.rsplit("\n", 1)[-1]
    lab = {"in_string": False, "paren_depth": 0, "brace_depth": 0, "in_math": False, "list_depth": 0, "heading_level": 0}
    if kind in ("py", "js"):
        dq, sq = _unescaped_count(line, '"'), _unescaped_count(line, "'")
        lab["in_string"] = (dq % 2 == 1) or (sq % 2 == 1)
        lab["paren_depth"] = max(0, prefix.count("(") - prefix.count(")"))
        lab["brace_depth"] = max(0, prefix.count("{") - prefix.count("}"))
    elif kind == "json":
        lab["brace_depth"] = max(0, prefix.count("{") - prefix.count("}"))
        lab["in_string"] = _unescaped_count(line, '"') % 2 == 1
    elif kind == "tex":
        lab["in_math"] = _unescaped_count(prefix, "$") % 2 == 1
        lab["paren_depth"] = max(0, prefix.count("(") - prefix.count(")"))
    elif kind == "md":
        m = re.match(r"( *)[-*] ", line)
        lab["list_depth"] = (len(m.group(1)) // 2 + 1) if m else 0
        h = re.match(r"(#{1,6}) ", line)
        lab["heading_level"] = len(h.group(1)) if h else 0
        lab["paren_depth"] = max(0, line.count("(") - line.count(")"))
    return lab


def candidate_points(text: str, min_ctx: int = 150, max_ctx: int = 1200) -> list[int]:
    """Offsets just after a word character, with enough context before them."""
    return [m.end() for m in re.finditer(r"[A-Za-z0-9]", text) if m.end() >= min_ctx and m.end() <= len(text)]


def names_structure(readout: str) -> bool:
    return bool(MENTION_ANY.search(readout or ""))
