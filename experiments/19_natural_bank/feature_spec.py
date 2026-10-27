"""exp 07/13 template-pipeline API for exp 18 (so extract_templates.py / select_freeze.py / the judge run
unchanged): FEATURES with desc + values, FIXED per reader, FIXED_JLENS word sets, MENTION_RE, and the small
helpers. Everything concept-specific lives in feature_spec19; this file only adapts the interface."""
from __future__ import annotations

import re

from feature_spec19 import FEATURES, FIXED, N_MINE_PER_VALUE, N_SELECT_PER_VALUE  # noqa: F401

K_READOUTS = 8
OL_LAYERS = (42, 36, 44, 52)       # prereg §6 step 4: OLens at its far trained layers too; 42 primary
NLA_LAYER = 42
N_SELECT_TEMPLATES = 3
QUALITATIVE_PRIMARY: tuple[str, ...] = ()
FIXED_NUMERIC: dict[str, dict] = {}
BINS: dict = {}
BREAK_SOON_MAX, BREAK_FAR_MIN, BREAK_COL_RANGE = 6, 25, (40, 75)   # unused; select_freeze imports them
LENGTH_FEATURES: tuple[str, ...] = ()

FIXED_JLENS: dict[str, dict[str, list[str]]] = {
    "py_string": {"inside": ["string", "quote"], "outside": ["code", "call"]},
    "js_string": {"inside": ["string", "quote"], "outside": ["code", "call"]},
    "latex_math": {"inside": ["math", "equation"], "outside": ["prose", "text"]},
    "py_arg": {"third": ["third", "three"], "first": ["first", "one"]},
    "json_depth": {"deep": ["nested", "deep"], "shallow": ["top", "shallow"]},
    "html_depth": {"deep": ["nested", "deep"], "shallow": ["shallow", "top"]},
    "md_heading": {"deep": ["subsection", "sub"], "shallow": ["title", "chapter"]},
    "md_list": {"nested": ["nested", "indented"], "top": ["top", "main"]},
}

MENTION_RE: dict[str, str] = {
    "py_string": r"string literal|string|quot",
    "js_string": r"string literal|string|quot",
    "latex_math": r"math|equation|latex|\\frac|formula",
    "py_arg": r"argument|param|positional",
    "json_depth": r"nest|depth|level|deep|inner|outer|top[- ]level",
    "html_depth": r"nest|tag|html|depth|<li>|<div>",
    "md_heading": r"heading|header|title|section|##",
    "md_list": r"list|bullet|nested|indent|item",
}

_NUMWORD = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                      r"\w+teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b", re.IGNORECASE)


def is_numeric(template: dict) -> bool:
    return any(_NUMWORD.search(template["template"].replace("{v}", f)) for f in template["fillers"].values())


def fixed_template(reader: str, feature: str) -> dict:
    return {"id": f"{reader}:{feature}:fixed", "family": "fixed_label", **FIXED[reader][feature]}


def fixed_extra(reader: str, feature: str) -> list[dict]:
    return []


def fill(template: dict, value: str, word: str | None = None) -> str:
    f = template["fillers"]
    slot = f["*"] if "*" in f else f[value]
    if word is not None:
        slot = slot.replace("{w}", word).replace("{W}", word[:1].upper() + word[1:])
    return template["template"].replace("{v}", slot)


def validate(template: dict, reader: str, feature: str) -> str | None:
    t = template.get("template", "")
    fl = template.get("fillers", {})
    if not isinstance(t, str) or t.count("{v}") != 1:
        return "template must contain exactly one {v}"
    if reader == "ol" and not t.startswith("- "):
        return "OL template must start with '- '"
    vals = FEATURES[feature]["values"]
    if set(fl) != set(vals):
        return f"fillers must have keys {vals}"
    a, b = (fill(template, v) for v in vals)
    if a == b or not all(isinstance(fl[v], str) and fl[v].strip() for v in vals):
        return "filled candidates must differ and fillers be non-empty"
    if len(t) > 160:
        return "template too long"
    return None
