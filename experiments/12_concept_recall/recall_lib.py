"""Pure helpers for judged spontaneous recall of concept types in free readouts (exp 12).

A readout "finds" a concept when it explicitly STATES the property (not merely quotes text that
has it). Separately, a readout may EXHIBIT the value in its own text (an OLens continuation
written in French exhibits 'French' without stating it). Both are judged; both are reported.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Readout source: exp 07 by default; RECALL_GEN_DIR / RECALL_SPEC override it (exp 13 reuses this judge).
GEN_DIR = Path(os.environ.get("RECALL_GEN_DIR", ROOT / "experiments" / "07_template_mining" / "results"))
TRUNC = 700
CHUNK = 20
READERS = ("nla", "ol42", "ol40", "ol44")


@dataclass(frozen=True)
class Readout:
    feature: str
    reader: str
    item: str
    value: str
    k: int
    text: str


def load_readouts(feature: str, reader: str) -> list[Readout]:
    """All free readouts of one reader for one feature, with the item's true value attached."""
    ol = json.loads((GEN_DIR / f"gen_ol_{feature}.json").read_text())
    values = {it["id"]: it["value"] for it in ol["items"]}
    if reader == "nla":
        gens = json.loads((GEN_DIR / f"gen_nla_{feature}.json").read_text())["nla_gens"]
    elif reader.startswith("ol"):
        gens = ol["ol_gens"][reader[2:]]
    else:
        raise ValueError(reader)
    return [Readout(feature, reader, item, values[item], k, t)
            for item, samples in gens.items() if item in values
            for k, t in enumerate(samples)]


PROMPT = """You are judging interpretability readouts. Each readout was produced by a reader that
saw only one activation vector from a language model while it processed a text; the reader
never saw the text. We want to know whether the readout FINDS a specific property.

Property: {desc}
Values: A = "{a}", B = "{b}".

For each numbered readout answer three things.
1. stated: true if the readout makes an explicit claim about this property (any wording;
   e.g. "near the end of a line", "the text is in French", "deeply nested code",
   "sentence just started"). Merely quoting or continuing text that happens to have the
   property is NOT a statement. Claims about unrelated properties are not statements.
2. stated_value: if stated, which value the claim implies: "A", "B", or "unclear" if the claim
   names the property without committing to a value. If not stated, "none".
3. exhibited_value: does the readout's OWN text exhibit value A or B of this property (for
   example it is itself written in French, in all capitals, in the past tense, continues a
   Python snippet, or begins with a noun of that gender/number)? "A", "B", or "neither".
   Only use this for properties a short text can visibly exhibit; otherwise "neither".

Readouts (newlines shown as " / "):
{lines}

Return JSON: {{"answers": [{{"i": <int>, "stated": <bool>, "stated_value": "A"|"B"|"unclear"|"none",
"exhibited_value": "A"|"B"|"neither"}}, ...]}} with one entry per readout, in order."""

SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
            "stated": {"type": "boolean"},
            "stated_value": {"type": "string", "enum": ["A", "B", "unclear", "none"]},
            "exhibited_value": {"type": "string", "enum": ["A", "B", "neither"]},
        },
        "required": ["i", "stated", "stated_value", "exhibited_value"],
        "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}


def build_prompt(desc: str, values: list[str], chunk: list[Readout]) -> str:
    lines = "\n".join(f"{i}. {r.text[:TRUNC].replace(chr(10), ' / ')}" for i, r in enumerate(chunk))
    return PROMPT.format(desc=desc, a=values[0], b=values[1], lines=lines)


def parse_answers(raw: str | None, n: int) -> dict[int, dict]:
    """Index -> answer dict for well-formed entries with 0 <= i < n. Malformed input -> {}."""
    if not raw:
        return {}
    try:
        answers = json.loads(raw).get("answers", [])
    except (json.JSONDecodeError, AttributeError):
        return {}
    out: dict[int, dict] = {}
    for a in answers:
        if not isinstance(a, dict) or not isinstance(a.get("i"), int) or not 0 <= a["i"] < n:
            continue
        if a.get("stated_value") not in ("A", "B", "unclear", "none"):
            continue
        if a.get("exhibited_value") not in ("A", "B", "neither"):
            continue
        out[a["i"]] = {"stated": bool(a.get("stated")), "stated_value": a["stated_value"],
                       "exhibited_value": a["exhibited_value"]}
    return out


def chunks(rows: list[Readout], size: int = CHUNK) -> list[list[Readout]]:
    return [rows[s: s + size] for s in range(0, len(rows), size)]


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def aggregate(judged: list[dict], values: list[str]) -> dict:
    """Per-reader recall statistics from judged rows.

    Each row: {item, value, stated, stated_value, exhibited_value} where *_value are "A"/"B"/...
    Returns sample-level rates and item-level "any of k" rates.
    """
    code = {values[0]: "A", values[1]: "B"}
    n = len(judged)
    stated = [r for r in judged if r["stated"]]
    committed = [r for r in stated if r["stated_value"] in ("A", "B")]
    correct = [r for r in committed if r["stated_value"] == code[r["value"]]]
    exhibited = [r for r in judged if r["exhibited_value"] in ("A", "B")]
    ex_correct = [r for r in exhibited if r["exhibited_value"] == code[r["value"]]]
    by_item: dict[str, dict[str, bool]] = defaultdict(lambda: {"stated": False, "correct": False})
    for r in judged:
        by_item[r["item"]]  # every judged item enters the denominator
        if r["stated"]:
            by_item[r["item"]]["stated"] = True
        if r["stated_value"] == code[r["value"]]:
            by_item[r["item"]]["correct"] = True
    items = list(by_item.values())
    return {
        "n_samples": n,
        "n_items": len(items),
        "stated_rate": _rate(len(stated), n),
        "stated_correct_rate": _rate(len(correct), n),
        "stated_wrong_rate": _rate(len(committed) - len(correct), n),
        "value_acc_given_committed": _rate(len(correct), len(committed)),
        "any_stated_per_item": _rate(sum(i["stated"] for i in items), len(items)),
        "any_correct_per_item": _rate(sum(i["correct"] for i in items), len(items)),
        "exhibited_rate": _rate(len(exhibited), n),
        "exhibited_correct_rate": _rate(len(ex_correct), n),
        "exhibited_acc_given_exhibited": _rate(len(ex_correct), len(exhibited)),
    }
