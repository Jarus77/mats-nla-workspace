"""The bank judge prompt shared by the single-token basic families; printed verbatim in each
family README."""

import json

from wsbench.llm import schema_block
from wsbench.readouts import Cell, Kind

PROMPT_VERSION = "bank-2026-09-16"

SYSTEM = """You are judging readouts from an activation lens for an interpretability
benchmark. You get one eval item: its TARGET concept(s) and the lens's readout samples at one
layer (several token positions; for the J-lens the samples are top-10 vocabulary tokens, for
the oracle lens they are free-text generations).

Decide whether ANY target concept is EXPRESSED in the readout. STRICT rules:
- Expressed means the target concept is NAMED: the exact target word/phrase as a standalone
  word (case/inflection of the SAME word is fine: "cats" for "cat"), an established synonym or
  alias that unambiguously denotes the SAME concept ("Mandarin" for "Chinese"), or a faithful
  translation of the target into another language ("二月" for "February").
- NOT expressed by a DIFFERENT or DERIVED word that merely contains the target string:
  "blackout" does NOT express "black"; "childhood" does NOT express "child".
- NOT expressed by word fragments or partial tokens, even distinctive ones: "paragus" does NOT
  express "asparagus"; " acet" does NOT express "acetaminophen". Fragments of different tokens
  sitting near each other in a token list NEVER combine.
- NOT expressed by thematically associated imagery, descriptions, or scene content that evokes
  the concept without naming it: "the wedding dress that never left the closet" does NOT
  express "sadness". Hypernyms, hyponyms, and category neighbors do NOT count.
- Judge only what is actually written. If uncertain, say not expressed.
- Every YES must include a VERBATIM quote copied exactly from one sample (the smallest span
  that names the target). No quote, no pass.
Return JSON: {"expressed": bool, "target": "<which target, or empty>", "quote": "<verbatim>"}."""

KIND_NOTE_PROSE = "The samples below are free-text readouts."
KIND_NOTE_TOKENS = (
    "The samples below are the lens's top VOCABULARY TOKENS, best first (one per line)."
)
USER = "targets: {targets}\n{kind_note}\n\n{blocks}"
BLOCK = "[position {pos}]\n{samples}"

PROMPTS: dict[str, str] = {
    "SYSTEM": SYSTEM,
    "KIND_NOTE_PROSE": KIND_NOTE_PROSE,
    "KIND_NOTE_TOKENS": KIND_NOTE_TOKENS,
    "USER": USER,
    "BLOCK": BLOCK,
}

SCHEMA = schema_block(
    "bank_verdict",
    {
        "expressed": {"type": "boolean"},
        "target": {"type": "string"},
        "quote": {"type": "string"},
    },
    ["expressed", "target", "quote"],
)


def cell_samples(cell: Cell) -> tuple[str, ...]:
    return cell.samples if cell.samples is not None else (cell.tokens or ())


def render_user(targets: list[str], kind: Kind, cells: list[Cell]) -> str:
    """One item at one layer: the targets line, the kind note, then every position's samples."""
    blocks = []
    for c in sorted(cells, key=lambda c: c.pos):
        samples = "\n".join(f"  {s}" for s in cell_samples(c))
        blocks.append(BLOCK.replace("{pos}", str(c.pos)).replace("{samples}", samples))
    note = KIND_NOTE_TOKENS if kind == "tokens" else KIND_NOTE_PROSE
    return (
        USER.replace("{targets}", json.dumps(targets))
        .replace("{kind_note}", note)
        .replace("{blocks}", "\n\n".join(blocks))
    )
