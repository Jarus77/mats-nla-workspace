"""Verbatim prompts of the jlens_concept_pr judge (``jlens-pr-v2``: prompts unchanged from
the source; v2 marks Stage A moving from DeepSeek-V4-Flash to Gemini 3.8 Flash).

Source: ``scripts/oracle_lens_evals/jlens_pr/judge_prompts.py`` in the source repo — the
system prompts (L17-30, L32-47, L182-209), schemas (L51-73, L211-228), ``GRADE_VALUE`` (L75),
``STAGE_P_CHUNK`` (L229-232), the user bodies of ``stage_a_params`` / ``stage_b_params`` /
``stage_p_params`` (L110-120, L123-130, L246-251) and the parsers (L137-171, L254-272).
``STAGE_P_SYSTEM``'s worked-example guide is calibrated against the hand-label audit: copied
byte-for-byte, never reflow it. Templates carry ``{name}`` placeholders substituted in one pass
(never ``str.format``). Every entry of :data:`PROMPTS` is printed verbatim in
``evals/jlens_concept_pr/README.md``.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from wsbench.llm import schema_block

PROMPT_VERSION = "jlens-pr-v2"

STAGE_A_SYSTEM = """You are given text produced by an interpretability lens that verbalizes \
a language model's internal state. Break the text into its UNIQUE concepts: short noun \
phrases naming each distinct thing, property, action or entity the text asserts. Merge \
duplicates and near-duplicates (inflections, restatements) into one concept. Ignore list \
bullets, markdown emphasis and boilerplate connectives, but NOT evaluative or affirmative \
markers: a checkmark or emoji (✅, ✔, 🎉), "Correct Answer", "Yes", "Final Answer" each name a \
concept ("correct / confirmed answer", "celebration") and must be listed when present. Do not \
add concepts the text does not contain.

Example. Text: "limited protein intake, potentially hindering wound healing. An elderly widow \
with inadequ" -> concepts: ["inadequate/insufficient amount", "protein intake", \
"wound healing", "elderly widow / older adult"].

Return JSON: {"concepts": [string, ...]}."""

STAGE_B_SYSTEM = """You grade whether ONE token read out of a language model's internal \
activations is supported by each of a list of concepts extracted from a lens's text. Grade \
every concept exactly once:

- "in": the concept has the token, modulo word inflection / casing / translation \
(e.g. tokens "accessing", "访问" match the concept "access pattern"). Emoji are words: an \
emoji token matches a concept that names the same thing in words ("✅" matches "correct \
answer"; "🎉" matches "celebration"), and emoji with the same meaning are the same token \
("🎉" and "🥳" both match "celebration"; "✅", "✔" and "✓" all match "confirmed / correct").
- "partial": the concept has a close relative of the token (e.g. token "lacks" matches \
"inadequate intake"; token "EXIT" matches "return 0"; token "😊" is partial for "gratitude").
- "out": everything else, including words the concept does not contain (e.g. tokens \
"SUCCESS", "printf" do not match "return 0;").

Return JSON: {"grades": [{"concept": <exact concept string>, "grade": "in"|"partial"|"out"}, \
...]} with one entry per listed concept, using the concept strings verbatim."""

# ``schema_block`` returns the Chat-Completions wrapper ``{name, strict, schema}`` (the same shape
# as the source's ``llm_client.schema_block``).
STAGE_A_SCHEMA = schema_block(
    "concepts",
    {"concepts": {"type": "array", "items": {"type": "string"}}},
    ["concepts"],
)
STAGE_B_SCHEMA = schema_block(
    "grades",
    {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "grade": {"type": "string", "enum": ["in", "partial", "out"]},
                },
                "required": ["concept", "grade"],
                "additionalProperties": False,
            },
        }
    },
    ["grades"],
)

GRADE_VALUE = {"in": 1.0, "partial": 0.5, "out": 0.0}

STAGE_P_SYSTEM = """You are given the SET of tokens read out of a language model's internal \
activations at one position (a lens's top-k readout, content tokens only), and a list of concepts \
extracted from another lens's text about that same position.

Stage B asks, of ONE token, which concepts have it. This is the same question asked the other way \
round: for each concept, grade the BEST correspondence between that concept and ANY token in the \
set. Grade every concept exactly once, on the same three-point scale:

- "in": the concept has one of the tokens, modulo word inflection / casing / translation \
(tokens "accessing", "访问" are in for the concept "access pattern"). A concept that says more \
than the token still counts as in — the token has to be in the concept, not equal to it: \
"working at Walt Disney" is in for the token "Disney", "no worry about overlap" is in for \
"overlap", "credit card approval rates" is in for "credit" or "rates", "objective lens" is in for \
"lens". Emoji are words: an emoji token matches a concept that names the same thing in words \
("✅" is in for "correct answer"; "🎉" for "celebration"), and emoji with the same meaning are the \
same token ("🎉" and "🥳" both match "celebration"; "✅", "✔" and "✓" all match "confirmed / \
correct").
- "partial": the concept has a close relative of one of the tokens — the token names the \
concept's category, a sibling, or a part of it (token "lacks" is partial for "inadequate intake"; \
"technologies" for "filtration technologies"; "police, arrest" for "traffic stop"; "people" for \
"Anji people"; "Alumni" for "Yale University"; "loan, funding" for "SBA approval requirement"; \
"MRI, MR" for "MR-ME"; "plants, herbs" for "flowers"; "😊" for "gratitude").
- "out": everything else — no token is in the concept and none is a close relative of it \
(tokens "Azure, Haskell, PowerShell" are out for "lambda expression (fun x -> ...)"; "Supplier" \
is out for "based in Victoria, Australia"; "はありません" is out for "final token").

Return JSON: {"grades": [{"concept": <exact concept string>, "grade": "in"|"partial"|"out"}, \
...]} with one entry per listed concept, using the concept strings verbatim."""

STAGE_P_SCHEMA = schema_block(
    "support",
    {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "grade": {"type": "string", "enum": ["in", "partial", "out"]},
                },
                "required": ["concept", "grade"],
                "additionalProperties": False,
            },
        }
    },
    ["grades"],
)
# One Stage P call grades at most this many concepts. Cells reach 279 concepts, which would both
# risk the judge's context window (every concept is echoed back verbatim) and dilute the grading;
# chunking keeps each call short. The cell's FULL token set goes into every chunk.
STAGE_P_CHUNK = 60


# User bodies (the f-strings of ``stage_a_params`` / ``stage_b_params`` / ``stage_p_params``).
STAGE_A_USER = "Text:\n{text}"
STAGE_B_USER = "Token: {token_repr}\n\nConcepts:\n{listing}"
STAGE_P_USER = "Tokens: {tok_list}\n\nConcepts:\n{listing}"

PROMPTS: dict[str, str] = {
    "STAGE_A_SYSTEM": STAGE_A_SYSTEM,
    "STAGE_B_SYSTEM": STAGE_B_SYSTEM,
    "STAGE_P_SYSTEM": STAGE_P_SYSTEM,
    "STAGE_A_USER": STAGE_A_USER,
    "STAGE_B_USER": STAGE_B_USER,
    "STAGE_P_USER": STAGE_P_USER,
}

_PLACEHOLDER = re.compile(r"\{(text|token_repr|tok_list|listing)\}")


def _render(template: str, **fields: str) -> str:
    """Single-pass placeholder substitution (a substituted value is never re-scanned)."""
    return _PLACEHOLDER.sub(lambda m: fields[m.group(1)], template)


def concept_listing(concepts: Sequence[str]) -> str:
    return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(concepts))


def concat_samples(samples: Sequence[str]) -> str:
    """Join the non-empty stripped samples of a cell with a ``---`` separator line."""
    return "\n---\n".join(s.strip() for s in samples if s.strip())


def render_stage_a(text: str) -> str:
    return _render(STAGE_A_USER, text=text)


def render_stage_b(token: str, concepts: Sequence[str]) -> str:
    return _render(STAGE_B_USER, token_repr=repr(token), listing=concept_listing(concepts))


def render_stage_p(tokens: Sequence[str], concepts: Sequence[str]) -> str:
    tok_list = ", ".join(repr(t) for t in tokens)
    return _render(STAGE_P_USER, tok_list=tok_list, listing=concept_listing(concepts))


def stage_p_chunks(concepts: Sequence[str]) -> list[tuple[int, list[str]]]:
    """``(offset, slice)`` pairs covering ``concepts`` in order, each at most ``STAGE_P_CHUNK``."""
    return [
        (i, list(concepts[i : i + STAGE_P_CHUNK])) for i in range(0, len(concepts), STAGE_P_CHUNK)
    ]


# --- parsers (a ``ValueError`` = reject, never a partial score) ------------------------------


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _data(text: str | dict[str, Any]) -> dict[str, Any]:
    """The source parsers took the raw JSON text; wsbench hands them the parsed object."""
    data = json.loads(text) if isinstance(text, str) else text
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def parse_stage_a(text: str | dict[str, Any]) -> list[str]:
    """Stage A JSON -> deduped (case/whitespace-insensitive) non-empty concepts, order kept."""
    data = _data(text)
    out: list[str] = []
    seen: set[str] = set()
    concepts = data.get("concepts", [])
    if not isinstance(concepts, list):
        raise ValueError("concepts is not a list")
    for c in concepts:
        if not isinstance(c, str):
            continue
        key = _norm(c)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c.strip())
    return out


def _parse_grades(text: str | dict[str, Any], concepts: Sequence[str]) -> list[float]:
    data = _data(text)
    want = {_norm(c): i for i, c in enumerate(concepts)}
    got: dict[int, float] = {}
    grades = data.get("grades", [])
    if not isinstance(grades, list):
        raise ValueError("grades is not a list")
    for entry in grades:
        if not isinstance(entry, dict):
            raise ValueError("grade entry is not an object")
        key = _norm(str(entry.get("concept", "")))
        grade = entry.get("grade")
        if key not in want:
            raise ValueError(f"unknown concept in response: {entry.get('concept')!r}")
        if want[key] in got:
            raise ValueError(f"duplicate concept in response: {entry.get('concept')!r}")
        if grade not in GRADE_VALUE:
            raise ValueError(f"bad grade {grade!r}")
        got[want[key]] = GRADE_VALUE[grade]
    if len(got) != len(concepts):
        missing = [c for c in concepts if want[_norm(c)] not in got]
        raise ValueError(f"missing grades for {missing}")
    return [got[i] for i in range(len(concepts))]


def parse_stage_b(text: str | dict[str, Any], concepts: Sequence[str]) -> list[float]:
    """Grades in ``concepts`` order; ValueError when the judge's concept set drifted."""
    return _parse_grades(text, concepts)


def parse_stage_p(text: str | dict[str, Any], concepts: Sequence[str]) -> list[float]:
    """Support grades in ``concepts`` order; ValueError when the concept set drifted."""
    return _parse_grades(text, concepts)
