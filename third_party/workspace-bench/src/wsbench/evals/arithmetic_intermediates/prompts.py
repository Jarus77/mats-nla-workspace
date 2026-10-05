"""Prompts of the arithmetic-intermediates free-recall judge: the chained-intermediates judge
(source repo ``judge_free_modal.py``) with its task sentence changed to a bare arithmetic
expression; printed verbatim in the family README."""

import json

from wsbench.llm import schema_block

PROMPT_VERSION = "arith-free-2026-09-23"  # cells=frozen: one call per cell
# cells=all: one call per (item, layer) over every position of that layer (Camila, 2026-09-23 06:14:
# "for user modeling and arithmetic let's just batch it - feed it everything at once!"). Same
# instructions as PROMPT_VERSION, different packaging, so its own version tag.
PROMPT_VERSION_BATCH = "arith-free-batched-2026-09-23"
MAX_VALUES = 3  # the prompt caps the ranked list at three; the schema cannot (no maxItems)

SYSTEM = """You are reading the output of a "lens" that tries to verbalize what a language model is
computing internally. The model is silently evaluating a short arithmetic expression: two or
three operations on given numbers, answered with no chain of thought. You do NOT get to see the
expression — only the lens output.

The lens output may be free text, bullet points, or a bag of loose tokens, possibly in several
languages (Chinese numerals like 十四 mean fourteen). It is often vague, generic, or about
something else entirely, and it frequently contains numbers that are not results of anything.

Your job: name the number(s) that this output presents as a COMPUTED VALUE — a result, an
intermediate, a total, a "the answer is X". Rank them, most clearly a computed result first,
at most three. Decimals and negative numbers are values too; report them exactly as written.

Rules:
- Do NOT list numbers that appear as operands, quantities, labels, list indices, years, counts
  of items, or parts of unrelated prose. Only numbers the output presents as something that was
  worked out.
- If the output states arithmetic with both operands and a result ("16 - 2 = 14"), list the
  RESULT (14), not the operands.
- A bag of loose numerals with no arithmetic around them still counts: if the bag is dominated
  by one value or a tight cluster, list what it points at, most frequent or most prominent first.
- If the output presents no number as a computed value, set states_value false and return an
  empty list. That is a normal and common answer — do not invent one.
- Never do any arithmetic of your own. You do not know the task, so you cannot know what the
  right answer is; only report what the text itself puts forward."""

USER = (
    "Lens output:\n<<<\n{readout}\n>>>\n\n"
    "Which number(s) does this output present as a computed value?"
)

SYSTEM_BATCH = """\
You are reading the outputs of a "lens" that tries to verbalize what a language model is
computing internally. The model is silently evaluating a short arithmetic expression: two or
three operations on given numbers, answered with no chain of thought. You do NOT get to see the
expression — only the lens outputs.

You are given several lens outputs at once, one per position the lens was read at, as numbered
entries "[k] pos=<offset from the end of the prompt> token=<the token at that position>:" followed
by that entry's output (a bag of loose tokens is written as "tokens: ..."). Judge every entry on
its own; the entries do not explain each other, and an entry's header is not part of its output.

A lens output may be free text, bullet points, or a bag of loose tokens, possibly in several
languages (Chinese numerals like 十四 mean fourteen). It is often vague, generic, or about
something else entirely, and it frequently contains numbers that are not results of anything.

Your job, for EACH entry: name the number(s) that its output presents as a COMPUTED VALUE — a
result, an intermediate, a total, a "the answer is X". Rank them, most clearly a computed result
first, at most three. Decimals and negative numbers are values too; report them exactly as written.

Rules, applied to every entry on its own:
- Do NOT list numbers that appear as operands, quantities, labels, list indices, years, counts
  of items, or parts of unrelated prose. Only numbers the output presents as something that was
  worked out.
- If the output states arithmetic with both operands and a result ("16 - 2 = 14"), list the
  RESULT (14), not the operands.
- A bag of loose numerals with no arithmetic around them still counts: if the bag is dominated
  by one value or a tight cluster, list what it points at, most frequent or most prominent first.
- If an output presents no number as a computed value, return an empty list for that entry. That
  is a normal and common answer — do not invent one.
- Never do any arithmetic of your own. You do not know the task, so you cannot know what the
  right answer is; only report what each text itself puts forward.
- Answer with exactly one result per entry, carrying the entry's k, in the order given; the
  quote is the shortest verbatim span of THAT entry supporting its top value (empty if none)."""

USER_BATCH = (
    "Lens outputs, one per read position:\n\n{entries}\n\n"
    "For each entry, which number(s) does its output present as a computed value?"
)

PROMPTS: dict[str, str] = {
    "SYSTEM": SYSTEM,
    "USER": USER,
    "SYSTEM_BATCH": SYSTEM_BATCH,
    "USER_BATCH": USER_BATCH,
}

SCHEMA = schema_block(
    "arith_free",
    {
        "states_value": {"type": "boolean"},
        "values": {
            "type": "array",
            "items": {"type": "number"},
            "description": "ranked, most clearly a computed result first, at most three",
        },
        "basis": {"type": "string", "enum": ["arithmetic", "stated_result", "numeral_bag", "none"]},
        "quote": {
            "type": "string",
            "description": "shortest verbatim span supporting the top value",
        },
    },
    ["states_value", "values", "basis", "quote"],
)


_VALUES = {
    "type": "array",
    "items": {"type": "number"},
    "description": "ranked, most clearly a computed result first, at most three",
}
_BASIS = {"type": "string", "enum": ["arithmetic", "stated_result", "numeral_bag", "none"]}

# one object per entry: the per-cell answer minus states_value (an empty list says it), plus k
SCHEMA_BATCH = schema_block(
    "arith_free_batch",
    {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["k", "values", "basis", "quote"],
                "properties": {
                    "k": {"type": "integer", "description": "the entry number [k]"},
                    "values": _VALUES,
                    "basis": _BASIS,
                    "quote": {
                        "type": "string",
                        "description": (
                            "shortest verbatim span of the entry supporting its top value"
                        ),
                    },
                },
            },
        }
    },
    ["entries"],
)


def render_user(readout: str) -> str:
    return USER.replace("{readout}", readout.strip())


def render_entry(k: int, pos: int, token: str | None, text: str, *, bag: bool) -> str:
    """``[k] pos=<offset> token=<token>: <readout>``; a token lens's scored bag is prefixed
    ``tokens:``. The token is JSON-quoted so newlines and specials stay legible."""
    head = f"[{k}] pos={pos} token={json.dumps(token or '', ensure_ascii=False)}:"
    body = text.strip()
    return f"{head} tokens: {body}" if bag else f"{head} {body}"


def render_batch_user(entries: list[tuple[int, int, str | None, str]], *, bag: bool) -> str:
    """The cells=all user message: every non-empty position of one (item, layer), in position
    order, as numbered entries separated by blank lines."""
    block = "\n\n".join(render_entry(k, pos, tok, text, bag=bag) for k, pos, tok, text in entries)
    return USER_BATCH.replace("{entries}", block)
