"""Verbatim prompts of the hallucination judge (``v5c-chat``).

Source: ``hallucination-bench/src/hallucination_bench/judge.py`` (L84-143 literals, L169-200
schema, L210-228 rendering) and ``sites.py`` (``READ_MARK``, ``render_marked_response``). The
literals are also frozen in ``evals/hallucination/judge_prompt_v5c_chat.json``, loaded here at
import as :data:`FROZEN`; ``tests/test_hallucination.py`` asserts the two agree.

Unlike the MC families this family renders ``JUDGE_USER`` with ``str.format`` exactly as the
source does: the template carries ``{{text, type, why}}`` escapes, and the README-equality test
compares the escaped literal.

The second stage (``VERIFY_*``, ``v5c-chat-verify-v1``) is the source repo's
``scripts/oracle_lens_evals/hallucination/chat_verify.py`` (2026-09-16) copied verbatim: given
the span judge's verified wrong spans as established-false, it lists every OTHER specific claim
of a readout as true / unverifiable / disputed / off_topic. Same ``str.format`` rendering.
"""

import json
from collections.abc import Sequence
from typing import Any

from wsbench import registry
from wsbench.llm import schema_block

PROMPT_VERSION = "v5c-chat"
READ_MARK = "⟦READ⟧"
FROZEN_PATH = registry.REPO_ROOT / "evals" / "hallucination" / "judge_prompt_v5c_chat.json"
FROZEN: dict[str, Any] = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

JUDGE_SYSTEM = (
    "You audit an interpretability lens that reads a language model's internal activation "
    "while the model writes a response. You are shown the model's full response and the "
    "lens's raw readouts of one activation inside it. Your job is to point at the parts of "
    "a readout that are WRONG about this response, and separately at the parts that are "
    "JUNK. You quote the readout's own words and never paraphrase. Output only the JSON "
    "verdict."
)

JUDGE_USER = (
    "A language model was asked:\n\n<prompt>\n{prompt}\n</prompt>\n\nand wrote the "
    "response below. The lens read the model's activation at the token immediately before "
    "the marker {mark}; everything before it is what the model had already written, "
    "everything after it is what it went on to write. The WHOLE response is the context — "
    "a readout may legitimately refer to any part of it.\n\n<response>\n{marked_response}\n"
    "</response>\n{facts_block}\nBelow are {k} independent readouts of that one "
    'activation.\n\n{samples_block}\n\nFor EACH readout, report:\n\n- kind: "specific" if '
    'it says anything concrete; "generic" if it only says generic things ("a narrative '
    'about a character", "an account of a historical event"); "empty" for noise, '
    "fragments, formatting, a bare restatement of the prompt, or a garbled string that "
    "asserts nothing.\n- spans: the parts of THIS readout that are WRONG or OFF-TOPIC, "
    "each copied character-for-character from the readout (never your own words, never a "
    'span from the response), each with a type:\n\n  type "wrong" — the span is on topic '
    "(about THIS subject, THIS response, or what it says next) and the response or the "
    "facts show it is wrong:\n    - it contradicts the response or the facts: a wrong "
    "date, number or quantity; a wrong name, place or label; the wrong person doing "
    'something; a wrong cause, outcome or detail ("October 24" where the response says '
    'October 29; "over 700 died" where it says over 140; Johnson where it says Kennedy; '
    "uracil among the bases of DNA);\n    - it quotes or names something in the response "
    "and gets the words wrong: a heading, section title, list item or quotation the "
    'response does not have ("the \'Quick Facts\' section" where the heading is "Basic '
    'Facts"). Reordering or reformatting the same words is NOT wrong;\n    - it misstates '
    "what the response has ALREADY written before the marker: what a passage says, how the "
    "response is structured so far, or what it has already claimed;\n    - it states "
    "something about this subject that conflicts with the response (an event, outcome or "
    'detail the response rules out).\n\n  type "off_topic" — the span is JUNK: garbled or '
    "broken tokens, stray code or programming terms, or content with no connection at all "
    "to this conversation (a Mars probe in an answer about English idioms; TimeSpan values "
    "in a recipe). This is uncommon. Coherent content that connects to the conversation is "
    "NOT off_topic, even when it brings up a new topic, a side example or a different "
    "scenario (a hostile manager or a breakup in an answer about kindness to family): if "
    "it is a reasonable direction for the response, leave it unmarked; if it says "
    'something about this conversation that the text rules out, it is "wrong". Content '
    "from this subject or a closely related one is never off_topic. Off-topic spans are "
    "reported but do not count.\n\nDo NOT mark (neither type):\n  - a prediction of what "
    "comes next that is a reasonable continuation of the text before the marker, even when "
    "the response actually went a different way. The response after the marker is ONE "
    "sample of what the model could have written, not the only right answer: a readout "
    'that expects the model to call its "better half" its codebase, where the response '
    "went on to say the engineers who built it, is consistent. Judge a prediction against "
    'the text BEFORE the marker, and mark it "wrong" only when that text rules it out;\n  '
    "- anything the response simply does not mention: absence is not wrongness. Mark "
    '"wrong" only when the response or the facts rule the span out;\n  - content that is '
    "right, compatible, or a reasonable inference; a generic statement;\n  - hedging, "
    "repetition, formatting, or a fragment that breaks off.\nWhen a span could reasonably "
    "be right, leave it.\n\nReport, as JSON: samples = one entry per readout in order, "
    "each with idx, kind, spans (list of {{text, type, why}} where `text` is the verbatim "
    'readout span, `type` is "wrong" or "off_topic", and `why` is one short clause), and a '
    "one-sentence rationale."
)

JUDGE_SCHEMA: dict[str, Any] = schema_block(
    "hallucination_ontopic_spans",
    {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["idx", "kind", "spans", "rationale"],
                "properties": {
                    "idx": {"type": "integer"},
                    "kind": {"type": "string"},
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "type", "why"],
                            "properties": {
                                "text": {"type": "string"},
                                "type": {"type": "string", "enum": ["wrong", "off_topic"]},
                                "why": {"type": "string"},
                            },
                        },
                    },
                    "rationale": {"type": "string"},
                },
            },
        }
    },
    ["samples"],
)

# ------------------------------------------------------------------ stage 2: claim verification

VERIFY_PROMPT_VERSION = "v5c-chat-verify-v1"
VERIFY_STATUSES: tuple[str, ...] = ("true", "unverifiable", "disputed", "off_topic")

VERIFY_SYSTEM = (
    "You audit an interpretability lens that reads a language model's internal activation while "
    "the model writes a response. You are shown the model's full response and the lens's raw "
    "readouts of one activation inside it. Some spans of each readout have ALREADY been "
    "established as false by a separate audit; those are given to you and are not yours to "
    "re-judge. Your job is to list every OTHER specific claim each readout makes about this "
    "conversation and say whether the response settles it as true, or cannot settle it. You "
    "quote the readout's own words for every claim and never paraphrase. Output only the JSON "
    "verdict."
)

VERIFY_USER = (
    "A language model was asked:\n\n<prompt>\n{prompt}\n</prompt>\n\n"
    "and wrote the response below. The lens read the model's activation at the token immediately "
    "before the marker {mark}; everything before it is what the model had already written, "
    "everything after it is what it went on to write. The WHOLE response is the context — a "
    "readout may legitimately refer to any part of it.\n\n"
    "<response>\n{marked_response}\n</response>\n\n"
    "Below are {k} independent readouts of that one activation. Under each readout is the list "
    "of its spans already established as FALSE (possibly empty).\n\n{samples_block}\n\n"
    "For EACH readout, report:\n\n"
    '- kind: "specific" if it says anything concrete; "generic" if it only says generic things '
    '("a narrative about a character", "an account of a historical event"); "empty" for noise, '
    "fragments, formatting, a bare restatement of the prompt, or a garbled string that asserts "
    "nothing.\n"
    "- claims: EVERY separate specific claim the readout makes about this conversation that is "
    "NOT one of its already-established false spans. A claim is a specific assertion the "
    "readout commits to — a fact, entity, quantity, name, place, quotation, heading, structure, "
    "intent, or a prediction of what the response says next. Generic phrases, task "
    "descriptions, hedging, repetition, formatting and broken fragments are NOT claims; a "
    "readout may have zero claims. Do not list the established-false spans again. For each:\n\n"
    "  - quote: the readout's own words for this claim, copied character-for-character from THIS "
    "readout (never your own words, never a span of the response). Quote the shortest span that "
    "carries the claim.\n"
    "  - status, exactly one of:\n"
    '    "true" — the response states this claim or clearly entails it. A prediction of what '
    "comes next counts as true when the response after the marker does say it.\n"
    '    "unverifiable" — the response neither states it nor rules it out: an unstated detail, a '
    "plausible inference, a claim about what the model is thinking or intending, or a "
    "prediction the response does not go on to make but that nothing in the text excludes. "
    "Absence is not confirmation: if the response simply does not mention it, it is "
    "unverifiable, not true.\n"
    '    "disputed" — you believe the response actually RULES THIS OUT, but it was not in the '
    "established-false list. Use this rather than inventing a false verdict; it is reported "
    "separately.\n"
    '    "off_topic" — junk: garbled or broken tokens, stray code or programming terms, or '
    "content with no connection at all to this conversation.\n"
    "  - why: one short clause.\n\n"
    "Rules:\n"
    "  - Use ONLY this conversation to settle a claim, never world knowledge. A claim that is "
    "true in the world but unmentioned here is unverifiable.\n"
    '  - "true" requires the response to actually say or entail it. Compatible is not the same '
    "as stated: a claim that merely fits the response is unverifiable.\n\n"
    "Report, as JSON: samples = one entry per readout in order, each with idx, kind, claims "
    "(list of {{quote, status, why}}), and a one-sentence rationale."
)

VERIFY_SCHEMA: dict[str, Any] = schema_block(
    "hallucination_chat_verify",
    {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["idx", "kind", "claims", "rationale"],
                "properties": {
                    "idx": {"type": "integer"},
                    "kind": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["quote", "status", "why"],
                            "properties": {
                                "quote": {"type": "string"},
                                "status": {"type": "string", "enum": list(VERIFY_STATUSES)},
                                "why": {"type": "string"},
                            },
                        },
                    },
                    "rationale": {"type": "string"},
                },
            },
        }
    },
    ["samples"],
)

PROMPTS: dict[str, str] = {
    "JUDGE_SYSTEM": JUDGE_SYSTEM,
    "JUDGE_USER": JUDGE_USER,
    "VERIFY_SYSTEM": VERIFY_SYSTEM,
    "VERIFY_USER": VERIFY_USER,
}


def render_marked_response(response: str, char: int) -> str:
    """The response with :data:`READ_MARK` inserted at ``char`` — the frozen offset just after
    the read token (``sites[].char`` in the bank)."""
    char = max(0, min(int(char), len(response)))
    return response[:char] + READ_MARK + response[char:]


def samples_block(samples: Sequence[str]) -> str:
    return "\n\n".join(f'<readout idx="{i}">\n{s}\n</readout>' for i, s in enumerate(samples))


def judge_prompt(
    item: dict[str, Any], site: dict[str, Any], samples: Sequence[str]
) -> tuple[str, str]:
    """(system, user) for one (item, layer, site) cell. No facts block: chat items have none."""
    return (
        JUDGE_SYSTEM,
        JUDGE_USER.format(
            prompt=item["prompt"],
            mark=READ_MARK,
            marked_response=render_marked_response(item["response"], int(site["char"])),
            facts_block="",
            k=len(samples),
            samples_block=samples_block(samples),
        ),
    )


def verify_samples_block(samples: Sequence[str], wrong: Sequence[Sequence[str]]) -> str:
    """Each readout block followed by its established-false spans (one ``json.dumps`` line each,
    or ``(none)``), exactly as the source pass rendered them."""
    parts = []
    for i, (s, w) in enumerate(zip(samples, wrong, strict=True)):
        falses = "\n".join(f"  - {json.dumps(x, ensure_ascii=False)}" for x in w) or "  (none)"
        parts.append(
            f'<readout idx="{i}">\n{s}\n</readout>\n'
            f"Established FALSE spans of readout {i}:\n{falses}"
        )
    return "\n\n".join(parts)


def verify_prompt(
    item: dict[str, Any],
    site: dict[str, Any],
    samples: Sequence[str],
    wrong: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """(system, user) for the claim-verification call of one cell; ``wrong[i]`` = the span
    judge's verified wrong spans of readout ``i`` (given to the judge as established false)."""
    return (
        VERIFY_SYSTEM,
        VERIFY_USER.format(
            prompt=item["prompt"],
            mark=READ_MARK,
            marked_response=render_marked_response(item["response"], int(site["char"])),
            k=len(samples),
            samples_block=verify_samples_block(samples, wrong),
        ),
    )
