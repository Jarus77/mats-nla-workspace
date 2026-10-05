"""Verbatim prompts of the role_bound_association judge.

Source: ``scripts/oracle_lens_evals/oa_eb_readout_judge.py`` in the source repo (L45-46,
L60-67, L72-83 with ``with_foils=False``, L93-99, L121-131, L258). The entity-binding branch is
not ported. Templates use ``{name}`` placeholders rendered with ``str.replace`` — the header
contains literal ``{person A} -> {action} -> {person B}`` which must survive. Every entry of
:data:`PROMPTS` is printed verbatim in ``evals/role_bound_association/README.md``.
"""

from collections.abc import Sequence

from wsbench.llm import schema_block
from wsbench.mc import listing

DATASET_SEED = 20260810
PROMPT_VERSION = "oa-v1"

SYSTEM = (
    "You judge the READOUT of an interpretability lens — noisy text (or a token list) "
    "decoded from a language model's internal activations. Answer the multiple-choice "
    "questions based ONLY on what the readout states or clearly implies. Do not use world "
    "knowledge or plausibility to fill gaps: if the readout does not support an answer, "
    "choose the 'cannot tell' option. Readouts may be garbled; partial name matches "
    "('Marc', 'Bangk') count as mentions of the full name."
)

SCHEMA = schema_block(
    "readout_mc",
    {
        "q1_choice": {"type": "integer"},
        "q2_choice": {"type": "integer"},
        "q3_choice": {"type": "integer"},
        "evidence": {"type": "string"},
    },
    ["q1_choice", "q2_choice", "q3_choice", "evidence"],
)

Q1 = "Q1. Who performs the action (the agent)?"
Q2 = "Q2. What is the action or state?"
Q3 = "Q3. Who is the action directed at (the patient)?"
HEADER = (
    "The readout may describe two people; the sentence it implies has the form "
    "{person A} -> {action} -> {person B}. People may be referred to by name or "
    "description."
)
MC_BLOCK = "{question}\n{listing}"
BLOCK = "{header}\n\n{q1}\n\n{q2}\n\n{q3}"
USER = "READOUT:\n{readout}\n\n{block}"

PROMPTS: dict[str, str] = {
    "SYSTEM": SYSTEM,
    "Q1": Q1,
    "Q2": Q2,
    "Q3": Q3,
    "HEADER": HEADER,
    "MC_BLOCK": MC_BLOCK,
    "BLOCK": BLOCK,
    "USER": USER,
}


def render_mc_block(question: str, shown: Sequence[str]) -> str:
    return MC_BLOCK.replace("{question}", question).replace("{listing}", listing(shown))


def render_block(q1: str, q2: str, q3: str) -> str:
    return (
        BLOCK.replace("{header}", HEADER)
        .replace("{q1}", q1)
        .replace("{q2}", q2)
        .replace("{q3}", q3)
    )


def render_user(readout: str, block: str) -> str:
    return USER.replace("{readout}", readout).replace("{block}", block)
