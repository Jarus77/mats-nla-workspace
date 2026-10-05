"""Prompts of the multi-concept directed-modulation judge; every entry of ``PROMPTS`` is printed
verbatim in the family README."""

from wsbench.llm import schema_block

PROMPT_VERSION = "mcdm-2026-09-16"
SEED = 20260916
N_OPTIONS = 6
LETTERS = "ABCDEF"

SYSTEM = (
    "You judge the READOUT of an interpretability lens: text decoded from a language model's "
    "internal activations while it was writing a sentence you cannot see. Before writing, the "
    "model may have been told to hold one or more unrelated concepts in mind. You are given the "
    "readout and a list of candidate concepts. Select EVERY candidate the readout clearly names: "
    "the same thing, in any language or wording. A paraphrase that unmistakably denotes one "
    "candidate counts; a single shared word that fits several candidates does not. Where two "
    "candidates differ only in who does what to whom, select one only if the readout states that "
    "direction. Select nothing when the readout names none of them, or only lists possibilities "
    "without committing. Do not use your own knowledge to guess what the model was probably "
    "told. For each selection give a SHORT VERBATIM quote (a few words copied exactly from the "
    "readout) that names it."
)
QUESTION = (
    "Which of these candidates does the readout name? Select every one that applies, or none."
)
USER = (
    'READOUT:\n"""\n{readout}\n"""\n\n{question}\n\n{options}\n\n'
    "Answer with the letters of your selections and a quote for each; an empty list means none."
)

PROMPTS: dict[str, str] = {"SYSTEM": SYSTEM, "QUESTION": QUESTION, "USER": USER}

SCHEMA = schema_block(
    "picks",
    {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["choice", "quote"],
                "properties": {"choice": {"type": "string"}, "quote": {"type": "string"}},
            },
        }
    },
    ["picks"],
)


def render_user(readout: str, options: list[str]) -> str:
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (
        USER.replace("{readout}", readout.strip())
        .replace("{question}", QUESTION)
        .replace("{options}", lines)
    )
