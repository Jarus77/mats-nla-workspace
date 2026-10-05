"""Prompts of the multi-token forced-choice judge (source repo ``mc_judge.py``; the language
question also credits a native-script quote, see the family READMEs). Printed verbatim there."""

from wsbench.llm import schema_block

PROMPT_VERSION = "mc-2026-09-16"
MC_SEED = 20260910
N_DISTRACTORS = 4
LETTERS = "ABCDEF"

SYSTEM = (
    "You judge the READOUT of an interpretability lens: text decoded from a language model's "
    "internal activations while it read a passage you cannot see. The readout may be noisy, "
    "verbose, in any language, and may speculate. You are given ONE multiple-choice question "
    "about what the readout NAMES. Pick an option ONLY if the readout clearly names that one "
    "option (the same thing, in any language or spelling). If the readout names NONE of the "
    "options, or names TWO OR MORE of them (a hedge, a list of candidates, 'X or Y'), choose the "
    '"cannot tell" option — a readout must commit to a single answer to be credited. Do not use '
    "your own knowledge to fill gaps or to decide which of several named candidates is right. "
    "Give a SHORT VERBATIM quote (a few words copied exactly from the readout) that supports "
    "your choice; leave it empty when you choose cannot-tell."
)
CANNOT = "cannot tell: the readout names none of these, or several of them"
QUESTION_UNIT = "Which ONE of these does the readout name (unit: {role})?"
QUESTION_LANGUAGE = (
    "In which LANGUAGE is the passage the readout describes written? Naming the language "
    "counts; so does the readout quoting the passage's words in that language's own script. "
    "English and Chinese are the readout's own working languages and are not evidence by "
    "themselves."
)
USER = 'READOUT:\n"""\n{readout}\n"""\n\n{question}\n\n{options}\n\nAnswer with the letter.'

PROMPTS: dict[str, str] = {
    "SYSTEM": SYSTEM,
    "CANNOT": CANNOT,
    "QUESTION_UNIT": QUESTION_UNIT,
    "QUESTION_LANGUAGE": QUESTION_LANGUAGE,
    "USER": USER,
}

SCHEMA = schema_block(
    "pick", {"choice": {"type": "string"}, "quote": {"type": "string"}}, ["choice", "quote"]
)


def render_user(readout: str, role: str, options: list[str]) -> str:
    question = QUESTION_LANGUAGE if role == "language" else QUESTION_UNIT.replace("{role}", role)
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (
        USER.replace("{readout}", readout.strip())
        .replace("{question}", question)
        .replace("{options}", lines)
    )
