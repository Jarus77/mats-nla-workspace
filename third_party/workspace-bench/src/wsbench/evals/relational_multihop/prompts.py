"""Verbatim prompts of the relational_multihop judge.

Source: ``scripts/oracle_lens_evals/judge_relational_multihop.py`` in the source repo (L70-85,
L95-107, L183-194, L209-214). Templates use ``{name}`` placeholders rendered with
``str.replace``. Every entry of :data:`PROMPTS` is printed verbatim in
``evals/relational_multihop/README.md``.
"""

from wsbench.llm import schema_block

REL_SEED = 20260813  # fixes MC option order + distractor draw (reproducible)
PROMPT_VERSION = "rel-v1"
KINSHIP = {"sibling", "cousin", "parent", "spouse", "child", "grandparent", "grandchild"}
BLANK_TOKEN = "'s"  # the read site: the possessive right before the composed relation
N_PER_DOMAIN = 4  # distractors drawn per domain (professional, kinship)
N_MC_OPTIONS = 2 * (N_PER_DOMAIN + 1) + 1  # both hops + distractors + "cannot tell" = 11

# Near-miss relations the BANK never uses as a gold, unioned into the distractor pools so a
# confabulated adjacent relation has its OWN option instead of being rounded off to the gold
# (see the module docstring). None of these is a synonym of a bank relation (the judge is told
# synonyms count, so a "teacher" line against gold `tutor` would penalise a correct read); they
# are relations a reader could mistake for the gold but that are a DIFFERENT relation. The two
# pools are disjoint on purpose ("friend" is professional-side only: it is not kinship, and a
# word in both pools would make a list's domain balance unreadable).
KIN_EXTRA = ("grandparent", "grandchild", "aunt/uncle", "niece/nephew", "in-law", "stepparent")
PROF_EXTRA = ("friend", "roommate", "classmate", "teammate", "accountant", "realtor")

JUDGE_SYSTEM = (
    "You judge the READOUT of an interpretability lens — noisy text decoded from a language "
    "model's internal activations. Answer the multiple-choice question based ONLY on what the "
    "readout states or clearly implies (synonyms, gendered forms, or another language's word "
    "for a relation count); do not use world knowledge or plausibility to fill gaps. The "
    'readout describes a relation of the form "X\'s Y" (X is the outer/first relation word, Y '
    "the inner/second). If the readout does not support an answer, choose 'cannot tell from the "
    "readout'. Give a SHORT exact quote (a few words, VERBATIM) from the readout that supports "
    "your choice."
)
MC_SCHEMA = schema_block(
    "relation_mc", {"choice": {"type": "integer"}, "quote": {"type": "string"}}, ["choice", "quote"]
)

ROLE_X = "X (the OUTER/first relation word)"
ROLE_Y = "Y (the INNER/second relation word)"
QUESTION = (
    "READOUT:\n---\n{readout}\n---\n"
    'The readout describes a relation of the form "X\'s Y". What is {role}?\n{listing}'
)
BUNDLE = "[position {token!r}] {txt}"  # samples joined by " | "

PROMPTS: dict[str, str] = {
    "JUDGE_SYSTEM": JUDGE_SYSTEM,
    "ROLE_X": ROLE_X,
    "ROLE_Y": ROLE_Y,
    "QUESTION": QUESTION,
    "BUNDLE": BUNDLE,
}


def render_question(readout: str, which: str, listing_block: str) -> str:
    role = ROLE_X if which == "X" else ROLE_Y
    return (
        QUESTION.replace("{readout}", readout)
        .replace("{role}", role)
        .replace("{listing}", listing_block)
    )


def render_bundle(token: str | None, pos: int, txt: str) -> str:
    """``[position '\\'s'] ...`` when the row carries its token, ``[position p<pos>]`` otherwise."""
    label = repr(token) if token is not None else f"p{pos}"
    return BUNDLE.replace("{token!r}", label).replace("{txt}", txt)
