"""Verbatim prompts of the conjunctive_association judge.

Source: ``scripts/oracle_lens/latent_eval/judge_mc.py`` in the source repo (L43-62, L76-85,
L190). Templates use ``{name}`` placeholders rendered with ``str.replace``. Every entry of
:data:`PROMPTS` is printed verbatim in ``evals/conjunctive_association/README.md``.
"""

from collections.abc import Sequence

from wsbench.llm import schema_block
from wsbench.mc import listing

COMP_SEED = 20260805  # compositional freeze date; fixes option order (reproducible)
PROMPT_VERSION = "comp-v1"

SYSTEM = """You judge the READOUT of an interpretability lens: text decoded from a language \
model's internal activations while it read a passage you cannot see (possibly labeled by \
layer; may contain noise, fragments, or other languages). You are given ONE multiple-choice \
question — which of the listed compositional descriptions the readout STATES. Pick an option \
ONLY if the readout names its content directly, as the direct word(s) or an EXACT synonym \
("mourning" names grief; "wife" names spouse; "his little boy" names her young child). A \
faithful translation of the option into another language counts as naming it. Do NOT \
pick an option you can merely INFER from evidence, causes, scene details, or narrative ("small \
sneakers" do not name "her young child"), do NOT count related-but-different states ("regret" \
is not grief; "worried" is not relieved) or generic words too broad to name it ("sad", \
"upset"), and do NOT use world knowledge or plausibility to fill gaps. If the readout does not \
state any listed option, choose the "cannot tell from the readout" option. Give a SHORT exact \
quote (a few words, VERBATIM) from the readouts that supports your choice."""

SCHEMA = schema_block(
    "readout_mc", {"choice": {"type": "integer"}, "quote": {"type": "string"}}, ["choice", "quote"]
)

QUESTION = "Which of the following does the readout state?\n{listing}"
USER = "READOUTS:\n{r}\n\n{q}"

PROMPTS: dict[str, str] = {"SYSTEM": SYSTEM, "QUESTION": QUESTION, "USER": USER}


def render_question(shown: Sequence[str]) -> str:
    return QUESTION.replace("{listing}", listing(shown))


def render_user(blob: str, q: str) -> str:
    return USER.replace("{r}", blob).replace("{q}", q)
