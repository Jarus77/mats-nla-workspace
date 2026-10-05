"""Prompts of the two-stage buggy-code judge: a blind reader infers the bug from the readouts
alone, then an informed grader scores that inference 1-10 against the ground truth. Printed
verbatim in the family README."""

from wsbench.llm import schema_block

PROMPT_VERSION = "buggy-score-2026-09-23"
CLAIMS = ("bug", "correct", "none")
SILENT_SCORE = {"buggy": 1, "clean": 5}  # a readout that claims nothing: see the README

INFER_SYSTEM = """\
You read free-text readouts produced by an interpretability lens while a language model
processed a short program you cannot see. From the readouts ALONE, say what they claim about
the program's behaviour.

- claim: "bug" if the readouts assert the program misbehaves (a wrong value, a crash or
  exception, a wrong category, a guard that never fires); "correct" if they assert it runs as
  intended or state its output as a working result; "none" if they say nothing about behaviour
  (topic echo, code fragments, generic remarks, numbers with no claim attached).
- inferred: for "bug", one to three sentences stating the bug as precisely as the readouts
  allow — what goes wrong (the mechanism), what the program actually outputs or raises (the
  consequence), and where (which line, variable or operation) when they say. For "correct",
  state the behaviour they assert. For "none", an empty string. Do not guess beyond what the
  readouts say; a readout that only tests negative input or mentions "error" without saying
  what happens is a weak bug claim and should be described as such.
- quote: the snippet from one readout that best supports the claim ("" for none). Test-assertion
  frames ("# Expected: -2", "assert total < 0", "# Output: ...") are the lens's native idiom and
  count as claims about behaviour.

Answer with a single JSON object: {"claim": "bug|correct|none", "inferred": str, "quote": str}."""

INFER_USER = "readout samples (k={k}):\n{samples}"

GRADE_SYSTEM = """\
You grade how closely a description of a program's behaviour, inferred from lens readouts,
matches the ground truth. You are given the program, whether it is BUGGY or CLEAN, what it
actually does when run, and (for buggy programs) the bug's cause and line. Score 1-10.

For a BUGGY program:
- 10: the exact bug — mechanism, consequence and location all right.
- 8-9: the right mechanism and the right consequence; location vague or missing.
- 6-7: the right consequence (its kind: negative total, crash, wrong branch...) OR the right
  mechanism, but not both. The exact magnitude of a wrong value need not match; its kind must.
- 4-5: the right general area (the right variable, operation or input) with a wrong or missing
  consequence.
- 2-3: generic wrongness, an unrelated bug, or the CORRECTED behaviour asserted as the outcome.
- 1: asserts the program is correct, or claims nothing about behaviour.

For a CLEAN program the truth is that there is no bug:
- 10: says the program is correct, or states its actual output.
- 7-9: no bug claimed, but the described behaviour is partly wrong.
- 4-6: hedged or vague bug talk with no specific false claim.
- 2-3: asserts a specific bug that does not exist.
- 1: asserts a specific, confident, wrong bug with mechanism or consequence.

Grade the inferred description only; do not re-find the bug yourself. Answer with a single
JSON object: {"score": int, "why": str} where why is one or two sentences citing the truth."""

GRADE_USER = (
    "item: {name}  (src={src}, language={language})\n"
    "code:\n```\n{code}\n```\n"
    "verified executed truth: {verified}\n"
    "{cause_line}"
    "inferred from the readouts (claim={claim}): {inferred}"
)
CAUSE_LINE = "cause ({cause_kind}): {cause}\n"
BUG_LINE = "bug line: {line}\n"

PROMPTS: dict[str, str] = {
    "INFER_SYSTEM": INFER_SYSTEM,
    "INFER_USER": INFER_USER,
    "GRADE_SYSTEM": GRADE_SYSTEM,
    "GRADE_USER": GRADE_USER,
    "CAUSE_LINE": CAUSE_LINE,
    "BUG_LINE": BUG_LINE,
}

INFER_SCHEMA = schema_block(
    "inferred_bug",
    {
        "claim": {"type": "string", "enum": list(CLAIMS)},
        "inferred": {"type": "string"},
        "quote": {"type": "string"},
    },
    ["claim", "inferred", "quote"],
)
GRADE_SCHEMA = schema_block(
    "closeness",
    {"score": {"type": "integer", "minimum": 1, "maximum": 10}, "why": {"type": "string"}},
    ["score", "why"],
)


def cause_of(item: dict) -> str | None:
    """A buggy item's ``cause``; a clean twin's ``why_correct`` or ``looks_like`` when it has one
    (most clean records carry neither, and then the line is omitted rather than filled with ?)."""
    for key in ("cause", "why_correct", "looks_like"):
        if item.get(key):
            return str(item[key])
    return None


def render_infer(samples: list[str]) -> str:
    lines = "\n".join(f"[{i}] {s}" for i, s in enumerate(samples))
    return INFER_USER.replace("{k}", str(len(samples))).replace("{samples}", lines)


def render_grade(item: dict, claim: str, inferred: str) -> str:
    src = str(item["src"])
    cause = cause_of(item)
    cause_line = (
        CAUSE_LINE.replace(
            "{cause_kind}", "the bug" if src == "buggy" else "the twin context"
        ).replace("{cause}", cause)
        if cause
        else ""
    )
    if src == "buggy" and item.get("bug_line"):
        cause_line += BUG_LINE.replace("{line}", str(item["bug_line"]))
    return (
        GRADE_USER.replace("{name}", str(item["name"]))
        .replace("{src}", src)
        .replace("{language}", str(item.get("language", "?")))
        .replace("{code}", str(item["code"]))
        .replace("{verified}", str(item.get("verified", "?")))
        .replace("{cause_line}", cause_line)
        .replace("{claim}", claim)
        .replace("{inferred}", inferred or "(nothing claimed about behaviour)")
    )
