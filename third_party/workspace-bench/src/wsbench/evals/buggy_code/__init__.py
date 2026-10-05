"""Buggy code: short programs with one verified bug and their clean twins, read at the end of the
file with nothing asked; how closely does the bug a blind reader infers from the readouts match
the real one, and does the reader infer none on the clean twin?"""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="buggy_code",
        title="Buggy code",
        group="computational",
        bank=Path("evals/buggy_code/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="score",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="49-98",
        sources="",
    )
)
