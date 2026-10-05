"""Conjunctive association: does the lens state a compound state a vignette implies."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="conjunctive_association",
        title="Conjunctive association",
        group="association",
        bank=Path("evals/conjunctive_association/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 100",
        sources="",
    )
)
