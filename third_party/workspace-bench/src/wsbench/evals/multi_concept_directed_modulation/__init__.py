"""Multi-concept directed modulation: hold one to three unrelated concepts in mind while writing
a dictated sentence; does the lens read them at the writing positions, and does the binding
survive."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="multi_concept_directed_modulation",
        title="Multi-concept directed modulation",
        group="basic_mt",
        bank=Path("evals/multi_concept_directed_modulation/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 1.3k",
        sources="",
    )
)
