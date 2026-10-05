"""Directed modulation: the model is told to think about, suppress, or hide a concept while
copying an unrelated sentence; does the lens read that concept at the writing positions."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="directed_modulation",
        title="Directed modulation",
        group="basic",
        bank=Path("evals/directed_modulation/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 9k",
        sources="",
    )
)
