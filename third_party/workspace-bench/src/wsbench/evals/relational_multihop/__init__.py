"""Relational multihop: does the lens hold a composed two-hop relation in the right order."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="relational_multihop",
        title="Relational multihop",
        group="bag_of_words",
        bank=Path("evals/relational_multihop/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 2.2k",
        sources="",
    )
)
