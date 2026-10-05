"""Brew intermediates: a ten-rule colour-rewrite table stirred twice with the start colour given
last; can a lens name the colour after the first stir, which is never written?"""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="brew_intermediates",
        title="Brew intermediates",
        group="computational",
        bank=Path("evals/brew_intermediates/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 10k (emission + stir cells; ≈ 26k with opts=regions=all)",
        sources="",
    )
)
