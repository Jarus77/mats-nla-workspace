"""J-lens concept precision: how much of an AO arm's prose is supported by the J-lens top-10."""

from pathlib import Path

from wsbench.judge_config import DEFAULT_JUDGE, JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="jlens_concept_pr",
        title="J-lens concept precision",
        group="precision",
        bank=Path("evals/jlens_concept_pr/manifest.json"),
        judge=JudgeConfig(
            model=DEFAULT_JUDGE,
            prompt_version=PROMPT_VERSION,
            reasoning={"effort": "minimal"},
        ),
        metric="precision",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 35k",
        sources="arXiv:2607.15495",
    )
)
