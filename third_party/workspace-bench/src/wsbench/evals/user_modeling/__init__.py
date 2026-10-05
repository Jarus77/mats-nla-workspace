"""User modelling: does the lens, read from the request sentence through the end of the render,
encode the user attribute Qwen inferred."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="user_modeling",
        title="User modelling",
        group="association",
        bank=Path("evals/user_modeling/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 20k x k",
        sources="Choi et al. 2025",
    )
)
