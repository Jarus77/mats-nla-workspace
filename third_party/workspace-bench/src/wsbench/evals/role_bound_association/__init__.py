"""Role-bound association: does the lens bind the direction of a scene's relation."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="role_bound_association",
        title="Role-bound association",
        group="association",
        bank=Path("evals/role_bound_association/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 11.4k",
        sources="",
    )
)
