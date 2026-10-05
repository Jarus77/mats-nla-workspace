"""Arithmetic intermediates: a bare two- or three-operation expression answered with no chain of
thought; does the lens assert the never-written intermediate — at the variant's frozen cell
(``cells=frozen``) or anywhere in the prompt, one batched call per item and layer
(``cells=all``)?"""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="arithmetic_intermediates",
        title="Arithmetic intermediates",
        group="computational",
        bank=Path("evals/arithmetic_intermediates/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm=(
            "≈ 1.2k batched calls (cells=all: every position x 2 layers ≈ 44k cells, "
            "one call per item and layer); cells=frozen: 596"
        ),
        sources="",
    )
)
