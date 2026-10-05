"""Chained intermediates: a two- or three-step arithmetic chain with the start number given last,
answered with no chain of thought; can a lens name an intermediate the model computed but never
wrote?"""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="chain_intermediates",
        title="Chained intermediates",
        group="computational",
        bank=Path("evals/chain_intermediates/items.json"),
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 1.3k",
        sources="",  # in-house items; the construct idea is named in the family README
    )
)
