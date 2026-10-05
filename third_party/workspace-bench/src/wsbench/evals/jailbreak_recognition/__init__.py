"""Jailbreak recognition: does the lens show Qwen realizing the last user turn is a jailbreak."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="jailbreak_recognition",
        title="Jailbreak recognition",
        group="safety",
        bank=Path("evals/jailbreak_recognition/items.json"),
        # Pinned (Camila, 2026-09-16): Gemini 3.8 Flash refuses a share of these cells;
        # Sonnet 5 is the judge of record for this family.
        judge=JudgeConfig(model="claude-sonnet-5", prompt_version=PROMPT_VERSION, reasoning=None),
        metric="pass_rate",
        higher_is_better=True,
        run=judge.run,
        # every token of the last user turn (28,799 positions) x 5 layers, one call per cell
        calls_per_arm="≈ 144k",
        sources="Zhao et al. 2024",
    )
)
