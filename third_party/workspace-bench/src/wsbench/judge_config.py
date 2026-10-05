"""Per-family judge configuration and the flag > env > family-pin override resolution."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from wsbench.llm import route

DEFAULT_JUDGE = "google/gemini-3.8-flash"
DEFAULT_REASONING = {"effort": "minimal"}  # Gemini cannot turn reasoning off
ENV_OVERRIDE = "WSBENCH_JUDGE_MODEL"


@dataclass(frozen=True)
class JudgeConfig:
    model: str = DEFAULT_JUDGE
    prompt_version: str = "v1"
    reasoning: dict | None = None  # None -> DEFAULT_REASONING if route is openrouter, else None
    aux_models: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedJudge:
    model: str
    reasoning: dict | None
    pinned: bool  # True iff model == config.model (no override took effect)
    source: Literal["flag", "env", "family"]


def resolve(
    config: JudgeConfig, *, flag: str | None = None, env: Mapping[str, str] | None = None
) -> ResolvedJudge:
    env = env or {}
    env_model = env.get(ENV_OVERRIDE, "")
    if flag:
        model, source = flag, "flag"
    elif env_model:
        model, source = env_model, "env"
    else:
        model, source = config.model, "family"
    if config.reasoning is not None:
        reasoning: dict | None = config.reasoning
    elif route(model) == "openrouter":
        reasoning = dict(DEFAULT_REASONING)
    else:
        reasoning = None
    return ResolvedJudge(
        model=model, reasoning=reasoning, pinned=model == config.model, source=source
    )
