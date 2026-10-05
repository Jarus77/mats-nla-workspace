"""Family registry: each ``wsbench.evals.<family>`` package registers one ``EvalSpec`` on import."""

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from wsbench.judge_config import JudgeConfig, ResolvedJudge
from wsbench.results import FamilyResult

# src layout (uv-managed checkout); a wheel install would resolve to site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES: dict[str, "EvalSpec"] = {}  # filled by register(); EvalSpec is defined below


@dataclass
class JudgeArgs:
    """What ``cli.judge`` passes to every family. ``limit == 0`` means no limit."""

    readouts: Path
    out: Path
    judge: ResolvedJudge
    layers: list[int] | None
    items: list[str] | None
    limit: int
    allow_missing: bool
    concurrency: int
    rpm: float
    dry_run: bool
    aux_models: Mapping[str, str] = field(default_factory=dict)  # e.g. {"summarizer": model}
    extra: dict[str, str] = field(default_factory=dict)  # family options from opts=key=value


@dataclass(frozen=True)
class EvalSpec:
    name: str  # family key, e.g. "moral_rationale"
    title: str  # README name, e.g. "Moral rationale"
    group: str  # "basic" | "basic_mt" | "safety" | "association" | "bag_of_words" | "precision"
    #            | "logic" | "computational"
    bank: Path  # the bank file, relative to REPO_ROOT (items.json; jlens_concept_pr: manifest.json)
    judge: JudgeConfig
    metric: str
    higher_is_better: bool
    run: Callable[[JudgeArgs], FamilyResult]  # returns a FamilyResult; writes nothing
    calls_per_arm: str = ""  # human string for `wsbench list`, e.g. "≈ 7k" (judge calls per arm)
    sources: str = ""  # credit substring that appears verbatim in README §Credits and NOTICE.md
    # a deterministic scorer of record ("regex"): `wsbench list` shows it as the judge and `run`
    # skips its preflight unless opts=judge=mc asks for the family's optional LLM judge
    scorer: str = ""


def register(spec: EvalSpec) -> EvalSpec:
    if spec.name in FAMILIES:
        raise ValueError(f"family {spec.name!r} is already registered")
    FAMILIES[spec.name] = spec
    return spec


def get(name: str) -> EvalSpec:
    try:
        return FAMILIES[name]
    except KeyError:
        known = ", ".join(sorted(FAMILIES)) or "(none)"
        raise KeyError(f"unknown family {name!r}; known families: {known}") from None


def load_all() -> None:
    """Import every ``wsbench.evals.<pkg>`` so FAMILIES is populated. Idempotent."""
    import wsbench.evals as evals_pkg

    for mod in pkgutil.iter_modules(evals_pkg.__path__):
        importlib.import_module(f"wsbench.evals.{mod.name}")
