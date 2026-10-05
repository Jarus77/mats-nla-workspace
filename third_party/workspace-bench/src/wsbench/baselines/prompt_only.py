"""Prompt-only floor: stock Qwen3.6-27B given the prompt text (no activation) is asked what the
model is thinking; its summaries are judged by each family's own instrument (``wsbench judge``)
and ``freeze`` records the rates. See ``evals/baselines/README.md``."""

import datetime as dt
import json
from pathlib import Path
from typing import Any

from wsbench.registry import REPO_ROOT
from wsbench.results import read_results

FROZEN = REPO_ROOT / "evals" / "baselines" / "prompt_only.json"
# Families where the prompt states the answer outright, so the stock model's summary names it
# every time (multi-concept DM scored 1.0): not a floor, never frozen or shown (Agam, 2026-09-16).
EXCLUDED: dict[str, str] = {
    "multi_concept_directed_modulation": "the prompt dictates the concepts; prompt-only is 1.0",
    "hallucination": "an invention rate, not a bar a lens clears",
    "jlens_concept_pr": "precision against the J-lens top-10, not a bar a lens clears",
}
SOURCE_KEYS = (
    "model",
    "base_model",
    "adapter",
    "prompt_kind",
    "prompt_template",
    "sampling",
    "layers",
)
# Families whose prompt text states or dictates the answer, so the rate is pinned high and
# bounds what the text gives away, not what a lens must clear. `report` marks these instead of
# drawing them as a bar; the comparison there is the family's own null (see the README).
SATURATED: dict[str, str] = {
    "role_bound_association": "the scene states who did what to whom",
    "relational_multihop": "the cloze states both hops",
    "brew_intermediates": "the rule table and the start colour are in the prompt",
    "typo": "the correction is recoverable from the misspelling",
}


def instrument_versions() -> dict[str, str]:
    from wsbench import registry

    registry.load_all()
    return {name: spec.judge.prompt_version for name, spec in registry.FAMILIES.items()}


def freeze(run_dir: Path, dst: Path = FROZEN, *, source: Path | None = None) -> dict[str, Any]:
    """Fold ``run_dir/<family>/results.json`` (each family's judge run on the prompt-only
    readouts) into ``dst`` as ``{family: entry}``, stamped with the judge model and prompt
    version the rate belongs to. ``source`` is the generation's ``run_config.json`` (model,
    template, sampling), copied under ``_source``. A subset run (``items`` / ``limit``) or a
    run with no value is refused (ValueError); families not in ``run_dir`` keep their entry."""
    versions = instrument_versions()
    frozen: dict[str, Any] = {}
    for path in sorted(run_dir.glob("*/results.json")):
        r = read_results(path.parent)
        if r.family in EXCLUDED:
            continue
        if r.config.get("items") or r.config.get("limit"):
            raise ValueError(f"freeze: {path} is a subset run, not a baseline")
        if r.value is None:
            raise ValueError(f"freeze: {path} has no value")
        if versions.get(r.family) != r.config.get("prompt_version"):
            raise ValueError(
                f"freeze: {path} was judged with prompt_version "
                f"{r.config.get('prompt_version')!r}, the family's instrument is "
                f"{versions.get(r.family)!r}; re-judge first"
            )
        frozen[r.family] = {
            "rate": r.value,
            "saturated": r.family in SATURATED,
            "ci95": list(r.ci95) if r.ci95 else None,
            "metric": r.metric,
            "higher_is_better": r.higher_is_better,
            "n_items": r.n_items,
            "n_items_decided": r.extras.get("n_items_decided"),
            "counts": {
                k: r.counts.get(k)
                for k in (
                    "n_expected_cells",
                    "n_missing_cells",
                    "n_unjudged_cells",
                    "n_empty_cells",
                )
            },
            "judge_model": r.config.get("judge_model"),
            "instrument": r.config.get("prompt_version"),
            "layers_judged": r.config.get("layers_judged"),
            "complete": r.complete,
            "written": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
    if not frozen:
        raise ValueError(f"freeze: no <family>/results.json under {run_dir}")
    merged = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
    if source is not None:
        cfg = json.loads(Path(source).read_text(encoding="utf-8"))
        merged["_source"] = {k: cfg.get(k) for k in SOURCE_KEYS if k in cfg}
    merged.update(frozen)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return merged


def floors(dst: Path = FROZEN) -> dict[str, dict[str, Any]]:
    """Frozen prompt-only entries whose ``instrument`` matches the family's current judge
    prompt version; stale entries are dropped."""
    if not dst.exists():
        return {}
    versions = instrument_versions()
    frozen = json.loads(dst.read_text(encoding="utf-8"))
    return {
        fam: e
        for fam, e in frozen.items()
        if not fam.startswith("_")
        and fam not in EXCLUDED
        and versions.get(fam) is not None
        and e.get("instrument") == versions.get(fam)
    }
