"""The judge loop every MC family shares: cached, fingerprinted, one schema per batch."""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from wsbench import llm, registry
from wsbench.cache import Cache, fingerprint
from wsbench.judge_config import ResolvedJudge
from wsbench.llm import Spend
from wsbench.readouts import Cell
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult


@dataclass
class Call:
    key: str
    system: str
    user: str
    meta: dict = field(default_factory=dict)  # gold positions etc.; cached beside the result


def run_calls(
    calls: list[Call],
    *,
    schema: dict,
    judge: ResolvedJudge,
    prompt_version: str,
    cache: Cache,
    spend: Spend,
    concurrency: int,
    rpm: float,
    dry_run: bool,
    preflight: Callable[[], None] | None = None,
    validate: Callable[[Call, dict], bool] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, dict | None]:
    """key -> parsed result (``None`` = failed). Cached non-failed results are reused; the rest
    go through :func:`llm.stream_json` in one batch. Every landed result (incl. ``None``) is
    ``cache.put`` immediately as ``{"result": r, "meta": call.meta}``. ``dry_run`` prints the
    first call's prompts and returns ``{}`` without any client.

    ``judge`` is per batch (a stage may run on an aux model with its own reasoning);
    ``temperature`` / ``max_tokens`` pass through to the client and the fingerprint is
    ``(prompt_version, model, reasoning, temperature, system, user)``. A landed result that fails
    ``validate(call, result)`` is stored as ``{"result": None, "meta": {**call.meta, "raw": r}}``
    (so the next run re-queues the key) and returned as ``None``."""
    if dry_run:
        if calls:
            c = calls[0]
            print(f"--- judge prompt for {c.key} (model={judge.model}) ---")
            print("[system]")
            print(c.system)
            print("[user]")
            print(c.user)
        else:
            print("dry run: no judge calls")
        return {}
    out: dict[str, dict | None] = {}
    pending: list[tuple[Call, str]] = []
    for c in calls:
        fp = fingerprint(
            prompt_version, judge.model, judge.reasoning, temperature, c.system, c.user
        )
        row = cache.get(c.key, fp)
        if row is not None:
            out[c.key] = row["result"]
        else:
            pending.append((c, fp))
    print(f"judge: {len(pending)} calls ({len(out)} cached, model={judge.model})")
    if not pending:
        return out

    def on_result(i: int, r: dict | None) -> None:
        c, fp = pending[i]
        if r is not None and validate is not None and not validate(c, r):
            cache.put(c.key, fp, {"result": None, "meta": {**c.meta, "raw": r}})
            out[c.key] = None
            return
        cache.put(c.key, fp, {"result": r, "meta": c.meta})
        out[c.key] = r

    if preflight is not None:
        preflight()
    llm.stream_json(
        [(c.system, c.user) for c, _fp in pending],
        schema=schema,
        model=judge.model,
        reasoning=judge.reasoning,
        temperature=temperature,
        on_result=on_result,
        concurrency=concurrency,
        rpm=rpm,
        max_tokens=max_tokens,
        spend=spend,
    )
    for c, _fp in pending:
        out.setdefault(c.key, None)
    return out


# Process-global "already preflighted" set, shared by every family's Preflighter and seeded by
# ``runner.run_families`` so ``run all=True`` preflights each judge model once, not once per family.
_PREFLIGHT_LOCK = threading.Lock()
_PREFLIGHTED: set[str] = set()


def preflight_key(model: str, reasoning: dict | None) -> str:
    return json.dumps([model, reasoning], sort_keys=True)


class Preflighter:
    """Runs :func:`llm.preflight` at most once per (model, reasoning) per process (the set is
    module-global and lock-guarded; the lock is held across the call so "once" is exact)."""

    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run

    def ensure(self, judge: ResolvedJudge) -> None:
        if self._dry_run:
            return
        k = preflight_key(judge.model, judge.reasoning)
        with _PREFLIGHT_LOCK:
            if k in _PREFLIGHTED:
                return
            llm.preflight(judge.model, judge.reasoning)
            _PREFLIGHTED.add(k)

    def for_judge(self, judge: ResolvedJudge) -> Callable[[], None]:
        return lambda: self.ensure(judge)


# ---------------------------------------------------------------- family plumbing


def load_bank(family: str) -> list[dict]:
    path = registry.REPO_ROOT / "evals" / family / "items.json"
    return json.loads(path.read_text(encoding="utf-8"))


def item_scope(bank: list[dict], args: JudgeArgs) -> list[dict]:
    """bank ∩ ``items=`` (bank order), then ``limit=``."""
    items = bank
    if args.items is not None:
        want = set(args.items)
        items = [it for it in items if it["id"] in want]
    if args.limit > 0:
        items = items[: args.limit]
    return items


def items_without_readouts(scope: list[dict], cells: list[Cell]) -> int:
    """In-scope items with zero rows in the readouts file (item id = ``id``, or ``name`` for the
    user_modeling bank). ``complete`` cannot see a truncated file for families whose bank has no
    position list; this can."""
    present = {c.id for c in cells}
    return sum(1 for it in scope if it.get("id", it.get("name")) not in present)


def with_readout_count(result: FamilyResult, scope: list[dict], cells: list[Cell]) -> FamilyResult:
    """Attach ``extras["n_items_without_readouts"]`` to a family's result (rendered by
    ``report`` as ``n (k no readouts)``)."""
    result.extras["n_items_without_readouts"] = items_without_readouts(scope, cells)
    return result


def is_subset(args: JudgeArgs) -> bool:
    return args.items is not None or args.limit > 0 or args.layers is not None


def base_config(args: JudgeArgs, prompt_version: str, **extra: Any) -> dict:
    return {
        "judge_model": args.judge.model,
        "prompt_version": prompt_version,
        "reasoning": args.judge.reasoning,
        "layers": args.layers,
        "dry_run": args.dry_run,
        "concurrency": args.concurrency,
        "rpm": args.rpm,
        "allow_missing": args.allow_missing,
        "items": args.items,
        "limit": args.limit,
        **extra,
    }


def base_counts(
    *,
    n_expected: int,
    n_unjudged: int,
    n_empty: int,
    skipped_rows: int,
    spend: Spend,
    n_missing: int = 0,
) -> dict:
    return {
        "n_expected_cells": n_expected,
        "n_missing_cells": n_missing,
        "n_unjudged_cells": n_unjudged,
        "n_empty_cells": n_empty,
        "skipped_rows": skipped_rows,
        "spend_usd": spend.usd,
    }


def choice_of(r: dict | None) -> Any:
    return r.get("choice") if isinstance(r, dict) else None


def quote_of(r: dict | None) -> str:
    q = r.get("quote") if isinstance(r, dict) else None
    return (q or "").strip() if isinstance(q, str) else ""
