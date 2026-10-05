"""``wsbench run``: judge several families concurrently, fail-soft, one preflight per model."""

import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from wsbench import llm, mcjudge
from wsbench.family import fail
from wsbench.judge_config import ResolvedJudge, resolve
from wsbench.llm import JudgeConfigError
from wsbench.registry import EvalSpec, JudgeArgs
from wsbench.results import FamilyResult, write_results

Status = Literal["ok", "skipped", "failed"]


@dataclass
class FamilyOutcome:
    family: str
    status: Status
    result: FamilyResult | None = None
    error: str | None = None  # for skipped / failed
    exit_code: int = 0  # 3 JudgeConfigError, 2 SystemExit


class Options(Protocol):
    """The judge options ``wsbench judge`` passes down (``cli.JudgeOptions``)."""

    judge_model: str | None
    layers: list[int] | None
    items: list[str] | None
    limit: int
    allow_missing: bool
    concurrency: int
    rpm: float
    dry_run: bool
    opt: list[str]


class RunOptions(Options, Protocol):
    """``wsbench run``'s options: the judge options plus the thread count."""

    family_workers: int


def parse_opts(pairs: Sequence[str]) -> dict[str, str]:
    """``opts=KEY=VALUE,...`` -> dict; a malformed pair is a usage error (exit 2)."""
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            fail(f"opts= expects KEY=VALUE pairs, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


def judge_family(
    spec: EvalSpec,
    args: Options,
    readouts: Path,
    out: Path,
    *,
    judge: ResolvedJudge,
    opts: Mapping[str, str],
) -> FamilyResult:
    """Run one family with an already-resolved judge and pre-parsed ``opts=``; write its
    ``results.json`` and print the one-line summary."""
    jargs = JudgeArgs(
        readouts=readouts,
        out=out,
        judge=judge,
        layers=args.layers,
        items=args.items,
        limit=args.limit,
        allow_missing=args.allow_missing,
        concurrency=args.concurrency,
        rpm=args.rpm,
        dry_run=args.dry_run,
        aux_models=spec.judge.aux_models,
        extra=dict(opts),
    )
    result = spec.run(jargs)
    path = write_results(out, result)
    spend = result.counts.get("spend_usd", 0.0) or 0.0
    print(
        f"{spec.name}: {result.metric}={result.value} n={result.n_items} "
        f"spend=${spend:.2f} pinned={result.pinned_instrument} complete={result.complete} "
        f"-> {path}"
    )
    return result


def _one(
    spec: EvalSpec,
    args: Options,
    readouts: Path,
    out: Path,
    judge: ResolvedJudge,
    opts: Mapping[str, str],
) -> FamilyOutcome:
    """One family, fail-soft: ``JudgeConfigError`` -> failed/3, ``SystemExit`` (missing cells,
    refused input) -> failed/2. Anything else propagates."""
    try:
        result = judge_family(spec, args, readouts, out, judge=judge, opts=opts)
    except JudgeConfigError as e:
        print(f"{spec.name}: judge config error: {e}", file=sys.stderr)
        return FamilyOutcome(spec.name, "failed", error=str(e), exit_code=3)
    except SystemExit as e:
        text = e.code if isinstance(e.code, str) else f"exit {e.code}"
        print(f"{spec.name}: failed ({text})", file=sys.stderr)
        return FamilyOutcome(spec.name, "failed", error=text, exit_code=2)
    return FamilyOutcome(spec.name, "ok", result=result)


def _seed_preflights(judges: Mapping[str, ResolvedJudge]) -> None:
    """Preflight each distinct (model, reasoning) once, through the module attribute (so one
    monkeypatch counts the runner and the families), and seed ``mcjudge._PREFLIGHTED``."""
    seen: dict[str, ResolvedJudge] = {}
    for j in judges.values():
        seen.setdefault(mcjudge.preflight_key(j.model, j.reasoning), j)
    for k, j in seen.items():
        with mcjudge._PREFLIGHT_LOCK:
            if k in mcjudge._PREFLIGHTED:
                continue
            llm.preflight(j.model, j.reasoning)
            mcjudge._PREFLIGHTED.add(k)


def run_families(
    specs: Sequence[EvalSpec],
    args: RunOptions,
    *,
    readouts_root: Path,
    out: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[list[FamilyOutcome], int]:
    """Judge ``specs`` from ``readouts_root/<family>.jsonl`` into ``out/<family>/``.

    Order of operations: parse ``opts=`` once (a bad pair is one ``SystemExit``, nothing
    started); resolve every judge once; skip families with no readouts file; unless
    ``dry_run=True``, preflight each distinct judge model once (a failure aborts with exit 3
    before any thread starts); then run the rest in a ``ThreadPoolExecutor`` with
    ``min(len(runnable), family_workers)`` workers (1 under ``dry_run`` so printed prompts
    do not interleave). The process-wide pacer is shared, so ``rpm=`` bounds the whole run.

    Returns the outcomes in ``specs`` order and the exit code: 130 if interrupted, else 3 if
    any family hit a ``JudgeConfigError``, else 2 if any raised ``SystemExit``, else 0.
    ``KeyboardInterrupt`` cancels unstarted families (``skipped``, "interrupted") and waits for
    in-flight ones to finish their current batch (they are recorded normally)."""
    env = os.environ if env is None else env
    opts = parse_opts(args.opt)
    judges = {s.name: resolve(s.judge, flag=args.judge_model, env=env) for s in specs}
    outcomes: dict[str, FamilyOutcome] = {}
    runnable: list[tuple[EvalSpec, Path]] = []
    for s in specs:
        readouts = readouts_root / f"{s.name}.jsonl"
        if not readouts.exists():
            msg = f"no readouts at {readouts}"
            print(f"{s.name}: skipped ({msg})")
            outcomes[s.name] = FamilyOutcome(s.name, "skipped", error=msg)
        else:
            runnable.append((s, readouts))
    # a family with a deterministic scorer of record makes no call unless opts=judge=mc
    needs_llm = [s for s, _r in runnable if not s.scorer or opts.get("judge") == "mc"]
    if needs_llm and not args.dry_run:
        try:
            _seed_preflights({s.name: judges[s.name] for s in needs_llm})
        except JudgeConfigError as e:
            print(f"judge config error: {e}", file=sys.stderr)
            return [], 3

    workers = 1 if args.dry_run else max(1, min(len(runnable), args.family_workers))
    interrupted = False
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wsbench-family")
    futures: dict[Future[FamilyOutcome], str] = {}
    try:
        for s, readouts in runnable:
            fut = executor.submit(_one, s, args, readouts, out / s.name, judges[s.name], opts)
            futures[fut] = s.name
        for fut in as_completed(futures):
            oc = fut.result()
            outcomes[oc.family] = oc
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted: cancelling unstarted families", file=sys.stderr)
        # unstarted families are cancelled; in-flight ones cannot be and finish their batch
        executor.shutdown(wait=True, cancel_futures=True)
        for fut, name in futures.items():
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                outcomes[name] = fut.result()
        for s, _readouts in runnable:
            outcomes.setdefault(s.name, FamilyOutcome(s.name, "skipped", error="interrupted"))
    except BaseException:
        # an unexpected worker exception propagates (plan), but must not leave the other
        # families spending in the background
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    ordered = [outcomes[s.name] for s in specs]
    if interrupted:
        code = 130
    elif any(o.exit_code == 3 for o in ordered):
        code = 3
    elif any(o.exit_code == 2 for o in ordered):
        code = 2
    else:
        code = 0
    return ordered, code


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_manifest(
    outcomes: Sequence[FamilyOutcome],
    *,
    started: datetime,
    finished: datetime,
    args: Options,
    out: Path,
    readouts_root: Path,
    env: Mapping[str, str] | None = None,
) -> dict:
    """The ``run.json`` payload: per-family status plus the judge overrides in force."""
    env = os.environ if env is None else env
    families: dict[str, dict] = {}
    for o in outcomes:
        d: dict = {"status": o.status}
        if o.error is not None:
            d["error"] = o.error
        if o.result is not None:
            r = o.result
            d.update(
                results_path=str(out / o.family / "results.json"),
                value=r.value,
                complete=r.complete,
                pinned_instrument=r.pinned_instrument,
                spend_usd=r.counts.get("spend_usd"),
            )
            if "n_calls" in r.extras:
                d["n_calls"] = r.extras["n_calls"]
        families[o.family] = d
    return {
        "started": _iso(started),
        "finished": _iso(finished),
        "families": families,
        "judge_overrides": {"flag": args.judge_model, "env": env.get("WSBENCH_JUDGE_MODEL")},
        "readouts_root": str(readouts_root),
    }
