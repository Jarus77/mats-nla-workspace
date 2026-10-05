"""What every judged family shares: the fatal-exit rule, cell text, the verbatim-quote check,
rates, the tri-state item rule and the pass-rate result epilogue."""

import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, NoReturn

from wsbench.llm import Spend
from wsbench.mc import fold
from wsbench.mcjudge import base_config, base_counts, is_subset
from wsbench.readouts import Cell, LoadReport
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness
from wsbench.summarizer import render_bag


def fail(msg: str) -> NoReturn:
    """A usage or data error: message on stderr, exit 2 (the CLI contract for missing cells)."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def rate(flags: Iterable[bool]) -> float | None:
    xs = list(flags)
    return sum(xs) / len(xs) if xs else None


def mean(xs: Iterable[float]) -> float | None:
    ys = list(xs)
    return sum(ys) / len(ys) if ys else None


def quote_in(quote: str, text: str) -> bool:
    """A verbatim quote check: present after folding case, accents and quotes. A quote that is
    empty after folding (e.g. bare combining marks) never counts."""
    q = fold(quote.strip())
    return bool(q) and q in fold(text)


def cell_text(cell: Cell) -> tuple[str, str]:
    """``(judge_text, plain_text)`` of a cell: a token lens is shown as its scored bag but
    verified against the bare tokens; a prose cell is its non-blank samples joined by newlines.
    Both are empty for an empty cell."""
    if cell.tokens is not None:
        if not cell.tokens:
            return "", ""
        return render_bag(cell.tokens, cell.scores), " | ".join(cell.tokens)
    text = "\n".join(s for s in (cell.samples or ()) if s.strip())
    return text, text


def last_pos_rows(cells: Sequence[Cell]) -> tuple[dict[tuple[str, int], Cell], int]:
    """Per (item, layer) the max-pos row (the last read position), and how many other rows the
    file carried at those (item, layer) keys."""
    groups: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for c in cells:
        groups[(c.id, c.layer)].append(c)
    chosen = {k: max(cs, key=lambda c: c.pos) for k, cs in groups.items()}
    return chosen, sum(len(cs) - 1 for cs in groups.values())


def require_cells(name: str, missing: Sequence[Any], n_expected: int, args: JudgeArgs) -> None:
    """Missing cells are fatal (exit 2) unless ``allow_missing=True``; a dry run only reports."""
    if missing and not args.allow_missing and not args.dry_run:
        fail(
            f"{name}: {len(missing)} of {n_expected} expected cells have no readout; "
            "pass allow_missing=True to score the rest"
        )


def tri_state(passed: bool, incomplete: bool) -> bool | None:
    """The item rule every pass-rate family shares: a pass stands on its own; no pass with an
    unjudged, unsummarized or missing cell is undecided (None) and leaves the denominator."""
    return True if passed else (None if incomplete else False)


def pass_rate_result(
    *,
    name: str,
    args: JudgeArgs,
    prompt_version: str,
    rows: list[dict[str, Any]],
    cells: Sequence[Cell],
    rep: LoadReport,
    n_expected: int,
    n_missing: int,
    n_unjudged: int,
    n_empty: int,
    spend: Spend,
    extras: dict[str, Any],
    chance_label: str,
    skipped_extra: int = 0,
    config_extra: dict[str, Any] | None = None,
) -> FamilyResult:
    """The epilogue of a pass-rate family: ``rows[*]["pass"]`` is True / False / None, the
    value is the rate over decided rows with a bootstrap CI, ``complete`` follows the shared
    rule, and ``extras`` gains the decided / undecided counts."""
    decided = [r for r in rows if r["pass"] is not None]
    passes = [1.0 if r["pass"] else 0.0 for r in decided]
    return FamilyResult(
        family=name,
        metric="pass_rate",
        value=sum(passes) / len(passes) if passes else None,
        ci95=bootstrap_ci(passes) if passes else None,
        n_items=len(rows),
        higher_is_better=True,
        chance=None,
        chance_label=chance_label,
        complete=bool(cells)
        and completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=n_expected,
            n_missing=n_missing,
            n_unjudged=n_unjudged,
            n_empty=n_empty,
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(args, prompt_version, kind=rep.kind or "prose", **(config_extra or {})),
        counts=base_counts(
            n_expected=n_expected,
            n_missing=n_missing,
            n_unjudged=n_unjudged,
            n_empty=n_empty,
            skipped_rows=sum(rep.skipped.values()) + skipped_extra,
            spend=spend,
        ),
        extras={
            **extras,
            "n_items_decided": len(decided),
            "n_items_undecided": len(rows) - len(decided),
        },
        rows=rows,
    )
