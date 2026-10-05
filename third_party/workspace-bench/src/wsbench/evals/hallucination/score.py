"""hallucination scoring: ``hallucination_rate`` = hallucinated / specific (lower is better).

The numbers block (``_counts`` / ``_rates`` / ``bootstrap_rate_ci`` / ``numbers``) is ported
verbatim from ``hallucination-bench/src/hallucination_bench/judge.py`` L345-508. The headline is
the UNREVOKED rate, read beside ``assert_share``; ``hallucination_rate_revoked`` is reported in
``extras`` (a k=1 arm can never be revoked, so only the unrevoked rate compares arms). ``ci95`` is
the source's item-level percentile bootstrap of the ratio of per-item sums (2,000 draws, seed 0),
not ``results.bootstrap_ci``. ``n_items`` is the number of items in scope; the value itself is a
readout-level ratio.

Stage-2 claim tallies (``ReadoutVerdict.claims``) add the ``n_claims_*`` counts and the
``verifiable_share`` / ``false_share_of_verifiable`` / ``unverifiable_claims_per_readout`` rates
to every block; they never touch the headline or ``complete``. ``n_claims_false`` is span-level
(the established wrong spans per tallied readout, revoked or not), unlike the readout-level
headline.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult, completeness

from .judge import HAL_K, SPECIFIC_CLASSES, CellRecord

N_BOOT = 2000
CHANCE_LABEL = "precision instrument: no chance line; excluded from the macro"


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def _counts(cells: Sequence[CellRecord], k: int = HAL_K) -> dict[str, int]:
    c = {
        "n_cells": len(cells),
        "n_unjudged_cells": 0,
        "n_short_cells": 0,
        "n_readouts": 0,
        "n_specific": 0,
        "n_hallucinated": 0,
        "n_off_topic": 0,
        "n_revoked": 0,
        "n_unverified_spans": 0,
        # stage 2 (claim verification) over the readouts that carry a tally
        "n_readouts_claims_judged": 0,
        "n_cells_claims_unjudged": 0,  # span-judged cells without a tally (failed or verify=0)
        "n_claims_false": 0,
        "n_claims_true": 0,
        "n_claims_unverifiable": 0,
        "n_claims_disputed": 0,
        "n_claims_off_topic": 0,
        "n_unverified_claim_quotes": 0,
    }
    for cell in cells:
        if cell.n_samples < k:
            c["n_short_cells"] += 1
        if cell.verdicts is None:
            c["n_unjudged_cells"] += 1
            continue
        if any(v.claims is None for v in cell.verdicts if v.cls != "unjudged"):
            c["n_cells_claims_unjudged"] += 1
        for v in cell.verdicts:
            if v.cls == "unjudged":
                continue
            c["n_readouts"] += 1
            c["n_unverified_spans"] += v.n_unverified
            if v.claims is not None:
                c["n_readouts_claims_judged"] += 1
                c["n_claims_false"] += v.claims["false"]
                c["n_claims_true"] += v.claims["true"]
                c["n_claims_unverifiable"] += v.claims["unverifiable"]
                c["n_claims_disputed"] += v.claims["disputed"]
                c["n_claims_off_topic"] += v.claims["off_topic"]
                c["n_unverified_claim_quotes"] += v.claims["n_unverified"]
            if v.cls in SPECIFIC_CLASSES:
                c["n_specific"] += 1
            if v.cls == "hallucinated":
                c["n_hallucinated"] += 1
                c["n_revoked"] += int(v.revoked)
            elif v.cls == "off_topic":
                c["n_off_topic"] += 1
    return c


def _rates(c: dict[str, int]) -> dict[str, float | None]:
    return {
        "hallucination_rate": _ratio(c["n_hallucinated"], c["n_specific"]),
        "hallucination_rate_revoked": _ratio(c["n_hallucinated"] - c["n_revoked"], c["n_specific"]),
        "off_topic_rate": _ratio(c["n_off_topic"], c["n_specific"]),
        "assert_share": _ratio(c["n_specific"], c["n_readouts"]),
        # stage 2: verifiable = true + false; specific claims = verifiable + unverifiable
        "verifiable_share": _ratio(
            c["n_claims_true"] + c["n_claims_false"],
            c["n_claims_true"] + c["n_claims_false"] + c["n_claims_unverifiable"],
        ),
        "false_share_of_verifiable": _ratio(
            c["n_claims_false"], c["n_claims_true"] + c["n_claims_false"]
        ),
        "unverifiable_claims_per_readout": _ratio(
            c["n_claims_unverifiable"], c["n_readouts_claims_judged"]
        ),
    }


def bootstrap_rate_ci(
    cells: Sequence[CellRecord], *, n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float] | None:
    """Item-level percentile bootstrap (2.5, 97.5) of hallucinated / specific, ratio of sums.
    Draws whose resampled denominator is 0 are skipped; None when no draw remains."""
    per_item: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for cell in cells:
        c = _counts([cell])
        per_item[cell.item][0] += c["n_hallucinated"]
        per_item[cell.item][1] += c["n_specific"]
    items = list(per_item.values())
    if not items:
        return None
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        h = s = 0
        for _ in range(len(items)):
            a, b = items[rng.randrange(len(items))]
            h += a
            s += b
        if s:
            draws.append(h / s)
    if not draws:
        return None
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return lo, hi


def _block(cells: Sequence[CellRecord], k: int) -> dict[str, Any]:
    c = _counts(cells, k)
    return {**c, **_rates(c), "ci95": bootstrap_rate_ci(cells)}


def _split(cells: Sequence[CellRecord], key: Callable[[CellRecord], Any], k: int) -> dict[str, Any]:
    groups: dict[str, list[CellRecord]] = defaultdict(list)
    for cell in cells:
        groups[str(key(cell))].append(cell)
    return {g: _block(v, k) for g, v in sorted(groups.items())}


def numbers(cells: Sequence[CellRecord], *, k: int = HAL_K) -> dict[str, Any]:
    """The family's numbers over judged cells: pooled, by layer and by site kind. The headline is
    `hallucination_rate` = hallucinated / specific readouts (unrevoked; lower is better), always
    read beside `assert_share` (a lens that commits to nothing scores 0). `k` is the arm's
    readouts per cell (1 for a top-k token lens); a cell with fewer counts in `n_short_cells`."""
    return {
        **_block(cells, k),
        "by_layer": _split(cells, lambda c: c.layer, k),
        "by_site_kind": _split(cells, lambda c: c.site_kind, k),
    }


def score(
    args: JudgeArgs,
    scope: list[dict],
    records: list[CellRecord],
    rows: list[dict],
    *,
    counts: dict,
    config: dict,
    k: int,
) -> FamilyResult:
    nums = numbers(records, k=k)
    extras = {name: v for name, v in nums.items() if name not in ("hallucination_rate", "ci95")}
    ci = nums["ci95"]
    # layers= is NOT a subset for this family: a single-layer lens judged at its one layer is
    # complete, as in the source; items= / limit= still are.
    subset = args.items is not None or args.limit > 0
    return FamilyResult(
        family="hallucination",
        metric="hallucination_rate",
        value=nums["hallucination_rate"] if not args.dry_run else None,
        ci95=(ci[0], ci[1]) if ci is not None and not args.dry_run else None,
        n_items=len(scope),
        higher_is_better=False,
        chance=None,
        chance_label=CHANCE_LABEL,
        complete=completeness(
            pinned=args.judge.pinned,
            subset=subset,
            n_expected=counts["n_expected_cells"],
            n_missing=counts["n_missing_cells"],
            n_unjudged=counts["n_unjudged_cells"],
            n_empty=counts["n_empty_cells"],
        )
        and not args.dry_run,
        pinned_instrument=args.judge.pinned,
        config=config,
        counts=counts,
        extras=extras,
        rows=[] if args.dry_run else rows,
    )
