"""moral_rationale scoring: item pass = any correct cell (committed) / both sides (deliberative)."""

from collections import defaultdict
from statistics import mean

from wsbench.mcjudge import is_subset
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness

CHANCE_LABEL = (
    "per call 1/6 (committed), 1/36 (deliberative); any-of-grid floors saturate — do not quote"
)


def _rate(n_pass: int, n: int) -> float | None:
    return n_pass / n if n else None


def score(
    args: JudgeArgs,
    scope: list[dict],
    rows: list[dict],
    *,
    counts: dict,
    config: dict,
    n_api_failed: int,
) -> FamilyResult:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for v in rows:
        by_item[v["id"]].append(v)

    def side_correct(vs: list[dict], side: str) -> bool:
        return any(v["correct"] for v in vs if v["side"] == side)

    committed = [it for it in scope if it["reason_class"] == "committed"]
    delib = [it for it in scope if it["reason_class"] == "deliberative"]
    indicators: list[float] = []
    comm_pass = 0
    yes_any = no_any = both = 0
    for it in scope:
        vs = by_item.get(it["id"], [])
        if it["reason_class"] == "committed":
            ok = any(v["correct"] for v in vs)
            comm_pass += ok
        else:
            y, n = side_correct(vs, "yes"), side_correct(vs, "no")
            yes_any += y
            no_any += n
            ok = y and n
            both += ok
        indicators.append(1.0 if ok else 0.0)

    # pass is ANY over the judged (layer, pos) grid — the honest guessing floor is
    # 1-(5/6)^n_calls per side, not the per-call 1/6 (with a 6x5 grid the naive floor is ~99%).
    #
    # 🚨 DO NOT QUOTE `any_of_grid_floor` FROM THIS FILE AS A FLOOR (2026-09-10). At the
    # last 5 positions over the 11-layer grid this is 55 sites per item, so the value below is
    # ~0.99 (committed) / ~0.99 (deliberative) and puts EVERY arm — and the no-activation
    # baseline — "below chance". The formula assumes a guesser that re-randomises at all ~55
    # sites; the MEASURED guesser (lucky_guessing.json) has a spread of 0.005 over five draws
    # and a majority-vote rate equal to its mean, i.e. it picks the same option every time, so
    # its grid rate IS its per-call rate (0.191 here). It is kept because it is the correct
    # value of a formula and because dropping it would hide the problem; the drawable floors
    # are the measured ones in evals/workspace-bench/baselines/, and workspace_bench.chance
    # (§saturation, is_saturated) suppresses this line everywhere it could be drawn.
    def comm_floor() -> float | None:
        ns = [len(by_item[it["id"]]) for it in committed if by_item.get(it["id"])]
        return round(mean(1 - (5 / 6) ** n for n in ns), 4) if ns else None

    def delib_floor() -> float | None:
        fs = []
        for it in delib:
            vs = by_item.get(it["id"])
            if not vs:
                continue
            ny = sum(1 for v in vs if v["side"] == "yes")
            nn = sum(1 for v in vs if v["side"] == "no")
            fs.append((1 - (5 / 6) ** ny) * (1 - (5 / 6) ** nn))
        return round(mean(fs), 4) if fs else None

    n_items = len(scope)
    passes = int(sum(indicators))
    extras = {
        "committed": {
            "n": len(committed),
            "pass": comm_pass,
            "rate": _rate(comm_pass, len(committed)),
        },
        "deliberative": {
            "n": len(delib),
            "both_sides": both,
            "yes_any": yes_any,
            "no_any": no_any,
            "rate": _rate(both, len(delib)),
        },
        "any_of_grid_floor": {"committed": comm_floor(), "deliberative": delib_floor()},
        "short_pool_items": sorted(
            i for i, vs in by_item.items() if any(v["short_pool"] for v in vs)
        ),
        "n_api_failed": n_api_failed,
    }
    return FamilyResult(
        family="moral_rationale",
        metric="pass_rate",
        value=_rate(passes, n_items) if not args.dry_run else None,
        ci95=bootstrap_ci(indicators) if n_items and not args.dry_run else None,
        n_items=n_items,
        higher_is_better=True,
        chance=None,
        chance_label=CHANCE_LABEL,
        complete=completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=counts["n_expected_cells"],
            n_missing=0,
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
