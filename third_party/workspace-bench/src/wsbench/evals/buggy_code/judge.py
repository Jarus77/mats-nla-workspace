"""Buggy code: a blind reader infers what the readouts claim about the program, then an informed
grader scores that inference 1-10 against the ground truth (the bug, or the fact that there is
none). Headline ``score`` = the mean 1-10 score over buggy programs and clean twins alike.
Stages, scores for silent readouts and strata are explained in the family README."""

from collections import defaultdict
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.family import cell_text, fail, mean, rate, require_cells
from wsbench.llm import Spend
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    base_counts,
    is_subset,
    item_scope,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult, bootstrap_ci, completeness

from .prompts import (
    CLAIMS,
    GRADE_SCHEMA,
    GRADE_SYSTEM,
    INFER_SCHEMA,
    INFER_SYSTEM,
    PROMPT_VERSION,
    SILENT_SCORE,
    render_grade,
    render_infer,
)

NAME = "buggy_code"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
CHANCE_LABEL = (
    "a lens that claims nothing about any program (score 1 on every buggy program, 5 on every "
    "clean twin), see extras.silent_floor"
)


def inference(res: dict[str, Any] | None) -> dict[str, Any]:
    """Stage A answer: the claim, the inferred behaviour and the supporting snippet (kept as
    evidence, not enforced)."""
    if res is None:
        return {"judged": False, "claim": None, "inferred": "", "quote": ""}
    claim = str(res.get("claim", "none"))
    if claim not in CLAIMS:
        claim = "none"
    return {
        "judged": True,
        "claim": claim,
        "inferred": str(res.get("inferred", "")).strip(),
        "quote": str(res.get("quote", "")).strip(),
    }


def grade(res: dict[str, Any] | None) -> int | None:
    if res is None:
        return None
    score = res.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return max(1, min(10, round(score)))


def run(args: JudgeArgs) -> FamilyResult:
    header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    read_layer = {it["id"]: int(header["read_cells"][it["lang_group"]]["layer"]) for it in scope}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    # one cell per item: its read layer (or the override), the max-pos row = the EOF anchor
    layer_of = {i: (args.layers[0] if args.layers else read_layer[i]) for i in ids}
    if args.layers and len(args.layers) > 1:
        fail(f"{NAME}: one read layer per item; pass a single layer or none, not {args.layers}")
    groups: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for c in cells:
        groups[(c.id, c.layer)].append(c)
    chosen = {
        i: max(groups[(i, layer_of[i])], key=lambda c: c.pos)
        for i in ids
        if (i, layer_of[i]) in groups
    }
    n_extra = len(cells) - len(chosen)  # rows at other layers or positions
    missing = [i for i in ids if i not in chosen]
    require_cells(NAME, missing, len(ids), args)
    samples: dict[str, list[str]] = {}
    empty: set[str] = set()
    for i, c in chosen.items():
        ss = (
            [cell_text(c)[0]]
            if c.tokens is not None
            else [s for s in (c.samples or ()) if s.strip()]
        )
        if ss:
            samples[i] = ss
        else:
            empty.add(i)

    spend = Spend()
    pre = Preflighter(args.dry_run)
    inferred: dict[str, dict[str, Any]] = {}
    scores: dict[str, int | None] = {}
    with Cache(args.out / "cells.jsonl") as cache:
        # stage A, blind: what do the readouts claim?
        infer_calls = [
            Call(
                key=f"{i}|L{layer_of[i]:03d}|infer",
                system=INFER_SYSTEM,
                user=render_infer(samples[i]),
                meta={"item": i, "layer": layer_of[i], "src": by_id[i]["src"], "stage": "infer"},
            )
            for i in ids
            if i in samples
        ]
        shared: dict[str, Any] = {
            "judge": args.judge,
            "prompt_version": PROMPT_VERSION,
            "cache": cache,
            "spend": spend,
            "concurrency": args.concurrency,
            "rpm": args.rpm,
            "dry_run": args.dry_run,
            "preflight": pre.for_judge(args.judge),
            "temperature": 0.0,
        }
        infer_results = run_calls(infer_calls, schema=INFER_SCHEMA, **shared)
        for i in ids:
            if i in samples:
                inferred[i] = inference(infer_results.get(f"{i}|L{layer_of[i]:03d}|infer"))
        # stage B, informed: how close is the inference to the truth? Silent readouts get the
        # fixed score for their source and make no call.
        grade_calls = [
            Call(
                key=f"{i}|L{layer_of[i]:03d}|grade",
                system=GRADE_SYSTEM,
                user=render_grade(by_id[i], inferred[i]["claim"], inferred[i]["inferred"]),
                meta={"item": i, "layer": layer_of[i], "src": by_id[i]["src"], "stage": "grade"},
            )
            for i in ids
            if i in inferred and inferred[i]["judged"] and inferred[i]["claim"] != "none"
        ]
        grade_results = run_calls(grade_calls, schema=GRADE_SCHEMA, **shared) if grade_calls else {}
        for c in grade_calls:
            scores[c.meta["item"]] = grade(grade_results.get(c.key))

    rows = []
    for i in ids:
        it = by_id[i]
        src = str(it["src"])
        if i in empty:
            v: dict[str, Any] = {
                "judged": True,
                "claim": "none",
                "inferred": "",
                "quote": "",
                "score": SILENT_SCORE[src],
                "empty": True,
            }
        elif i in inferred and inferred[i]["judged"]:
            inf = inferred[i]
            score = SILENT_SCORE[src] if inf["claim"] == "none" else scores.get(i)
            v = {**inf, "score": score, "judged": score is not None}
        else:
            v = {
                "judged": False,
                "claim": None,
                "inferred": "",
                "quote": "",
                "score": None,
                "missing": i in missing,
            }
        rows.append(
            {
                "id": i,
                "src": src,
                "lang_group": it["lang_group"],
                "consequence_class": it.get("consequence_class"),
                "layer": layer_of[i],
                **v,
            }
        )
    judged = [r for r in rows if r["judged"]]
    values = [float(r["score"]) for r in judged]
    value = mean(values)
    n_unjudged = sum(1 for r in rows if not r["judged"] and not r.get("missing"))
    by_src = {s: [r for r in judged if r["src"] == s] for s in ("buggy", "clean")}
    silent_floor = mean(float(SILENT_SCORE[str(it["src"])]) for it in scope)
    strata = sorted(
        {(r["consequence_class"] or "none", r["lang_group"]) for r in judged if r["src"] == "buggy"}
    )
    per_stratum = {
        f"{cls}/{lg}": {
            "n": sum(
                1
                for r in by_src["buggy"]
                if (r["consequence_class"] or "none") == cls and r["lang_group"] == lg
            ),
            "score": mean(
                float(r["score"])
                for r in by_src["buggy"]
                if (r["consequence_class"] or "none") == cls and r["lang_group"] == lg
            ),
        }
        for cls, lg in strata
    }
    hist = {
        s: {str(k): sum(1 for r in rs if r["score"] == k) for k in range(1, 11)}
        for s, rs in by_src.items()
    }
    result = FamilyResult(
        family=NAME,
        metric="score",
        value=value,
        ci95=bootstrap_ci(values) if values else None,
        n_items=len(ids),
        higher_is_better=True,
        chance=silent_floor,
        chance_label=CHANCE_LABEL,
        complete=bool(cells)
        and completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=len(ids),
            n_missing=len(missing),
            n_unjudged=n_unjudged,
            n_empty=len(empty),
        ),
        pinned_instrument=args.judge.pinned,
        config=base_config(
            args,
            PROMPT_VERSION,
            kind=rep.kind or "prose",
            layers_judged=sorted(set(layer_of.values())),
        ),
        # a cell is one item at its read layer; one blind call per non-empty cell plus one
        # graded call per cell that claimed something
        counts=base_counts(
            n_expected=len(ids),
            n_missing=len(missing),
            n_unjudged=n_unjudged,
            n_empty=len(empty),
            skipped_rows=sum(rep.skipped.values()) + n_extra,
            spend=spend,
        ),
        rows=rows,
        extras={
            "n_calls": len(infer_calls) + len(grade_calls),
            "n_rows_not_at_read_cell": n_extra,
            "silent_floor": silent_floor,
            "score_buggy": mean(float(r["score"]) for r in by_src["buggy"]),
            "score_clean": mean(float(r["score"]) for r in by_src["clean"]),
            "n_buggy_judged": len(by_src["buggy"]),
            "n_clean_judged": len(by_src["clean"]),
            "claims": {
                s: {c: rate(r["claim"] == c for r in rs) for c in CLAIMS}
                for s, rs in by_src.items()
            },
            "false_alarm_rate_clean": rate(r["claim"] == "bug" for r in by_src["clean"]),
            "score_hist": hist,
            "per_stratum": per_stratum,
            "rows_claims": [
                {
                    "id": r["id"],
                    "src": r["src"],
                    "claim": r["claim"],
                    "score": r["score"],
                    "inferred": r["inferred"][:300],
                }
                for r in judged
            ],
        },
    )
    return with_readout_count(result, scope, cells)
