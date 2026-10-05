"""Chained intermediates: a blind free-recall judge names the value(s) a readout presents as
computed; an item passes when the top-named value is one of its never-written intermediates at
some layer. Floors and the decoy null are explained in the family README."""

import re
from collections import defaultdict
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.family import (
    cell_text,
    fail,
    last_pos_rows,
    mean,
    pass_rate_result,
    quote_in,
    rate,
    require_cells,
    tri_state,
)
from wsbench.llm import Spend
from wsbench.mcjudge import Call, Preflighter, item_scope, run_calls, with_readout_count
from wsbench.readouts import load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult

from .prompts import MAX_VALUES, NEAR, PROMPT_VERSION, SCHEMA, SYSTEM, render_user

NAME = "chain_intermediates"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
CHANCE_LABEL = (
    "no analytic floor (free recall); the measured floors are the magnitude-matched decoy null "
    "(extras.null_top1_near) and the prompt-only baseline"
)


def near_set(gold: list[int], exclude: tuple[int, ...] = ()) -> list[int]:
    """Decoys within +-NEAR of an intermediate, excluding the intermediates themselves and
    ``exclude`` (the item's start and answer, which a lens may echo for other reasons)."""
    out: list[int] = []
    for v in gold:
        out += [
            x
            for x in range(max(1, v - NEAR), v + NEAR + 1)
            if x not in gold and x not in exclude and x not in out
        ]
    return out


def names_number(n: int, text: str) -> bool:
    """``n`` appears in ``text`` as a whole number (not inside 10, 2019 or 10.83)."""
    return re.search(rf"(?<!\d)(?<!\d\.){n}(?!\d)(?!\.\d)", text) is not None


def verdict(
    res: dict[str, Any] | None, gold: list[int], near: list[int], readout: str
) -> dict[str, Any]:
    """One judge answer. ``values`` are the judge's ranked integers capped at three; the top value
    is credited only when it appears in ``readout`` as a whole number or the quote is verbatim in
    it (a digit-only quote must also be a whole number there). ``readout`` is the plain text (a
    token bag without its scores). ``any_of_3`` checks the ranked values against the gold without
    verifying the lower ranks: a diagnostic, not a pass rule."""
    if res is None:
        return {"judged": False, "kind": "unavailable", "values": [], "hit": False}
    raw = res.get("values")
    values = (
        [int(v) for v in raw if isinstance(v, int) and not isinstance(v, bool)][:MAX_VALUES]
        if isinstance(raw, list)
        else []
    )
    if not values:
        return {
            "judged": True,
            "kind": "none",
            "values": [],
            "hit": False,
            "basis": res.get("basis"),
        }
    top = values[0]
    quote = str(res.get("quote", "")).strip()
    quote_ok = names_number(int(quote), readout) if quote.isdigit() else quote_in(quote, readout)
    verified = names_number(top, readout) or quote_ok
    if not verified:
        kind = "unverified"
    elif top in gold:
        kind = "hit"
    elif top in near:
        kind = "near"
    else:
        kind = "other"
    return {
        "judged": True,
        "kind": kind,
        "values": values,
        "hit": kind == "hit",
        "any_of_3": verified and any(v in gold for v in values),
        "basis": res.get("basis"),
        "quote": quote,
    }


def run(args: JudgeArgs) -> FamilyResult:
    _header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    mode = args.extra.get("cells", "last")
    if mode not in ("last", "all"):
        fail(f"{NAME}: opts=cells must be last or all, not {mode!r}")
    if mode == "last":
        # the read cell is the last prompt token: the max-pos row per (item, layer)
        chosen, n_extra_rows = last_pos_rows(cells)
    else:
        # every row in the file is a cell; the producer chose the span (an any-cell rule)
        chosen, n_extra_rows = {(c.id, c.layer, c.pos): c for c in cells}, 0
    seen = {(k[0], k[1]) for k in chosen}
    missing = [(i, layer) for i in ids for layer in layers if (i, layer) not in seen]
    require_cells(NAME, missing, len(ids) * len(layers), args)
    texts, plain, empty = {}, {}, set()
    for k, c in chosen.items():
        judge_text, bare = cell_text(c)
        if judge_text.strip():
            texts[k], plain[k] = judge_text, bare
        else:
            empty.add(k)
    gold = {it["id"]: [int(v) for v in it["intermediates"]] for it in scope}
    near = {
        i: near_set(g, (int(by_id[i]["start"]), int(by_id[i]["answer"]))) for i, g in gold.items()
    }

    spend = Spend()
    with Cache(args.out / "cells.jsonl") as cache:
        calls = [
            Call(
                key=f"{k[0]}|L{k[1]:03d}" + (f"|p{k[2]}" if len(k) == 3 else ""),
                system=SYSTEM,
                user=render_user(text),
                meta={"item": k[0], "layer": k[1], "pos": k[2] if len(k) == 3 else None},
            )
            for k, text in sorted(texts.items())
        ]
        results = run_calls(
            calls,
            schema=SCHEMA,
            judge=args.judge,
            prompt_version=PROMPT_VERSION,
            cache=cache,
            spend=spend,
            concurrency=args.concurrency,
            rpm=args.rpm,
            dry_run=args.dry_run,
            preflight=Preflighter(args.dry_run).for_judge(args.judge),
            temperature=0.0,
        )

    verdicts = [
        {
            "item": c.meta["item"],
            "layer": c.meta["layer"],
            "pos": c.meta["pos"],
            **verdict(
                results.get(c.key),
                gold[c.meta["item"]],
                near[c.meta["item"]],
                plain[
                    (c.meta["item"], c.meta["layer"])
                    if c.meta["pos"] is None
                    else (c.meta["item"], c.meta["layer"], c.meta["pos"])
                ],
            ),
        }
        for c in calls
    ]
    rows = [_item_row(by_id[i], gold[i], near[i], verdicts, missing, empty) for i in ids]
    decided = [r for r in rows if r["pass"] is not None]
    judged = [v for v in verdicts if v["judged"]]
    kinds: dict[str, int] = defaultdict(int)
    basis: dict[str, int] = defaultdict(int)
    for v in judged:
        kinds[v["kind"]] += 1
        basis[str(v.get("basis"))] += 1
    result = pass_rate_result(
        name=NAME,
        args=args,
        prompt_version=PROMPT_VERSION,
        rows=rows,
        cells=cells,
        rep=rep,
        n_expected=len(ids) * len(layers),
        n_missing=len(missing),
        n_unjudged=len({(v["item"], v["layer"], v["pos"]) for v in verdicts if not v["judged"]}),
        n_empty=len(empty),
        spend=spend,
        chance_label=CHANCE_LABEL,
        skipped_extra=n_extra_rows,
        config_extra={"layers_judged": layers, "cells": mode},
        extras={
            "n_calls": len(calls),
            "n_rows_not_last_token": n_extra_rows,
            "cells_per_item": len(chosen) / max(1, len(ids)),
            "per_layer_hit_rate": {
                str(L): rate(bool(v["hit"]) for v in judged if v["layer"] == L) for L in layers
            },
            # mean over decided items (the source averaged every item with a null)
            "null_top1_near": mean(
                r["null_top1_near"] for r in decided if r["null_top1_near"] is not None
            ),
            "any_of_3_rate": rate(r["any_of_3"] for r in decided),
            "committed_rate": rate(bool(v["values"]) for v in judged),
            "kinds": dict(kinds),
            "basis": dict(basis),
            "per_depth": {
                str(d): rate(bool(r["pass"]) for r in decided if r["depth"] == d)
                for d in sorted({r["depth"] for r in decided})
            },
        },
    )
    return with_readout_count(result, scope, cells)


def _item_row(
    item: dict[str, Any],
    gold: list[int],
    near: list[int],
    verdicts: list[dict[str, Any]],
    missing: list[tuple[str, int]],
    empty: set[tuple[str, int]],
) -> dict[str, Any]:
    """``pass`` = some layer whose top-named value is an intermediate (tri-state: undecided when
    no layer hits and a cell is unjudged or missing; an empty cell is a judged negative).
    ``null_top1_near`` is the item's decoy rate: layers whose top value is a decoy within +-NEAR
    of an intermediate, weighted by len(gold)/len(near), over the judged layers."""
    item_id = item["id"]
    mine = sorted((v for v in verdicts if v["item"] == item_id), key=lambda v: v["layer"])
    hits = [v["layer"] for v in mine if v["hit"]]
    incomplete = any(not v["judged"] for v in mine) or any(k[0] == item_id for k in missing)
    judged = [v for v in mine if v["judged"]]
    n_judged = len(judged) + sum(1 for k in empty if k[0] == item_id)
    near_hits = sum(1 for v in judged if v["kind"] == "near")
    return {
        "id": item_id,
        "depth": item.get("depth"),
        "intermediates": gold,
        "answer": item.get("answer"),
        "start": item.get("start"),
        "pass": tri_state(bool(hits), incomplete),
        "earliest_layer": min(hits) if hits else None,
        "hitting_layers": sorted(set(hits)),
        "hits_at": sorted(
            (v["layer"], v["pos"]) for v in mine if v["hit"] and v.get("pos") is not None
        ),
        "any_of_3": any(v.get("any_of_3") for v in mine),
        "null_top1_near": (near_hits * len(gold) / len(near) / n_judged)
        if n_judged and near
        else None,
        "named_by_layer": {str(v["layer"]): v["values"] for v in mine if v.get("pos") is None},
        "kind_by_layer": {str(v["layer"]): v["kind"] for v in mine if v.get("pos") is None},
        "basis_by_layer": {str(v["layer"]): v.get("basis") for v in mine if v.get("pos") is None},
    }
