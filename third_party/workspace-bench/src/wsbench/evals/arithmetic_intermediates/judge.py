"""Arithmetic intermediates: a blind free-recall judge names the values a readout presents as
computed; an item passes when a named value lands within its variant's tolerance of the
never-written intermediate. ``cells=frozen`` (default) judges the variant's one pre-registered
cell per item, one call per cell; ``cells=all`` judges every (layer, position) row in the file
with ONE call per (item, layer) that lists every position as a numbered entry and answers per
entry (Camila, 2026-09-23). Null and per-variant table: the README."""

import re
from collections import defaultdict
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.family import (
    cell_text,
    fail,
    mean,
    pass_rate_result,
    rate,
    require_cells,
    tri_state,
)
from wsbench.llm import Spend
from wsbench.mc import fold
from wsbench.mcjudge import Call, Preflighter, item_scope, run_calls, with_readout_count
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult

from .prompts import (
    MAX_VALUES,
    PROMPT_VERSION,
    PROMPT_VERSION_BATCH,
    SCHEMA,
    SCHEMA_BATCH,
    SYSTEM,
    SYSTEM_BATCH,
    render_batch_user,
    render_user,
)

NAME = "arithmetic_intermediates"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
TOLERANCES = {"exact": 0.0, "rel0.5pct": 0.005, "rel1pct": 0.01, "rel2pct": 0.02, "rel5pct": 0.05}
CHANCE_LABEL = (
    "no analytic floor (free recall); the measured floors are the permutation null over each "
    "item's null set (extras.cross) and the prompt-only baseline"
)
_NUMERAL = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?(?!\d)(?!\.\d)")
# fullwidth digits, full stop, comma, hyphen-minus (U+FF10.., U+FF0E, U+FF0C, U+FF0D) and the
# real minus sign (U+2212) -> ASCII
_FULLWIDTH = str.maketrans(
    "".join(chr(0xFF10 + d) for d in range(10)) + "\uff0e\uff0c\uff0d\u2212", "0123456789.,--"
)
_NUMERIC_QUOTE = re.compile(r"[-\s\d.,]*")  # a quote that is only a number never vouches
_HAS_NUMERAL = re.compile(r"[\d零一二两三四五六七八九十百千万点负]")


def tolerance_ok(got: float, target: float, tol: str) -> bool:
    """The source spec's single numeric match: exact, or within a relative tolerance; sign-aware."""
    if tol == "exact":
        return got == target
    return abs(got - target) <= TOLERANCES[tol] * abs(target)


def quantities(text: str) -> list[float]:
    """The numerals a readout writes (ASCII or fullwidth), with list markers ("2. foo") and step
    labels stripped as the source scorer does; used only to VERIFY that a value the judge names
    is in the text, never to score."""
    t = text.translate(_FULLWIDTH)
    t = re.sub(
        r"(?m)^[ \t]*(?:-[ \t]+)?\d+\.[ \t]+(?=[A-Za-z*_#`(])", " ", t
    )  # "2. foo" / "- 2. foo"
    t = re.sub(r"(?i)step\s*\d+", " ", t)
    out: list[float] = []
    for m in _NUMERAL.finditer(t):
        s = m.group(0).replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def verified(value: float, text: str, quote: str, *, top: bool = True) -> bool:
    """A named value counts when the readout writes it (to six decimals). The judge's quote can
    vouch only for the TOP value (the schema says it supports that one), and only when it is
    verbatim in the readout, is not itself a bare number, and contains a numeral (digits or a
    Chinese numeral such as 负五十一: the route for values the numeral regex cannot read)."""
    if any(abs(q - value) <= 1e-6 * max(1.0, abs(value)) for q in quantities(text)):
        return True
    q = quote.strip().translate(_FULLWIDTH)
    return (
        top
        and bool(q)
        and not _NUMERIC_QUOTE.fullmatch(q)
        and _HAS_NUMERAL.search(q) is not None
        and fold(q) in fold(text.translate(_FULLWIDTH))
    )


def verdict(
    res: dict[str, Any] | None, target: float, tol: str, nulls: list[float], text: str
) -> dict[str, Any]:
    """One judge answer. ``values`` = the judge's ranked numbers, capped at three, kept only when
    verified in the readout. ``hit`` = some kept value within tolerance of the intermediate;
    ``cross`` = the fraction of the item's null set whose intermediate some kept value hits."""
    if res is None:
        return {"judged": False, "kind": "unavailable", "values": [], "hit": False, "cross": None}
    raw = res.get("values")
    named = (
        [float(v) for v in raw if isinstance(v, int | float) and not isinstance(v, bool)][
            :MAX_VALUES
        ]
        if isinstance(raw, list)
        else []
    )
    quote = str(res.get("quote", "")).strip()
    kept = [v for k, v in enumerate(named) if verified(v, text, quote, top=k == 0)]
    hit = any(tolerance_ok(v, target, tol) for v in kept)
    cross = (
        sum(any(tolerance_ok(v, n, tol) for v in kept) for n in nulls) / len(nulls)
        if nulls
        else None
    )
    if not named:
        kind = "none"
    elif not kept:
        kind = "unverified"
    elif hit:
        kind = "hit"
    else:
        kind = "other"
    return {
        "judged": True,
        "kind": kind,
        "values": named,
        "kept": kept,
        "hit": hit,
        "top1_hit": bool(kept) and tolerance_ok(kept[0], target, tol),
        "cross": cross,
        "basis": res.get("basis"),
        "quote": quote,
    }


BANDS = ("exact", "rel5pct")  # the accuracy bands reported beside the variant's own tolerance


def batch_entries(res: dict[str, Any] | None, n: int) -> dict[int, dict[str, Any]]:
    """A batched reply's entries by ``k`` (1..n). The first well-formed entry per k wins; an
    entry with a missing, non-integer or out-of-range k, or without a values list, is dropped,
    and every k without an entry is an unjudged cell."""
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(res, dict) or not isinstance(res.get("entries"), list):
        return out
    for e in res["entries"]:
        if not isinstance(e, dict):
            continue
        k = e.get("k")
        if isinstance(k, bool) or not isinstance(k, int | float) or int(k) != k:
            continue
        k = int(k)
        if 1 <= k <= n and k not in out and isinstance(e.get("values"), list):
            out[k] = e
    return out


def valid_batch(call: Call, res: dict[str, Any]) -> bool:
    """A batched reply with no usable entry at all is re-queued on the next run."""
    return bool(batch_entries(res, int(call.meta["n_entries"])))


def cell_grid(args: JudgeArgs) -> str:
    """``opts=cells=frozen`` (default) reads the variant's one pre-registered cell;
    ``opts=cells=all`` reads every (layer, position) row in the file, any-cell rule."""
    mode = args.extra.get("cells", "frozen")
    if mode not in ("frozen", "all"):
        fail(f"{NAME}: opts=cells must be frozen or all, not {mode!r}")
    return mode


def run(args: JudgeArgs) -> FamilyResult:
    header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    by_name = {it["name"]: it for it in items}
    mode = cell_grid(args)
    if mode == "frozen" and args.layers and len(args.layers) > 1:
        fail(f"{NAME}: one frozen cell per item; pass a single layer or none, not {args.layers}")
    frozen_layer = {
        i: (args.layers[0] if mode == "frozen" and args.layers else int(by_id[i]["cell"]["layer"]))
        for i in ids
    }
    frozen_pos = {i: int(by_id[i]["cell"]["pos"]) for i in ids}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    cells_of: dict[str, list[Cell]] = defaultdict(list)
    n_off_cell = 0
    for c in cells:
        if mode == "all" or (c.layer == frozen_layer[c.id] and c.pos == frozen_pos[c.id]):
            cells_of[c.id].append(c)
        else:
            n_off_cell += 1
    missing = [i for i in ids if not cells_of[i]]
    require_cells(NAME, missing, len(ids), args)
    texts: dict[tuple[str, int, int], str] = {}
    plain: dict[tuple[str, int, int], str] = {}
    tokens: dict[tuple[str, int, int], str | None] = {}
    empty: set[tuple[str, int, int]] = set()
    for i, cs in cells_of.items():
        for c in cs:
            judge_text, bare = cell_text(c)
            key = (i, c.layer, c.pos)
            if judge_text.strip():
                texts[key], plain[key], tokens[key] = judge_text, bare, c.token
            else:
                empty.add(key)
    target = {i: float(by_id[i]["intermediates"][0]) for i in ids}
    tol = {i: str(by_id[i]["tolerance"]) for i in ids}
    nulls = {
        i: [float(by_name[n]["intermediates"][0]) for n in by_id[i]["null_set"] if n in by_name]
        for i in ids
    }

    spend = Spend()
    prompt_version = PROMPT_VERSION_BATCH if mode == "all" else PROMPT_VERSION
    # answer[(item, layer, pos)] -> the judge's per-cell dict (values / basis / quote) or None
    answer: dict[tuple[str, int, int], dict[str, Any] | None] = {}
    with Cache(args.out / "cells.jsonl") as cache:
        common: dict[str, Any] = {
            "judge": args.judge,
            "prompt_version": prompt_version,
            "cache": cache,
            "spend": spend,
            "concurrency": args.concurrency,
            "rpm": args.rpm,
            "dry_run": args.dry_run,
            "preflight": Preflighter(args.dry_run).for_judge(args.judge),
            "temperature": 0.0,
        }
        if mode == "all":
            # one call per (item, layer): every non-empty position of that layer, in position
            # order, as entries [1..n]; the reply is one result per entry
            groups: dict[tuple[str, int], list[int]] = defaultdict(list)
            for i, layer, pos in sorted(texts):
                groups[(i, layer)].append(pos)
            slot: dict[tuple[str, int, int], tuple[str, int]] = {}
            calls = []
            for (i, layer), poss in sorted(groups.items()):
                key = f"{i}|L{layer:03d}|all"
                entries = []
                for k, pos in enumerate(poss, 1):
                    slot[(i, layer, pos)] = (key, k)
                    entries.append((k, pos, tokens[(i, layer, pos)], texts[(i, layer, pos)]))
                calls.append(
                    Call(
                        key=key,
                        system=SYSTEM_BATCH,
                        user=render_batch_user(entries, bag=rep.kind == "tokens"),
                        meta={"item": i, "layer": layer, "n_entries": len(poss), "positions": poss},
                    )
                )
            results = run_calls(calls, schema=SCHEMA_BATCH, validate=valid_batch, **common)
            parsed = {
                c.key: batch_entries(results.get(c.key), int(c.meta["n_entries"])) for c in calls
            }
            for cell_key, (key, k) in slot.items():
                answer[cell_key] = parsed[key].get(k)
        else:
            calls = [
                Call(
                    key=f"{i}|L{layer:03d}|p{pos}",
                    system=SYSTEM,
                    user=render_user(texts[(i, layer, pos)]),
                    meta={"item": i, "layer": layer, "pos": pos},
                )
                for (i, layer, pos) in sorted(texts)
            ]
            results = run_calls(calls, schema=SCHEMA, **common)
            for i, layer, pos in texts:
                answer[(i, layer, pos)] = results.get(f"{i}|L{layer:03d}|p{pos}")

    rows = []
    cell_rows = []
    for i in ids:
        it = by_id[i]
        verdicts = []
        for c in cells_of[i]:
            key = (i, c.layer, c.pos)
            if key in empty:
                v: dict[str, Any] = {
                    "judged": True,
                    "kind": "empty",
                    "values": [],
                    "kept": [],
                    "hit": False,
                    "cross": 0.0 if nulls[i] else None,
                }
            else:
                v = verdict(answer.get(key), target[i], tol[i], nulls[i], plain[key])
            verdicts.append({"layer": c.layer, "pos": c.pos, **v})
            cell_rows.append({"id": i, **verdicts[-1]})
        judged = [v for v in verdicts if v["judged"]]
        unjudged = len(verdicts) - len(judged)
        kept = [x for v in judged for x in v.get("kept", [])]
        hit = any(v["hit"] for v in judged)
        # every band over the same kept values: the variant's own tolerance is the headline
        bands = {b: any(tolerance_ok(x, target[i], b) for x in kept) for b in BANDS}
        cross = (
            sum(any(tolerance_ok(x, n, tol[i]) for x in kept) for n in nulls[i]) / len(nulls[i])
            if nulls[i] and judged
            else None
        )
        hits_at = sorted((v["layer"], v["pos"]) for v in judged if v["hit"])
        kind = (
            (
                "hit"
                if hit
                else "empty"
                if all(v["kind"] == "empty" for v in judged)
                else "none"
                if not kept
                else "other"
            )
            if judged
            else "missing"
        )
        # an item with no hit and an unjudged cell is undecided (a hit could still be there)
        rows.append(
            {
                "id": i,
                "variant": it["variant"],
                "role": it["role"],
                "expr": it["expr"],
                "intermediate": target[i],
                "tolerance": tol[i],
                "layer": frozen_layer[i],
                "pos": frozen_pos[i],
                "n_cells": len(verdicts),
                "n_unjudged": unjudged,
                "pass": tri_state(hit, unjudged > 0 or not verdicts),
                "kind": kind,
                "judged": bool(judged),
                "values": [x for v in judged for x in v.get("values", [])][: MAX_VALUES * 3],
                "kept": kept[: MAX_VALUES * 3],
                "hit": hit,
                "top1_hit": any(v.get("top1_hit") for v in judged),
                "cross": cross,
                "bands": bands,
                "hits_at": hits_at,
                "unjudged_at": sorted((v["layer"], v["pos"]) for v in verdicts if not v["judged"]),
                "earliest_layer": min((L for L, _p in hits_at), default=None),
            }
        )
    decided = [r for r in rows if r["pass"] is not None]
    kinds: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    value_rate = rate(bool(r["pass"]) for r in decided)
    cross_rate = mean(r["cross"] for r in decided if r["cross"] is not None)
    per_variant = {}
    for v in header["variants"]:
        vs = [r for r in decided if r["variant"] == v]
        if not vs:
            continue
        val = rate(bool(r["pass"]) for r in vs)
        crs = mean(r["cross"] for r in vs if r["cross"] is not None)
        per_variant[v] = {
            "n": len(vs),
            "value": val,
            "cross": crs,
            "net": (val - crs) if val is not None and crs is not None else None,
            "tolerance": header["variants"][v]["tolerance"],
            "role": header["variants"][v]["role"],
            "bands": {b: rate(r["bands"][b] for r in vs) for b in BANDS},
        }
    judged_cells = [c for c in cell_rows if c["judged"]]
    layers_seen = sorted({c["layer"] for c in cell_rows})
    result = pass_rate_result(
        name=NAME,
        args=args,
        prompt_version=prompt_version,
        rows=rows,
        cells=cells,
        rep=rep,
        # a cell is one (item, layer, pos) read; one call per non-empty cell
        n_expected=sum(len(v) for v in cells_of.values()) + len(missing),
        n_missing=len(missing),
        n_unjudged=sum(r["n_unjudged"] for r in rows),
        n_empty=len(empty),
        spend=spend,
        chance_label=CHANCE_LABEL,
        skipped_extra=n_off_cell,
        config_extra={
            "cells": mode,
            "packaging": "one call per (item, layer)" if mode == "all" else "one call per cell",
            "layers_judged": layers_seen,
        },
        extras={
            "n_calls": len(calls),
            "n_calls_failed": sum(1 for c in calls if results.get(c.key) is None) if calls else 0,
            "n_rows_off_cell": n_off_cell,
            "cells_per_item": mean(float(r["n_cells"]) for r in rows),
            "cross": cross_rate,
            "net": (value_rate - cross_rate)
            if value_rate is not None and cross_rate is not None
            else None,
            "top1_rate": rate(bool(r.get("top1_hit")) for r in decided),
            "committed_rate": rate(bool(c["values"]) for c in judged_cells),
            # accuracy at fixed bands, any cell: exact and within 5% of the intermediate
            "bands": {b: rate(r["bands"][b] for r in decided) for b in BANDS},
            "cell_hit_rate": rate(bool(c["hit"]) for c in judged_cells),
            "per_layer_hit_rate": {
                str(L): rate(bool(c["hit"]) for c in judged_cells if c["layer"] == L)
                for L in layers_seen
            },
            "kinds": dict(kinds),
            "per_variant": per_variant,
            "per_role": {
                role: rate(bool(r["pass"]) for r in decided if r["role"] == role)
                for role in sorted({r["role"] for r in decided})
            },
        },
    )
    return with_readout_count(result, scope, cells)
