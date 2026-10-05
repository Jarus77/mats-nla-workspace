"""Multi-token basic families. Scorer of record (since 2026-09-23): the regex contract in
``wsbench.multitoken.regex`` — a layer passes when every required unit's form is found in a
sample at that layer, an item at any layer. Prose readouts are scored as they are (no call).
Token-bag readouts are first interpreted by the shared summarizer (``docs/summarizer.md``, one
cached call per cell, the same fingerprint the MC judge used) and the contract is applied to the
interpretation as one more sample beside the raw tokens; for those arms the instrument is the
summarizer model (pinned, spend and a failed summary = undecided item follow it). The
forced-choice Gemini judge that was the instrument from 2026-09-16 stays reachable with
``opts=judge=mc`` (one call per (item, layer, unit); its numbers are diagnostics, never pinned)."""

import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache, fingerprint
from wsbench.family import (
    cell_text,
    fail,
    pass_rate_result,
    quote_in,
    rate,
    require_cells,
    tri_state,
)
from wsbench.judge_config import JudgeConfig, ResolvedJudge
from wsbench.llm import Spend
from wsbench.mc import fold, letter_index  # fold is part of this module's public surface
from wsbench.mcjudge import Call, Preflighter, item_scope, run_calls, with_readout_count
from wsbench.multitoken import regex
from wsbench.multitoken.options import judged_roles, option_sets
from wsbench.multitoken.prompts import LETTERS, PROMPT_VERSION, SCHEMA, SYSTEM, render_user
from wsbench.multitoken.regex import SCORER_VERSION, SUMMARIZED_SCORER_VERSION
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, EvalSpec, JudgeArgs
from wsbench.results import FamilyResult
from wsbench.summarizer import SUMMARIZER_PROMPT_VERSION, aux_judge, render_bag, summarize

__all__ = ["fold", "letter_index", "mt_family", "run_family", "run_mc", "run_regex", "verdict"]

GROUP = "basic_mt"
SCORER = "regex"
CHANCE_LABEL = (
    "no analytic floor for an any-layer conjunction of exact-form matches; the measured floor is "
    "the prompt-only baseline (the MC lucky-guessing floor does not apply to the regex scorer)"
)
MC_CHANCE_LABEL = (
    "no analytic floor for an any-layer conjunction (per call 1/6 per unit); the measured floors "
    "are the lucky-guessing and prompt-only baselines"
)
REGEX_JUDGE = ResolvedJudge(model=SCORER, reasoning=None, pinned=True, source="family")


def mt_family(name: str, title: str, *, calls_per_arm: str) -> EvalSpec:
    return EvalSpec(
        name=name,
        title=title,
        group=GROUP,
        bank=Path("evals") / name / "items.json",
        # the instrument is the regex contract; the JudgeConfig model is what opts=judge=mc uses
        judge=JudgeConfig(prompt_version=SCORER_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=lambda args: run_family(args, name=name),
        calls_per_arm=f"0 (regex); {calls_per_arm} with opts=judge=mc",
        sources="",
        scorer=SCORER,
    )


def run_family(args: JudgeArgs, *, name: str) -> FamilyResult:
    """``opts=judge=regex`` (default) or ``opts=judge=mc``."""
    which = args.extra.get("judge", SCORER)
    if which == SCORER:
        return run_regex(args, name=name)
    if which == "mc":
        return run_mc(args, name=name)
    fail(f"{name}: opts=judge must be 'regex' (default) or 'mc', not {which!r}")


# ---------------------------------------------------------------------------------------------
# the regex contract (scorer of record)
# ---------------------------------------------------------------------------------------------
def _cell_samples(cell: Cell, summary: str | None = None) -> list[str]:
    """One sample per prose sample, or one per top-k token string (as written by the producer)
    plus, when given, the summarizer's interpretation of the bag as ONE more sample; scaffolding
    stripped; blanks dropped. Samples are never joined, so a form found in the summary or in a
    single token hits, and nothing is assembled across them."""
    raw = cell.tokens if cell.tokens is not None else (cell.samples or ())
    out = [regex.extract_phrase(str(s)) for s in raw]
    if summary is not None:
        out.append(regex.extract_phrase(summary))
    return [s for s in out if s]


def _previous_mc(out: Path) -> dict[str, Any] | None:
    """The MC judge's number when ``out/results.json`` still holds one (the regime before this
    scorer, or an ``opts=judge=mc`` run) — carried into ``extras["mc"]`` so the two instruments
    sit side by side without a judge call."""
    path = out / "results.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cfg = d.get("config") or {}
    if not str(cfg.get("prompt_version", "")).startswith("mc-"):
        return None
    return {
        "value": (d.get("numbers") or {}).get("value"),
        "prompt_version": cfg.get("prompt_version"),
        "judge_model": cfg.get("judge_model"),
        "n_items_decided": (d.get("numbers") or {}).get("extras", {}).get("n_items_decided"),
    }


def run_regex(args: JudgeArgs, *, name: str) -> FamilyResult:
    header, items = load_bank(REPO_ROOT / "evals" / name / "items.json")
    contract = regex.contract_for(header, str(header.get("family", name)))
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    try:
        units_of = {it["id"]: regex.scored_units(it, contract) for it in scope}
    except ValueError as e:
        fail(f"{name}: bank violates its own contract: {e}")
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    positions: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for c in cells:
        positions[(c.id, c.layer)].append(c)
    multi = sorted(k for k, cs in positions.items() if len({c.pos for c in cs}) > 1)
    if multi:
        fail(
            f"{name}: {len(multi)} (item, layer) cells carry more than one read position "
            f"(first: {multi[0]}); this family reads the final prompt token only"
        )
    missing = [(i, layer) for i in ids for layer in layers if (i, layer) not in positions]
    require_cells(name, missing, len(ids) * len(layers), args)
    previous_mc = _previous_mc(args.out)
    tokens = rep.kind == "tokens"
    spend = Spend()
    sjudge = aux_judge(args.judge, args.aux_models, "summarizer")
    bundles = (
        {c.key: render_bag(c.tokens or (), c.scores) for c in cells if c.tokens} if tokens else {}
    )
    summaries: dict[str, str | None] = {}
    n_cached = 0
    if bundles:
        # the shared bag -> prose step, same bundle text / cache key / fingerprint as the MC judge
        # used, so every interpretation it already paid for is reused
        pre = Preflighter(args.dry_run)
        with Cache(args.out / "cells.jsonl") as cache:
            n_cached = sum(
                1
                for key, txt in bundles.items()
                if cache.get(
                    f"summ:{key}", fingerprint(SUMMARIZER_PROMPT_VERSION, sjudge.model, txt)
                )
                is not None
            )
            summaries = summarize(
                bundles,
                judge=sjudge,
                cache=cache,
                spend=spend,
                concurrency=args.concurrency,
                rpm=args.rpm,
                dry_run=args.dry_run,
                preflight=pre.for_judge(sjudge),
            )

    if args.dry_run:
        _print_dry_run(name, contract, scope, units_of, positions, layers, missing)

    empty: set[tuple[str, int]] = set()
    unsummarized: set[tuple[str, int]] = set()  # a bag the summarizer failed on: undecided
    hits_by: dict[str, dict[int, dict[str, list[str]]]] = defaultdict(dict)
    source_by: dict[str, dict[int, dict[str, list[str]]]] = defaultdict(dict)
    for (item_id, layer), cs in sorted(positions.items()):
        cell = cs[0]
        summary = summaries.get(cell.key) if (tokens and cell.tokens) else None
        if tokens and cell.tokens and summary is None and not args.dry_run:
            unsummarized.add((item_id, layer))
        samples = _cell_samples(cell, summary)
        if not samples:
            empty.add((item_id, layer))
        if args.dry_run:
            hits_by[item_id][layer] = {}
            continue
        units = units_of[item_id]
        hits_by[item_id][layer] = regex.layer_unit_hits(samples, units)
        if tokens:
            raw_hits = regex.layer_unit_hits(_cell_samples(cell), units)
            summ_hits = (
                regex.layer_unit_hits([regex.extract_phrase(summary)], units)
                if summary is not None
                else {}
            )
            source_by[item_id][layer] = {
                u.role: [
                    src for src, h in (("bag", raw_hits), ("summary", summ_hits)) if h.get(u.role)
                ]
                for u in units
            }

    rows: list[dict[str, Any]] = []
    for item_id in ids:
        units = units_of[item_id]
        res = regex.item_result(hits_by.get(item_id, {}), units)
        item_missing = [layer for i, layer in missing if i == item_id]
        item_unsummarized = sorted(layer for i, layer in unsummarized if i == item_id)
        incomplete = (
            bool(item_missing)
            or bool(item_unsummarized)
            or not hits_by.get(item_id)
            or args.dry_run
        )
        row = {
            "id": item_id,
            "roles": [u.role for u in units if u.required],
            "optional_roles": [u.role for u in units if not u.required],
            "pass": tri_state(res["pass"], incomplete),
            "earliest_layer": res["earliest_layer"],
            "passing_layers": res["passing_layers"],
            "unit_hit": res["unit_hit"],
            "unit_langs": res["unit_langs"],
            "first_lang": res["first_lang"],
            "any_hit": res["any_hit"],
            "missing_layers": item_missing,
            "layers": {
                str(layer): {
                    "hits": h,
                    "empty": (item_id, layer) in empty,
                    **(
                        {"source": source_by[item_id].get(layer, {})}
                        if tokens and not args.dry_run
                        else {}
                    ),
                }
                for layer, h in sorted(hits_by.get(item_id, {}).items())
            },
        }
        if tokens:
            row["unsummarized_layers"] = item_unsummarized
        rows.append(row)
    decided = [r for r in rows if r["pass"] is not None]
    roles = sorted({r for row in rows for r in (*row["roles"], *row["optional_roles"])})
    first_lang: dict[str, Counter[str]] = defaultdict(Counter)
    for row in decided:
        for role, lang in row["first_lang"].items():
            if lang is not None:
                first_lang[role][lang] += 1
    # which source carried the passing layers of token arms: bag only / summary only / both
    pass_source: Counter[str] = Counter()
    if tokens:
        for row in decided:
            for layer in row["passing_layers"]:
                srcs = source_by[row["id"]].get(layer, {})
                req = [srcs.get(r, []) for r in row["roles"]]
                if all("bag" in x for x in req) and all("summary" in x for x in req):
                    pass_source["both"] += 1
                elif all("bag" in x for x in req):
                    pass_source["bag_only"] += 1
                elif all("summary" in x for x in req):
                    pass_source["summary_only"] += 1
                else:
                    pass_source["mixed"] += 1
    # the instrument: the contract alone for prose; the summarizer model for token bags
    judge_used = sjudge if tokens else REGEX_JUDGE
    version = SUMMARIZED_SCORER_VERSION if tokens else SCORER_VERSION
    result = pass_rate_result(
        name=name,
        args=dataclasses.replace(args, judge=judge_used),
        prompt_version=version,
        rows=rows,
        cells=cells,
        rep=rep,
        n_expected=len(ids) * len(layers),
        n_missing=len(missing),
        n_unjudged=len(unsummarized),
        n_empty=len(empty),
        spend=spend,
        chance_label=CHANCE_LABEL,
        config_extra={
            "layers_judged": layers,
            "scorer": SCORER,
            "scorer_version": version,
            "contract": dataclasses.asdict(contract),
            "summarizer": SUMMARIZER_PROMPT_VERSION if tokens else None,
            "summarizer_model": sjudge.model if tokens else None,
        },
        extras={
            "n_calls": spend.calls,
            "any_hit_rate": rate(row["any_hit"] for row in decided),
            "unit_any_layer": {
                r: rate(row["unit_hit"][r] for row in decided if r in row["unit_hit"])
                for r in roles
            },
            "language_of_readout": {r: dict(c) for r, c in sorted(first_lang.items())},
            **(
                {
                    "n_summaries": len(bundles),
                    "n_summaries_cached": n_cached,
                    "n_summaries_failed": len(unsummarized),
                    "passing_layers_by_source": dict(pass_source),
                }
                if tokens
                else {}
            ),
            **({"mc": previous_mc} if previous_mc else {}),
        },
    )
    return with_readout_count(result, scope, cells)


def _print_dry_run(
    name: str,
    contract: regex.BankContract,
    scope: list[dict[str, Any]],
    units_of: dict[str, list[regex.ScoredUnit]],
    positions: dict[tuple[str, int], list[Cell]],
    layers: list[int],
    missing: list[tuple[str, int]],
) -> None:
    print(
        f"[regex] {name}: scorer {SCORER_VERSION}, contract {dataclasses.asdict(contract)}, "
        f"{len(scope)} items x {len(layers)} layers, {len(missing)} missing cells"
    )
    if scope:
        first = scope[0]["id"]
        print(f"[regex] units of {first}:")
        for u in units_of[first]:
            print(f"  {u.role} (required={u.required}): {json.dumps(u.forms, ensure_ascii=False)}")
        key = next((k for k in sorted(positions) if k[0] == first), None)
        if key is not None:
            samples = _cell_samples(positions[key][0])
            print(f"[regex] first cell {key}: {len(samples)} samples, scored -> ", end="")
            print(json.dumps(regex.layer_unit_hits(samples, units_of[first]), ensure_ascii=False))


# ---------------------------------------------------------------------------------------------
# the forced-choice judge (opts=judge=mc): the instrument from 2026-09-16 to 2026-09-23
# ---------------------------------------------------------------------------------------------
def verdict(
    res: dict[str, Any] | None, gold_idx: int, options: list[str], readout: str
) -> dict[str, Any]:
    """One judge answer: ``correct`` / ``distractor`` / ``cannot`` / ``invalid``; a correct pick
    counts only with a verbatim (folded) quote from the readout. ``invalid`` (no lone letter, or
    a letter past the option list) is judged and negative: the instrument answered, wrongly."""
    if res is None:
        return {"judged": False, "kind": "unavailable", "correct": False}
    idx = letter_index(res.get("choice"), options)
    if idx is None:
        return {
            "judged": True,
            "pick": str(res.get("choice", "")).strip() or None,
            "kind": "invalid",
            "correct": False,
            "quote_ok": False,
        }
    letter = LETTERS[idx]
    if idx == len(options) - 1:
        return {
            "judged": True,
            "pick": letter,
            "kind": "cannot",
            "correct": False,
            "quote_ok": False,
        }
    quote = str(res.get("quote", ""))
    ok = quote_in(quote, readout)
    kind = "correct" if idx == gold_idx else "distractor"
    return {
        "judged": True,
        "pick": letter,
        "kind": kind,
        "correct": idx == gold_idx and ok,
        "quote": quote,
        "quote_ok": ok,
    }


def run_mc(args: JudgeArgs, *, name: str) -> FamilyResult:
    """The forced-choice judge, one call per (item, layer, unit). Not the instrument of record:
    the result is never pinned, so it is never ``complete`` and never enters the macro."""
    _header, items = load_bank(REPO_ROOT / "evals" / name / "items.json")
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    options = option_sets(items)  # over the whole bank, never the scored subset
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    positions: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for c in cells:
        positions[(c.id, c.layer)].append(c)
    multi = sorted(k for k, cs in positions.items() if len({c.pos for c in cs}) > 1)
    if multi:
        fail(
            f"{name}: {len(multi)} (item, layer) cells carry more than one read position "
            f"(first: {multi[0]}); this family reads the final prompt token only"
        )
    missing = [(i, layer) for i in ids for layer in layers if (i, layer) not in positions]
    require_cells(name, missing, len(ids) * len(layers), args)

    spend = Spend()
    pre = Preflighter(args.dry_run)
    texts: dict[str, str] = {}
    empty: set[tuple[str, int]] = set()  # (item, layer) cells with nothing to judge: negatives
    with Cache(args.out / "cells.jsonl") as cache:
        if rep.kind == "tokens":
            bundles = {c.key: render_bag(c.tokens or (), c.scores) for c in cells if c.tokens}
            sjudge = aux_judge(args.judge, args.aux_models, "summarizer")
            summ = summarize(
                bundles,
                judge=sjudge,
                cache=cache,
                spend=spend,
                concurrency=args.concurrency,
                rpm=args.rpm,
                dry_run=args.dry_run,
                preflight=pre.for_judge(sjudge),
            )
            texts = {k: v for k, v in summ.items() if v is not None}
            empty = {(c.id, c.layer) for c in cells if not c.tokens}
        else:
            for c in cells:
                text, _plain = cell_text(c)
                if text.strip():
                    texts[c.key] = text
                else:
                    empty.add((c.id, c.layer))
        calls: list[Call] = []
        meta: dict[
            str, tuple[str, int, str, int, list[str], str]
        ] = {}  # key -> item, layer, role, gold, options, readout
        for (item_id, layer), cs in sorted(positions.items()):
            cell = cs[0]
            readout = texts.get(cell.key)
            if readout is None:
                continue
            for role, (opts, gold_idx) in options[item_id].items():
                key = f"{item_id}|L{layer:03d}|{role}"
                meta[key] = (item_id, layer, role, gold_idx, opts, readout)
                calls.append(
                    Call(
                        key=key,
                        system=SYSTEM,
                        user=render_user(readout, role, opts),
                        meta={"item": item_id, "layer": layer, "role": role},
                    )
                )
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
            preflight=pre.for_judge(args.judge),
            temperature=0.0,
        )

    verdicts: list[dict[str, Any]] = []
    for key, (item_id, layer, role, gold_idx, opts, readout) in meta.items():
        v = verdict(results.get(key), gold_idx, opts, readout)
        verdicts.append(
            {
                "item": item_id,
                "layer": layer,
                "role": role,
                "gold": options[item_id][role][0][gold_idx],
                **v,
            }
        )
    unsummarized = {
        (c.id, c.layer) for c in cells if rep.kind == "tokens" and c.tokens and c.key not in texts
    }
    rows = [
        _item_row(i, judged_roles(by_id[i]), verdicts, missing, unsummarized, empty) for i in ids
    ]
    decided = [r for r in rows if r["pass"] is not None]
    unjudged_cells = {(v["item"], v["layer"]) for v in verdicts if not v["judged"]} | unsummarized
    kinds: dict[str, int] = defaultdict(int)
    for v in verdicts:
        kinds[v["kind"]] += 1
    roles = sorted({r for i in ids for r in judged_roles(by_id[i])})
    result = pass_rate_result(
        name=name,
        # a diagnostic instrument: never pinned, so never complete and never in the macro
        args=dataclasses.replace(args, judge=dataclasses.replace(args.judge, pinned=False)),
        prompt_version=PROMPT_VERSION,
        rows=rows,
        cells=cells,
        rep=rep,
        # a cell is one (item, layer) readout at the single read position; a call is one unit
        n_expected=len(ids) * len(layers),
        n_missing=len(missing),
        n_unjudged=len(unjudged_cells),
        n_empty=len(empty),
        spend=spend,
        chance_label=MC_CHANCE_LABEL,
        config_extra={
            "layers_judged": layers,
            "summarizer": SUMMARIZER_PROMPT_VERSION if rep.kind == "tokens" else None,
            "instrument_of_record": SCORER_VERSION,
        },
        extras={
            "n_calls": len(calls),
            "n_unjudged_units": sum(1 for v in verdicts if not v["judged"]),
            "unit_any_layer": {
                r: rate(row["unit_correct"][r] for row in decided if r in row["unit_correct"])
                for r in roles
            },
            "kinds": dict(kinds),
            "abstain_rate": rate(v["kind"] == "cannot" for v in verdicts if v["judged"]),
        },
    )
    return with_readout_count(result, scope, cells)


def _item_row(
    item_id: str,
    roles: list[str],
    verdicts: list[dict[str, Any]],
    missing: list[tuple[str, int]],
    unsummarized: set[tuple[str, int]],
    empty: set[tuple[str, int]],
) -> dict[str, Any]:
    """``pass`` = some layer where every judged unit is correct; None (undecided) when no layer
    passes and a unit verdict is unjudged, a summary failed or a layer is missing. An empty cell
    is a judged negative at that layer (nothing was read), never a reason to be undecided."""
    mine = [v for v in verdicts if v["item"] == item_id]
    by_layer: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for v in mine:
        by_layer[v["layer"]][v["role"]] = v
    for i, layer in sorted(empty):
        if i == item_id:
            by_layer[layer] = {
                r: {"judged": True, "kind": "empty", "correct": False, "quote_ok": False}
                for r in roles
            }
    passing = sorted(
        layer for layer, vs in by_layer.items() if all(vs.get(r, {}).get("correct") for r in roles)
    )
    incomplete = (
        any(not v["judged"] for v in mine)
        or any(i == item_id for i, _l in unsummarized)
        or any(i == item_id for i, _l in missing)
        or not by_layer
    )
    return {
        "id": item_id,
        "roles": roles,
        "pass": tri_state(bool(passing), incomplete),
        "earliest_layer": passing[0] if passing else None,
        "passing_layers": passing,
        "unit_correct": {
            r: any(vs.get(r, {}).get("correct", False) for vs in by_layer.values()) for r in roles
        },
        "layers": {
            str(layer): {
                r: {k: v for k, v in vs[r].items() if k not in ("item", "layer", "role")}
                for r in vs
            }
            for layer, vs in sorted(by_layer.items())
        },
    }
