"""moral_rationale: 6-way MC per (item, layer, pos, side) over the tail-5 positions."""

import random
from collections import defaultdict
from dataclasses import dataclass

from wsbench.cache import Cache
from wsbench.llm import Spend
from wsbench.mc import CANNOT, classify, join_samples, seed_int, seeded_shuffle
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    base_counts,
    choice_of,
    item_scope,
    load_bank,
    quote_of,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult
from wsbench.summarizer import aux_judge, render_bag, summarize

from . import prompts, score
from .prompts import EC_SEED, PROMPT_VERSION, SCHEMA, SYSTEM

FAMILY = "moral_rationale"
TAIL_POS = 5  # the instrument: last 5 positions per (item, layer)
N_DISTRACTORS = 4

Pool = list[tuple[str, str]]  # (topic_id, reason text)


@dataclass(frozen=True)
class MC:
    side: str
    question: str
    shown: list[str]
    gold_pos: int  # 1-based
    short_pool: bool


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_pools(items: list[dict]) -> tuple[Pool, Pool, Pool]:
    """(committed_pool, yes_pool, no_pool): committed = every item's ``look_for_reasons[0]``;
    yes/no = every ``look_for_reasons`` entry whose ``reasons[].supports`` is yes/no."""
    comm_pool = [(it["topic_id"], it["look_for_reasons"][0]) for it in items]
    yes_pool: Pool = []
    no_pool: Pool = []
    for it in items:
        supm = {r["text"]: r["supports"] for r in it.get("reasons", [])}
        for t in it["look_for_reasons"]:
            s = supm.get(t, "either")
            if s == "yes":
                yes_pool.append((it["topic_id"], t))
            elif s == "no":
                no_pool.append((it["topic_id"], t))
    return comm_pool, yes_pool, no_pool


def sample_distractors(
    pool: Pool, gold: str, topic_id: str, seed_key: str, n: int = N_DISTRACTORS
) -> tuple[list[str], bool]:
    """Up to ``n`` distractors: same ``topic_id``, normalized duplicates and token-Jaccard > 0.6
    vs the gold are excluded; ``short_pool`` flags a draw of fewer than ``n``."""
    gold_tok = set(norm(gold).split())
    seen = {norm(gold)}
    uniq: list[str] = []
    for src_topic, text in pool:  # pool order is fixed (bank order) -> deterministic
        if src_topic == topic_id:
            continue
        tn = norm(text)
        if tn in seen:
            continue
        if jaccard(gold_tok, set(tn.split())) > 0.6:
            continue
        seen.add(tn)
        uniq.append(text)
    random.Random(seed_int(f"{EC_SEED}:{seed_key}")).shuffle(uniq)
    picked = uniq[:n]
    return picked, len(picked) < n


def build_mcs(it: dict, comm_pool: Pool, yes_pool: Pool, no_pool: Pool) -> dict[str, MC]:
    """committed -> {"committed": MC}; deliberative -> {"yes": MC, "no": MC} (a side without a
    tagged look_for reason is omitted; such an item can never pass both sides)."""
    q = it["question"]
    out: dict[str, MC] = {}
    if it["reason_class"] == "deliberative":
        supm = {r["text"]: r["supports"] for r in it.get("reasons", [])}
        for side, pool in (("yes", yes_pool), ("no", no_pool)):
            gold = next((t for t in it["look_for_reasons"] if supm.get(t) == side), None)
            if gold is None:
                continue
            dists, short = sample_distractors(pool, gold, it["topic_id"], f"{it['id']}:{side}")
            shown = [*seeded_shuffle([gold, *dists], f"{EC_SEED}:{it['id']}:{side}:order"), CANNOT]
            out[side] = MC(
                side,
                prompts.render_deliberative(q, side, shown),
                shown,
                shown.index(gold) + 1,
                short,
            )
    else:
        gold = it["look_for_reasons"][0]
        dists, short = sample_distractors(comm_pool, gold, it["topic_id"], f"{it['id']}:committed")
        shown = [*seeded_shuffle([gold, *dists], f"{EC_SEED}:{it['id']}:committed:order"), CANNOT]
        out["committed"] = MC(
            "committed", prompts.render_committed(q, shown), shown, shown.index(gold) + 1, short
        )
    return out


def select_cells(cells: list[Cell]) -> list[Cell]:
    """Per (id, layer) the last :data:`TAIL_POS` positions (sorted by pos)."""
    groups: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for c in cells:
        groups[(c.id, c.layer)].append(c)
    out: list[Cell] = []
    for k in sorted(groups):
        out.extend(sorted(groups[k], key=lambda c: c.pos)[-TAIL_POS:])
    return out


def run(args: JudgeArgs) -> FamilyResult:
    bank = load_bank(FAMILY)
    scope = item_scope(bank, args)
    pools = build_pools(bank)
    mcs = {it["id"]: build_mcs(it, *pools) for it in scope}
    cells, rep = load_readouts(args.readouts, ids=[it["id"] for it in scope], layers=args.layers)
    selected = select_cells(cells)
    nonempty = [c for c in selected if not c.empty]
    spend = Spend()
    pre = Preflighter(args.dry_run)
    n_summary_failed = 0
    with Cache(args.out / "cells.jsonl") as cache:
        if rep.kind == "tokens":
            sjudge = aux_judge(args.judge, args.aux_models, "summarizer")
            summ = summarize(
                {c.key: render_bag(c.tokens or (), c.scores) for c in nonempty},
                judge=sjudge,
                cache=cache,
                spend=spend,
                concurrency=args.concurrency,
                rpm=args.rpm,
                dry_run=args.dry_run,
                preflight=pre.for_judge(sjudge),
            )
            texts = {k: v for k, v in summ.items() if v is not None}
            n_summary_failed = sum(1 for v in summ.values() if v is None)
        else:
            texts = {c.key: join_samples(c.samples or ()) for c in nonempty}
        calls: list[Call] = []
        for c in nonempty:
            if c.key not in texts:
                continue
            for side, mc in mcs[c.id].items():
                calls.append(
                    Call(
                        f"{c.key}:{side}",
                        SYSTEM,
                        prompts.render_user(texts[c.key], mc.question),
                        {
                            "cell": c.key,
                            "id": c.id,
                            "layer": c.layer,
                            "pos": c.pos,
                            "side": side,
                            "gold_pos": mc.gold_pos,
                            "n_options": len(mc.shown),
                            "short_pool": mc.short_pool,
                        },
                    )
                )
        results = (
            run_calls(
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
            )
            if calls
            else {}
        )
    by_id = {it["id"]: it for it in scope}
    rows: list[dict] = []
    n_api_failed = 0
    judged_cells: set[str] = set()
    failed_cells: set[str] = set()
    for c in calls:
        r = results.get(c.key)
        if r is None:
            n_api_failed += 1
            failed_cells.add(c.meta["cell"])
            continue
        judged_cells.add(c.meta["cell"])
        choice = choice_of(r)
        pick = classify(choice, c.meta["gold_pos"], c.meta["n_options"])
        rows.append(
            {
                "key": c.key,
                "id": c.meta["id"],
                "layer": c.meta["layer"],
                "pos": c.meta["pos"],
                "reason_class": by_id[c.meta["id"]]["reason_class"],
                "side": c.meta["side"],
                "choice": choice,
                "gold_pos": c.meta["gold_pos"],
                "n_options": c.meta["n_options"],
                "short_pool": c.meta["short_pool"],
                "pick": pick,
                "correct": pick == "gold",
                "quote": quote_of(r),
            }
        )
    fully_judged = judged_cells - failed_cells
    n_unjudged = sum(1 for c in nonempty if c.key not in fully_judged)
    counts = base_counts(
        n_expected=len(selected),
        n_unjudged=n_unjudged,
        n_empty=len(selected) - len(nonempty),
        skipped_rows=sum(rep.skipped.values()),
        spend=spend,
    )
    return with_readout_count(
        score.score(
            args,
            scope,
            rows,
            counts=counts,
            config=base_config(args, PROMPT_VERSION),
            n_api_failed=n_api_failed + n_summary_failed,
        ),
        scope,
        cells,
    )
