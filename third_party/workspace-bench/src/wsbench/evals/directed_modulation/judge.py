"""directed_modulation: one MC judge call per readout row, same-stratum options, evidence gate."""

import random
import re
from dataclasses import dataclass
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.family import require_cells
from wsbench.llm import Spend
from wsbench.mc import classify, seed_int
from wsbench.mcjudge import Call, Preflighter, item_scope, run_calls, with_readout_count
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult

from . import score
from .prompts import N_DISTRACTORS, PROMPT_VERSION, SCHEMA, SEED, SYSTEM, render_user

BANK = REPO_ROOT / "evals" / "directed_modulation" / "items.json"
_BPE_JUNK = re.compile(r"[ĠĊ▁]")


@dataclass(frozen=True)
class Row:
    """One judgeable readout: a (layer, position, sample) of one item."""

    item_id: str
    layer: int
    pos: int
    sample: int
    readout: str

    @property
    def key(self) -> str:
        return f"{self.item_id}|L{self.layer:03d}|p{self.pos}|s{self.sample}"


def norm_token(t: str) -> str:
    return _BPE_JUNK.sub(" ", t).replace("_", " ").strip()


def _rng(seed: int, key: str) -> random.Random:
    return random.Random(seed_int(f"{seed}:{key}"))


def option_sets(
    items: list[dict[str, Any]], *, seed: int = SEED
) -> dict[str, tuple[list[str], int]]:
    """Per item: ``(shown options, gold 1-based position)``. Distractors are ``N_DISTRACTORS``
    other concepts of the same subfamily, drawn and ordered with a per-item seeded rng, so an
    item's options are identical across rows, runs and arms. The escape line is appended at
    render time, never part of the list."""
    by_group: dict[str, set[str]] = {}
    for it in items:
        by_group.setdefault(str(it["subfamily"]), set()).add(str(it["concept"]))
    out: dict[str, tuple[list[str], int]] = {}
    for it in items:
        gold = str(it["concept"])
        pool = sorted(by_group[str(it["subfamily"])] - {gold})
        picked = _rng(seed, f"draw:{it['name']}").sample(pool, min(N_DISTRACTORS, len(pool)))
        opts = [gold, *picked]
        _rng(seed, f"order:{it['name']}").shuffle(opts)
        out[it["id"]] = (opts, opts.index(gold) + 1)
    return out


def readout_rows(cells: list[Cell]) -> tuple[list[Row], int]:
    """Every non-empty (layer, position, sample) readout, keeping the sample's index in its
    cell; a tokens cell becomes one ``" | "`` bag. Returns the rows and the number of cells whose
    every sample is blank (they score as negatives without a call)."""
    rows: list[Row] = []
    n_empty = 0
    for c in sorted(cells, key=lambda c: (c.id, c.layer, c.pos)):
        if c.tokens is not None:
            toks = [norm_token(t) for t in c.tokens]
            texts = [(0, " | ".join(t for t in toks if t))] if any(toks) else []
        else:
            texts = [(k, s) for k, s in enumerate(c.samples or ()) if s.strip()]
        if not texts:
            n_empty += 1
            continue
        rows.extend(Row(c.id, c.layer, c.pos, k, t) for k, t in texts)
    return rows, n_empty


def missing_cells(ids: list[str], layers: list[int], cells: list[Cell]) -> list[tuple[str, int]]:
    """(item, layer) pairs with no readout row, over the file's own layer set."""
    present = {(c.id, c.layer) for c in cells}
    return [(i, layer) for i in ids for layer in layers if (i, layer) not in present]


def _norm_quote(t: str) -> str:
    return " ".join(t.lower().split())


def decode(
    result: dict[str, Any] | None, options: list[str], gold_pos: int, readout: str
) -> dict[str, Any]:
    """The judge's answer as a verdict: ``pick`` (gold / distractor / cannot_tell), the
    ``basis``, per-candidate ``domain_overlap``, and the evidence gate — a positive (a pick or
    any overlap) whose evidence is not a verbatim span of the readout is voided."""
    n = len(options)
    if result is None:
        return {"judged": False}
    kind = classify(result.get("choice"), gold_pos, n + 1)
    pick = "cannot_tell" if kind in ("invalid", "cannot_tell") else kind
    ov = result.get("domain_overlap")
    overlap = [bool(x) for x in ov][:n] if isinstance(ov, list) else []
    overlap += [False] * (n - len(overlap))
    evidence = str(result.get("evidence") or "").strip()
    positive = pick != "cannot_tell" or any(overlap)
    verified = bool(evidence) and _norm_quote(evidence) in _norm_quote(readout)
    voided = positive and not verified
    if voided:
        pick, overlap = "cannot_tell", [False] * n
    return {
        "judged": True,
        "pick": pick,
        "choice_invalid": kind == "invalid",
        "form": "none" if voided else str(result.get("form") or "none"),
        "basis": "absent" if voided else str(result.get("basis") or "absent"),
        "domain_overlap": overlap,
        "gold_overlap": overlap[gold_pos - 1],
        "distractor_overlap": any(b for i, b in enumerate(overlap) if i != gold_pos - 1),
        "evidence": evidence,
        "evidence_verified": verified if positive else None,
        "voided": voided,
    }


def run(args: JudgeArgs) -> FamilyResult:
    _header, bank = load_bank(BANK)
    scope = item_scope(bank, args)
    by_id = {it["id"]: it for it in scope}
    cells, rep = load_readouts(args.readouts, ids=list(by_id), layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    missing = missing_cells(list(by_id), layers, cells)
    require_cells("directed_modulation", missing, len(by_id) * len(layers), args)
    rows, n_empty = readout_rows(cells)
    tokens = rep.kind == "tokens"
    options = option_sets(bank)  # drawn over the whole bank, never the selected subset
    calls = [
        Call(
            key=r.key,
            system=SYSTEM,
            user=render_user(
                prompt=by_id[r.item_id]["prompt"],
                readout=r.readout,
                options=options[r.item_id][0],
                subfamily=str(by_id[r.item_id]["subfamily"]),
                polarity=str(by_id[r.item_id]["polarity"]),
                tokens=tokens,
            ),
            meta={"item": r.item_id, "layer": r.layer, "pos": r.pos, "sample": r.sample},
        )
        for r in rows
    ]
    spend = Spend()
    with Cache(args.out / "cells.jsonl") as cache:
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
    verdicts = []
    for r in rows:
        opts, gold_pos = options[r.item_id]
        verdicts.append(
            {
                "item": r.item_id,
                "layer": r.layer,
                "pos": r.pos,
                "sample": r.sample,
                "options": opts,
                "gold_position": gold_pos,
                **decode(results.get(r.key), opts, gold_pos, r.readout),
            }
        )
    result = score.score(
        scope,
        verdicts,
        args=args,
        kind=rep.kind or "prose",
        layers=layers,
        n_cells=len(cells),
        missing=missing,
        n_empty=n_empty,
        skipped_rows=sum(rep.skipped.values()),
        spend=spend,
    )
    return with_readout_count(result, scope, cells)
