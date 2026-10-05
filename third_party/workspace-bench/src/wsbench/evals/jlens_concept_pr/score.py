"""jlens_concept_pr scoring: join the stage rows into precision / recall@10 per cell and layer.

Ported from ``scripts/oracle_lens_evals/jlens_pr/score_pr.py`` (``assemble_grids``,
``full_grid``, ``score_one_cell``, ``assemble_support``, ``_block``, ``_reject_block``) over the
unchanged numerics of :mod:`.concept_pr`. Every present cell gets a ``status``:

- ``ok``            judged; an empty text or an empty concept list scores P = 0 / R = 0 (a lens
                    fact, never dropped); a cell whose J-lens top-10 has no content token is
                    ``ok`` with NaN on both axes (no information) and is dropped from the means
- ``missing_a``     the cell HAD text but Stage A never returned
- ``incomplete_b``  some content token has no Stage B grade row
- ``missing_p``     Stage P is missing for this cell: PRECISION is NaN (dropped from the
                    precision mean) but the recall numbers are kept

Headline layer: the one judged layer if there is exactly one, else L44; with several judged
layers and no L44 the headline is ``None`` (value / ci95 ``None``, ``n_items`` 0, incomplete).
"""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from wsbench.mcjudge import is_subset
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult, completeness

from .concept_pr import ItemScore, bootstrap_ci, is_content_token, score_item

M = 10
HEADLINE_LAYER = 44
MAX_REJECT_RATE = 0.05  # JLENS_PR_MAX_REJECT_RATE of the source stage script
CHANCE_LABEL = "shuffled-partner foil precision (measured, see extras.foil)"
FAILURE_STATUSES = ("missing_a", "incomplete_b", "missing_p")


def cell_of(key: str) -> str:
    return key.rsplit(":", 1)[0]


@dataclass(frozen=True)
class CellInput:
    """One present cell of the readouts file with everything the scorer needs."""

    key: str
    id: str
    layer: int
    pos: int
    family: str
    has_text: bool
    n_text_tokens: int
    concepts: list[str] | None  # None = no Stage A result (empty text, or a judge failure)
    tokens: list[str]  # the cell's reference J-lens tokens (decoded, unstripped)
    foil_tokens: list[str] | None  # the derangement partner's tokens (None = no partner)


# ------------------------------------------------------------------ joins (ported)


def assemble_grids(stage_b_rows: Iterable[dict[str, Any]]) -> dict[str, dict[int, list[float]]]:
    """``{cell_key: {token_idx: grades[c]}}`` — keyed by J-lens token index, not positional."""
    per_cell: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for r in stage_b_rows:
        per_cell[cell_of(r["key"])][int(r["token_idx"])] = [float(g) for g in r["grades"]]
    return dict(per_cell)


def full_grid(
    by_idx: dict[int, list[float]], content_idx: Sequence[int]
) -> list[list[float]] | None:
    """Rows in ``content_idx`` order, or ``None`` when any content token has no grade row.

    A partial grid must never be scored: recall's denominator would shrink and precision's
    ``max_t`` would range over fewer tokens (review 2026-09-12).
    """
    if any(i not in by_idx for i in content_idx):
        return None
    return [by_idx[i] for i in content_idx]


def zero_grid(n_content_tokens: int) -> list[list[float]]:
    """The grid of an item with zero concepts: one empty row per content token."""
    return [[] for _ in range(n_content_tokens)]


def score_cell(
    grid: Sequence[Sequence[float]],
    *,
    n_concepts: int,
    n_tokens_total: int,
    support: Sequence[float] | None = None,
) -> ItemScore:
    """``score_item`` at the eval's fixed ``m`` (``support`` = Stage P precision grades)."""
    return score_item(
        grid, n_concepts=n_concepts, n_tokens_total=n_tokens_total, m=M, support=support
    )


def score_one_cell(
    *,
    concepts: list[str] | None,
    had_text: bool,
    by_idx: dict[int, list[float]],
    content_idx: Sequence[int],
    n_tokens_total: int,
    support: Sequence[float] | None = None,
    support_expected: bool = False,
) -> tuple[str, ItemScore]:
    """``(status, ItemScore)`` for one (item, layer) cell — see the module docstring.

    Judge failures never reach ``score_item`` (a partial grid would raise there): they get a
    NaN placeholder that carries only the lens facts (n_concepts, token counts).
    """
    n_content = len(content_idx)
    missing_p = bool(concepts) and support_expected and support is None
    if concepts is None:
        if had_text:  # Stage A never returned for a cell that had text
            return "missing_a", ItemScore(
                math.nan, math.nan, math.nan, 0, n_content, n_tokens_total
            )
        # no verbalizer text at all -> zero concepts, a lens fact
        return "ok", score_cell(zero_grid(n_content), n_concepts=0, n_tokens_total=n_tokens_total)
    if not concepts:  # judged, zero concepts -> P=0, R=0
        return "ok", score_cell(zero_grid(n_content), n_concepts=0, n_tokens_total=n_tokens_total)
    grid = full_grid(by_idx, content_idx)
    if grid is None:
        return "incomplete_b", ItemScore(
            math.nan, math.nan, math.nan, len(concepts), n_content, n_tokens_total
        )
    scored = score_cell(
        grid, n_concepts=len(concepts), n_tokens_total=n_tokens_total, support=support
    )
    if missing_p:
        # The precision judge failed for this cell, but the Stage B grid did not: keep the recall
        # numbers and NaN only precision.
        return "missing_p", replace(scored, precision=math.nan)
    return "ok", scored


def assemble_support(
    p_rows: Iterable[dict[str, Any]], concepts: dict[str, list[str]]
) -> dict[str, list[float]]:
    """Concatenate each cell's Stage P chunks back into one per-concept support list.

    A cell is usable only when its chunks tile the list from offset 0 with no gap or overlap AND
    the total length matches the cell's Stage A concept count (``concepts`` keyed by cell key);
    anything else is left out so the scorer books ``missing_p`` rather than silently scoring a
    partial concept list.
    """
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for r in p_rows:
        by_cell.setdefault(cell_of(r["key"]), []).append(r)
    out: dict[str, list[float]] = {}
    for ck, rows in by_cell.items():
        rows = sorted(rows, key=lambda r: int(r.get("offset", 0)))
        support: list[float] = []
        for r in rows:
            if int(r.get("offset", 0)) != len(support):
                support = []
                break
            support.extend(float(x) for x in r["support"])
        n_want = len(concepts.get(ck, []))
        if support and len(support) == n_want:
            out[ck] = support
    return out


def reject_block(ok_keys: set[str], rej_keys: set[str]) -> dict[str, Any]:
    """Unique keys still missing (rejected and never re-collected) over unique requests."""
    still_missing = rej_keys - ok_keys
    n_req = len(ok_keys | rej_keys)
    return {
        "n_requests": n_req,
        "n_missing": len(still_missing),
        "rate": len(still_missing) / n_req if n_req else 0.0,
    }


# ------------------------------------------------------------------ blocks


def _mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return float(np.mean(vals)) if vals else math.nan


def _has_recall(status: str) -> bool:
    """``missing_p`` is a PRECISION failure only — its recall numbers are real."""
    return status in ("ok", "missing_p")


def _num(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _ci(values: Sequence[float]) -> list[float | None] | None:
    lo, hi = bootstrap_ci(values)
    return None if math.isnan(lo) else [float(lo), float(hi)]


Scored = tuple[dict[str, Any], ItemScore]


def block(real: list[Scored], foil: list[Scored]) -> dict[str, Any]:
    scored = [(m, s) for m, s in real if m["status"] == "ok"]
    recalled = [(m, s) for m, s in real if _has_recall(m["status"])]
    p = [s.precision for _, s in scored]
    r = [s.recall_at_m for _, s in recalled]
    n_content = sum(s.n_content_tokens for _, s in real)
    n_tok = sum(s.n_tokens for _, s in real)
    return {
        "precision": _num(_mean(p)),
        "precision_ci": _ci(p),
        "recall_at_10": _num(_mean(r)),
        "recall_at_10_ci": _ci(r),
        "raw_recall": _num(_mean([s.raw_recall for _, s in recalled])),
        "foil_precision": _num(_mean([s.precision for m, s in foil if m["status"] == "ok"])),
        "foil_recall_at_10": _num(
            _mean([s.recall_at_m for m, s in foil if _has_recall(m["status"])])
        ),
        # Stage A succeeded for incomplete_b cells too: their concept / text counts are known
        "mean_n_concepts": _num(
            _mean([float(s.n_concepts) for m, s in real if m["status"] != "missing_a"])
        ),
        "mean_text_tokens": _num(
            _mean([float(m["n_text_tokens"]) for m, _ in real if m["status"] != "missing_a"])
        ),
        "punct_frac": _num(1.0 - n_content / n_tok) if n_tok else None,
        "n_items": len(real),
        "n_items_scored": sum(1 for _, s in scored if not math.isnan(s.precision)),
        "n_items_no_text": sum(1 for m, _ in real if not m["has_text"]),
        # judge failures, EXCLUDED from every mean above (not booked as zeros)
        "n_items_missing_a": sum(1 for m, _ in real if m["status"] == "missing_a"),
        "n_items_missing_p": sum(1 for m, _ in real if m["status"] == "missing_p"),
        "n_items_incomplete_b": sum(1 for m, _ in real if m["status"] == "incomplete_b"),
    }


def headline_layer(layers: Sequence[int]) -> int | None:
    if len(layers) == 1:
        return int(layers[0])
    if HEADLINE_LAYER in layers:
        return HEADLINE_LAYER
    return None


# ------------------------------------------------------------------ the FamilyResult


def score(
    args: JudgeArgs,
    inputs: Sequence[CellInput],
    *,
    layers: Sequence[int],
    grids: dict[str, dict[str, dict[int, list[float]]]],
    support: dict[str, dict[str, list[float]]],
    reject_rate: dict[str, dict[str, Any]],
    counts_base: dict,
    config: dict,
) -> FamilyResult:
    real: dict[int, list[Scored]] = defaultdict(list)
    foil: dict[int, list[Scored]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for ci in inputs:
        content_idx = [i for i, t in enumerate(ci.tokens) if is_content_token(t)]
        status, s = score_one_cell(
            concepts=ci.concepts,
            had_text=ci.has_text,
            by_idx=grids["b"].get(ci.key, {}),
            content_idx=content_idx,
            n_tokens_total=len(ci.tokens),
            support=support["p"].get(ci.key),
            support_expected=bool(content_idx),  # no content token -> no Stage P call -> ok/NaN
        )
        meta = {"status": status, "has_text": ci.has_text, "n_text_tokens": ci.n_text_tokens}
        real[ci.layer].append((meta, s))
        statuses[status] += 1
        row: dict[str, Any] = {
            "key": ci.key,
            "id": ci.id,
            "layer": ci.layer,
            "pos": ci.pos,
            "family": ci.family,
            "has_text": ci.has_text,
            "status": status,
            "n_text_tokens": ci.n_text_tokens,
            "n_concepts": s.n_concepts,
            "n_content_tokens": s.n_content_tokens,
            "n_tokens": s.n_tokens,
            "precision": _num(s.precision),
            "recall_at_10": _num(s.recall_at_m),
            "raw_recall": _num(s.raw_recall),
            "passed": bool(not math.isnan(s.precision) and s.precision >= 0.5),  # display only
            "foil_status": None,
            "foil_precision": None,
            "foil_recall_at_10": None,
        }
        if ci.foil_tokens is not None:
            f_idx = [i for i, t in enumerate(ci.foil_tokens) if is_content_token(t)]
            f_status, fs = score_one_cell(
                concepts=ci.concepts,
                had_text=ci.has_text,
                by_idx=grids["foil"].get(ci.key, {}),
                content_idx=f_idx,
                n_tokens_total=len(ci.foil_tokens),
                support=support["pfoil"].get(ci.key),
                support_expected=bool(f_idx),
            )
            foil[ci.layer].append(({**meta, "status": f_status}, fs))
            row["foil_status"] = f_status
            row["foil_precision"] = _num(fs.precision)
            row["foil_recall_at_10"] = _num(fs.recall_at_m)
        rows.append(row)
    by_layer = {str(L): block(real.get(L, []), foil.get(L, [])) for L in layers}
    hl = headline_layer(layers)
    if hl is None and layers:
        print(
            f"[jlens_concept_pr] several judged layers {list(layers)} without L{HEADLINE_LAYER}: "
            "no headline (value=None, complete=False)"
        )
    blk = by_layer.get(str(hl)) if hl is not None else None
    n_unjudged = sum(statuses[s] for s in FAILURE_STATUSES)
    counts = {**counts_base, "n_unjudged_cells": n_unjudged}
    rejects_ok = all(rb["rate"] <= MAX_REJECT_RATE for rb in reject_rate.values())
    complete = (
        completeness(
            pinned=args.judge.pinned,
            subset=is_subset(args),
            n_expected=counts["n_expected_cells"],
            n_missing=counts["n_missing_cells"],
            n_unjudged=0,  # replaced by the reject-rate clause; counts still report the number
            n_empty=counts["n_empty_cells"],
        )
        and rejects_ok
        and hl is not None
        and not args.dry_run
    )
    extras = {
        "headline_layer": hl,
        "recall_at_10": blk["recall_at_10"] if blk else None,
        "recall_at_10_ci": blk["recall_at_10_ci"] if blk else None,
        "raw_recall": blk["raw_recall"] if blk else None,
        "foil": {
            "recall_at_10": blk["foil_recall_at_10"] if blk else None,
            "precision": blk["foil_precision"] if blk else None,
        },
        "by_layer": by_layer,
        "mean_n_concepts": blk["mean_n_concepts"] if blk else None,
        "punct_frac": blk["punct_frac"] if blk else None,
        "reject_rate": reject_rate,
        "statuses": dict(sorted(statuses.items())),
    }
    ci95 = blk["precision_ci"] if blk else None
    return FamilyResult(
        family="jlens_concept_pr",
        metric="precision",
        value=blk["precision"] if blk and not args.dry_run else None,
        ci95=(ci95[0], ci95[1]) if ci95 and not args.dry_run else None,
        n_items=blk["n_items_scored"] if blk and not args.dry_run else 0,
        higher_is_better=True,
        chance=None,
        chance_label=CHANCE_LABEL,
        complete=complete,
        pinned_instrument=args.judge.pinned,
        config=config,
        counts=counts,
        extras=extras,
        rows=[] if args.dry_run else rows,
    )
