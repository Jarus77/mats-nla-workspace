"""jlens_concept_pr: an AO arm's prose readouts scored against the frozen J-lens top-10.

Port of ``scripts/oracle_lens_evals/jlens_pr/{judge_openrouter,judge_pr,items}.py`` (the judge
of record) onto the shared client and cache. Three judged stages, one ``run_calls`` batch each:

* Stage A (concept split) on the family judge (Gemini 3.8 Flash in this repo; the source ran
  it on DeepSeek-V4-Flash, see the README's instrument note): one call per cell with text,
  ``text = concat_samples(samples)``;
* Stage B (per-token recall grades) on the judge: one call per CONTENT reference token per cell
  with concepts; the ``foil`` pass grades the same concepts against a seeded within-family
  derangement partner's tokens (keys ``:F..``) and runs by default;
* Stage P (per-concept precision, T = 0) on the judge: one call per 60-concept chunk with the
  cell's full content-token set; ``pfoil`` (keys ``:Q..``) only with ``opts=stage_p_foil=1``.

A cell is (label, layer, eval position) — exactly one eval position per label per the manifest.
An ABSENT cell is missing (fatal without ``allow_missing=True``); a PRESENT-but-empty cell scores
as zero concepts ("lens silent") and counts in ``n_empty_cells``. A parser ``ValueError`` on any
stage is a reject: the result is cached as a failure and re-queued by the next run, never
partially scored. Keys are ``f"{cell.key}:A"``, ``:B{ti:02d}`` / ``:F{ti:02d}``,
``:P{ci:03d}`` / ``:Q{ci:03d}``.
"""

import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from wsbench import registry
from wsbench.cache import Cache
from wsbench.judge_config import ResolvedJudge
from wsbench.llm import Spend
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    item_scope,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import Cell, expected_cells, load_readouts, missing_cells
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult

from . import score
from .concept_pr import is_content_token
from .prompts import (
    PROMPT_VERSION,
    STAGE_A_SCHEMA,
    STAGE_A_SYSTEM,
    STAGE_B_SCHEMA,
    STAGE_B_SYSTEM,
    STAGE_P_SCHEMA,
    STAGE_P_SYSTEM,
    concat_samples,
    parse_stage_a,
    parse_stage_b,
    parse_stage_p,
    render_stage_a,
    render_stage_b,
    render_stage_p,
    stage_p_chunks,
)

FAMILY = "jlens_concept_pr"
HEADLINE_LAYER = 44
FOIL_SEED = 0
MAX_TOKENS = 16000  # every stage: a 60-concept response echoes every concept back
STAGE_P_TEMPERATURE = 0.0
BANK_DIR = registry.REPO_ROOT / "evals" / FAMILY
REF_DIR = BANK_DIR / "gen-jlens-pr-jlens"
STAGES = ("a", "b", "foil", "p", "pfoil")
STAGE_TAG = {"a": "A", "b": "B", "foil": "F", "p": "P", "pfoil": "Q"}


# ------------------------------------------------------------------ items.py (verbatim)


def _bytes_to_unicode() -> dict[str, int]:
    """Inverse of the GPT-2 / Qwen byte-level BPE alphabet (unicode char -> byte)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


_UNICODE_TO_BYTE = _bytes_to_unicode()


def bpe_display_to_text(tok: str) -> str:
    """Decode a J-lens unit ``samples`` token (byte-level BPE display form) to readable text.

    ``jlens_eval`` writes ``convert_ids_to_tokens`` strings with ``Ġ``/``▁`` -> space already
    applied, so CJK / accented tokens arrive as mojibake (``çļĦç»ĵæŀľ`` = 的结果). Every char
    in the byte alphabet maps back to its byte; anything else (the substituted space) is kept.
    """
    out = bytearray()
    for ch in tok:
        b = _UNICODE_TO_BYTE.get(ch)
        if b is None:
            out.extend(ch.encode("utf-8"))
        else:
            out.append(b)
    return out.decode("utf-8", errors="replace")


def foil_pairing(labels_by_group: Mapping[str, Sequence[str]], *, seed: int) -> dict[str, str]:
    """Seeded derangement inside each group (shuffle, then rotate by one)."""
    out: dict[str, str] = {}
    for gkey in sorted(labels_by_group):
        labels = list(labels_by_group[gkey])
        if len(labels) < 2:
            continue
        rng = random.Random(f"{seed}:{gkey}")
        rng.shuffle(labels)
        for i, lab in enumerate(labels):
            out[lab] = labels[(i + 1) % len(labels)]
    return out


# ------------------------------------------------------------------ bank


def load_manifest() -> dict[str, Any]:
    return json.loads((BANK_DIR / "manifest.json").read_text(encoding="utf-8"))


def manifest_items(man: dict[str, Any]) -> list[dict[str, Any]]:
    """``{id, family, pos}`` per label (``id`` = label, ``pos`` = the one eval position)."""
    return [
        {"id": p["label"], "family": p["family"], "pos": int(p["eval_positions"][0])}
        for p in man["prompts"]
    ]


def reference_path(label: str, layer: int) -> Path:
    return REF_DIR / label / f"L{layer:03d}.jsonl"


def reference_tokens(label: str, layer: int, pos: int) -> list[str]:
    """The J-lens top-10 display strings at (label, layer, pos), decoded, unstripped
    (punctuation tokens must survive to be classified). ``[]`` when the row is absent."""
    path = reference_path(label, layer)
    for line in path.read_bytes().decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("pos") == pos:
            return [bpe_display_to_text(t) for t in row["samples"]]
    return []


def foil_map(items: Sequence[dict[str, Any]]) -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    for it in items:
        groups.setdefault(it["family"], []).append(it["id"])
    return foil_pairing(groups, seed=FOIL_SEED)


def stage_key(cell_key: str, stage: str, idx: int | None = None) -> str:
    tag = STAGE_TAG[stage]
    if stage == "a":
        return f"{cell_key}:{tag}"
    if stage in ("b", "foil"):
        return f"{cell_key}:{tag}{idx:02d}"
    return f"{cell_key}:{tag}{idx:03d}"


def cell_of(key: str) -> str:
    """Strip the stage suffix: ``<cell key>``."""
    return key.rsplit(":", 1)[0]


def n_text_tokens(text: str) -> int:
    """Whitespace token count of the judged text (the length statistic reported per arm)."""
    return len(text.split())


def extract_judge(args: JudgeArgs) -> ResolvedJudge:
    """Stage A's judge: ``aux_models["extract"]`` if a family ever sets one, else the family
    judge itself (Camila, 2026-09-16: Stage A runs on Gemini 3.8 Flash like Stages B and P)."""
    model = args.aux_models.get("extract")
    if model is None or model == args.judge.model:
        return args.judge
    return ResolvedJudge(
        model=model, reasoning=None, pinned=args.judge.pinned, source=args.judge.source
    )


def _valid(parse: Callable[..., Any], *parse_args: Any) -> bool:
    try:
        parse(*parse_args)
    except (ValueError, TypeError):
        return False
    return True


# ------------------------------------------------------------------ the family run


def run(args: JudgeArgs) -> FamilyResult:
    items = manifest_items(load_manifest())
    scope = item_scope(items, args)
    family_of = {it["id"]: it["family"] for it in items}
    pos_of = {it["id"]: it["pos"] for it in items}
    positions = {it["id"]: [it["pos"]] for it in scope}
    cells, rep = load_readouts(args.readouts, layers=args.layers, positions=positions)
    if rep.kind == "tokens":
        print(f"[{FAMILY}] judges prose readouts only (the J-lens is the reference, not an arm)")
        raise SystemExit(2)
    layers = sorted(args.layers) if args.layers else rep.layers
    bad = sorted({L for L in layers for it in scope if not reference_path(it["id"], L).exists()})
    if bad:
        print(f"[{FAMILY}] no reference J-lens file for layers {bad} under {REF_DIR}")
        raise SystemExit(2)
    expected = expected_cells(positions, layers)
    missing = missing_cells(cells, expected)
    print(
        f"[{FAMILY}] layers={layers} items={len(scope)} expected={len(expected)} "
        f"cells={len(cells)} missing={len(missing)} | skipped rows: {rep.skipped}",
        flush=True,
    )
    if missing:
        print(
            f"[{FAMILY}] missing cells (first 10): "
            + ", ".join(f"{i}__L{layer:03d}__p{p}" for i, layer, p in missing[:10])
        )
        if not args.allow_missing and not args.dry_run:
            raise SystemExit(2)
    fmap = foil_map(items)
    pfoil = args.extra.get("stage_p_foil") == "1"
    ref_cache: dict[tuple[str, int], list[str]] = {}

    def ref(label: str, layer: int) -> list[str]:
        k = (label, layer)
        if k not in ref_cache:
            ref_cache[k] = reference_tokens(label, layer, pos_of[label])
        return ref_cache[k]

    def partner(c: Cell) -> str | None:
        return fmap.get(c.id)

    spend = Spend()
    pre = Preflighter(args.dry_run)
    common: dict[str, Any] = {
        "cache": None,
        "spend": spend,
        "concurrency": args.concurrency,
        "rpm": args.rpm,
        "dry_run": args.dry_run,
    }
    a_judge = extract_judge(args)
    calls: dict[str, list[Call]] = {s: [] for s in STAGES}
    results: dict[str, dict[str, dict | None]] = {s: {} for s in STAGES}
    concepts: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    with Cache(args.out / "cells.jsonl") as cache:
        common["cache"] = cache
        # ---- Stage A: concepts of every cell with text
        for c in cells:
            if c.empty:
                continue
            text = concat_samples(c.samples or ())
            texts[c.key] = text
            calls["a"].append(
                Call(
                    stage_key(c.key, "a"),
                    STAGE_A_SYSTEM,
                    render_stage_a(text),
                    {"cell": c.key, "stage": "a", "n_text_tokens": n_text_tokens(text)},
                )
            )
        results["a"] = run_calls(
            calls["a"],
            schema=STAGE_A_SCHEMA,
            judge=a_judge,
            prompt_version=PROMPT_VERSION,
            preflight=pre.for_judge(a_judge),
            validate=lambda call, r: _valid(parse_stage_a, r),
            max_tokens=MAX_TOKENS,
            **common,
        )
        for call in calls["a"]:
            r = results["a"].get(call.key)
            if r is not None:
                concepts[call.meta["cell"]] = parse_stage_a(r)
        # ---- Stage B + foil: one call per content reference token
        for c in cells:
            cs = concepts.get(c.key)
            if not cs:
                continue
            for stage in ("b", "foil"):
                src = c.id if stage == "b" else partner(c)
                if src is None:
                    continue
                for ti, tok in enumerate(ref(src, c.layer)):
                    if not is_content_token(tok):
                        continue
                    calls[stage].append(
                        Call(
                            stage_key(c.key, stage, ti),
                            STAGE_B_SYSTEM,
                            render_stage_b(tok, cs),
                            {"cell": c.key, "stage": stage, "token_idx": ti, "token": tok},
                        )
                    )
        b_all = calls["b"] + calls["foil"]
        if b_all and not args.dry_run:
            got = run_calls(
                b_all,
                schema=STAGE_B_SCHEMA,
                judge=args.judge,
                prompt_version=PROMPT_VERSION,
                preflight=pre.for_judge(args.judge),
                validate=lambda call, r: _valid(parse_stage_b, r, concepts[call.meta["cell"]]),
                max_tokens=MAX_TOKENS,
                **common,
            )
            for stage in ("b", "foil"):
                results[stage] = {call.key: got.get(call.key) for call in calls[stage]}
        # ---- Stage P (+ pfoil): one call per 60-concept chunk with the full token set
        for c in cells:
            cs = concepts.get(c.key)
            if not cs:
                continue
            for stage in ("p", "pfoil"):
                if stage == "pfoil" and not pfoil:
                    continue
                src = c.id if stage == "p" else partner(c)
                if src is None:
                    continue
                toks = [t for t in ref(src, c.layer) if is_content_token(t)]
                if not toks:
                    continue
                for ci, (offset, part) in enumerate(stage_p_chunks(cs)):
                    calls[stage].append(
                        Call(
                            stage_key(c.key, stage, ci),
                            STAGE_P_SYSTEM,
                            render_stage_p(toks, part),
                            {
                                "cell": c.key,
                                "stage": stage,
                                "chunk": ci,
                                "offset": offset,
                                "concepts": part,
                            },
                        )
                    )
        p_all = calls["p"] + calls["pfoil"]
        if p_all and not args.dry_run:
            got = run_calls(
                p_all,
                schema=STAGE_P_SCHEMA,
                judge=args.judge,
                prompt_version=PROMPT_VERSION,
                preflight=pre.for_judge(args.judge),
                validate=lambda call, r: _valid(parse_stage_p, r, call.meta["concepts"]),
                temperature=STAGE_P_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                **common,
            )
            for stage in ("p", "pfoil"):
                results[stage] = {call.key: got.get(call.key) for call in calls[stage]}

    # ---- collect the stage rows the scorer joins
    grade_rows: dict[str, list[dict[str, Any]]] = {"b": [], "foil": []}
    for stage in ("b", "foil"):
        for call in calls[stage]:
            r = results[stage].get(call.key)
            if r is None:
                continue
            grade_rows[stage].append(
                {
                    "key": call.key,
                    "token_idx": call.meta["token_idx"],
                    "token": call.meta["token"],
                    "grades": parse_stage_b(r, concepts[call.meta["cell"]]),
                }
            )
    support_rows: dict[str, list[dict[str, Any]]] = {"p": [], "pfoil": []}
    for stage in ("p", "pfoil"):
        for call in calls[stage]:
            r = results[stage].get(call.key)
            if r is None:
                continue
            support_rows[stage].append(
                {
                    "key": call.key,
                    "chunk": call.meta["chunk"],
                    "offset": call.meta["offset"],
                    "support": parse_stage_p(r, call.meta["concepts"]),
                }
            )
    reject_rate = {
        stage: score.reject_block(
            {call.key for call in calls[stage] if results[stage].get(call.key) is not None},
            {call.key for call in calls[stage] if results[stage].get(call.key) is None},
        )
        for stage in STAGES
        if calls[stage] or stage in ("a", "b", "foil", "p")
    }
    inputs: list[score.CellInput] = []
    for c in cells:
        p = partner(c)
        inputs.append(
            score.CellInput(
                key=c.key,
                id=c.id,
                layer=c.layer,
                pos=c.pos,
                family=family_of[c.id],
                has_text=not c.empty,
                n_text_tokens=n_text_tokens(texts.get(c.key, "")),
                concepts=concepts.get(c.key),
                tokens=ref(c.id, c.layer),
                foil_tokens=ref(p, c.layer) if p is not None else None,
            )
        )
    counts_base = {
        "n_expected_cells": len(expected),
        "n_missing_cells": len(missing),
        "n_empty_cells": rep.n_empty,
        "skipped_rows": sum(rep.skipped.values()),
        "spend_usd": spend.usd,
    }
    config = base_config(
        args,
        PROMPT_VERSION,
        extract_model=a_judge.model,
        stage_p_temperature=STAGE_P_TEMPERATURE,
        stage_p_foil=pfoil,
        judged_layers=layers,
    )
    return with_readout_count(
        score.score(
            args,
            inputs,
            layers=layers,
            grids={s: score.assemble_grids(grade_rows[s]) for s in ("b", "foil")},
            support={s: score.assemble_support(support_rows[s], concepts) for s in ("p", "pfoil")},
            reject_rate=reject_rate,
            counts_base=counts_base,
            config=config,
        ),
        scope,
        cells,
    )
