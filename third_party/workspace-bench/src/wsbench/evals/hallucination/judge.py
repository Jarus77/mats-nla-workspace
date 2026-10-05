"""hallucination (chat): one judge call per (item, layer, read site) cell, labels derived in code.

Port of ``hallucination-bench/src/hallucination_bench/{judge,cli}.py`` onto the shared client and
cache. The judge quotes the spans of each readout that are WRONG about the model's response and
separately the spans that are JUNK; every label is derived here:

* a span must appear verbatim (whitespace-, quote-mark- and case-normalised) in ITS OWN readout,
  otherwise it is dropped and counted; an unknown span type counts as wrong (fail closed);
* readout class: any verified wrong span -> ``hallucinated`` (whatever the judge's kind); else a
  generic / empty kind -> ``generic``; else any junk span -> ``off_topic``; else ``consistent``;
* a hallucinated readout is ``revoked`` when every wrong span it has is also a verified wrong span
  of another readout of the same cell (the lens repeats it: the model's own confusion).

Cells = every row of the readouts file on a bank read site (``items[].sites[].pos``); the
expected grid is every site of every in-scope item x the selected layers, and a missing cell is
fatal (exit 2) unless ``allow_missing=True`` (a dry run only reports it). A prose cell is judged on
its first :data:`HAL_K` non-empty readouts; a ``tokens`` cell is decoded (:func:`decode_token`),
summarised by the shared summarizer and judged as one readout (k = 1). A verdict in which any
readout parses to ``unjudged`` fails validation and is re-queued by the next run.

Stage 2 (claim verification, ``v5c-chat-verify-v1``; ``opts=verify=0`` skips it): one more call
per span-judged cell, the judge GIVEN each readout's verified wrong spans as established false,
listing every other specific claim as true / unverifiable / disputed / off_topic. Every quote is
verified against its own readout; a quote overlapping an established-false span is never
counted twice; ``disputed`` (the judge thinks the response rules it out but the span judge did
not flag it) is folded into ``unverifiable`` and reported beside it. The headline is untouched:
the tallies live in ``extras`` (``n_claims_*``, ``verifiable_share``, ...).
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from wsbench.cache import Cache
from wsbench.llm import Spend
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    item_scope,
    load_bank,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import expected_cells, load_readouts, missing_cells
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult
from wsbench.summarizer import SUMMARIZER_PROMPT_VERSION, aux_judge, render_bag, summarize

from .prompts import (
    JUDGE_SCHEMA,
    PROMPT_VERSION,
    VERIFY_PROMPT_VERSION,
    VERIFY_SCHEMA,
    judge_prompt,
    verify_prompt,
)

FAMILY = "hallucination"
HAL_K = 3  # readouts per cell (a maximum); k = 1 for a summarised top-k token lens
SITE_KINDS: tuple[str, ...] = ("sentence", "clause", "quote", "newline", "markup", "eos")
READOUT_KINDS: tuple[str, ...] = ("specific", "generic", "empty")
SPECIFIC_CLASSES = frozenset({"hallucinated", "off_topic", "consistent"})

_WS = re.compile(r"\s+")
# normalising the typographic forms IS the point here
_QUOTES = str.maketrans(
    {"“": '"', "”": '"', "‘": "'", "’": "'", "«": '"', "»": '"'}  # noqa: RUF001
)


def normalize_ws(s: str) -> str:
    return _WS.sub(" ", s.translate(_QUOTES)).strip().lower()


def verify_quote(quote: str, haystacks: Iterable[str]) -> bool:
    """True when ``quote`` is a verbatim span (whitespace-, quote-mark- and case-normalised, at
    least 3 characters) of one of ``haystacks``."""
    q = normalize_ws(quote)
    if len(q) < 3:
        return False
    return any(q in normalize_ws(h) for h in haystacks)


# ------------------------------------------------------------------ byte-level BPE -> text


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE table (Qwen's tokenizer uses the same): byte -> printable char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


_U2B = {c: b for b, c in _bytes_to_unicode().items()}


def decode_token(tok: str) -> str:
    """Byte-level BPE vocabulary string -> text, char by char: byte-table chars back to their byte,
    anything else (e.g. a literal space) to its own UTF-8 bytes. Pre-decoded non-ASCII text that
    happens to use byte-table characters (Latin-1 letters) is mangled — pass vocabulary strings."""
    raw = b"".join(bytes([_U2B[ch]]) if ch in _U2B else ch.encode("utf-8") for ch in tok)
    return raw.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ verdicts


@dataclass
class ReadoutVerdict:
    idx: int
    kind: str
    wrong: list[str]  # verified wrong spans (verbatim readout text)
    off_topic: list[str]  # verified junk spans
    n_unverified: int  # spans dropped because they are not verbatim in this readout
    cls: str  # hallucinated | off_topic | generic | consistent | unjudged
    revoked: bool = False  # every wrong span is also a verified wrong span of another readout
    claims: dict[str, int] | None = None  # stage-2 tally (see tally_claims); None = not verified


def _readout_class(kind: str, wrong: Sequence[str], off_topic: Sequence[str]) -> str:
    if wrong:
        return "hallucinated"  # whatever the kind: a made-up claim is never "generic"
    if kind in ("generic", "empty"):
        return "generic"  # junk on a non-specific readout does not enter the denominator
    return "off_topic" if off_topic else "consistent"


def parse_verdict(
    raw: dict[str, Any] | None, samples: Sequence[str]
) -> list[ReadoutVerdict] | None:
    """Judge output -> one verdict per readout, or None for an unjudged cell (never zeros).

    Every span must be verbatim in ITS OWN readout (`verify_quote` normalisation) or it is
    dropped and counted; an unknown span type counts as wrong (fail closed)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        return None
    by_idx: dict[int, dict[str, Any]] = {}
    for s in raw["samples"]:
        if not isinstance(s, dict):
            continue
        idx = s.get("idx")
        if isinstance(idx, int | str) and str(idx).strip().lstrip("-").isdigit():
            by_idx[int(idx)] = s
    out: list[ReadoutVerdict] = []
    for i, readout in enumerate(samples):
        s = by_idx.get(i)
        if s is None:
            out.append(ReadoutVerdict(i, "specific", [], [], 0, "unjudged"))
            continue
        kind = str(s.get("kind") or "").strip().lower()
        kind = kind if kind in READOUT_KINDS else "specific"
        wrong: list[str] = []
        off: list[str] = []
        n_unverified = 0
        for sp in s.get("spans") or []:
            if not isinstance(sp, dict):
                continue
            text = str(sp.get("text") or "")
            if not verify_quote(text, [readout]):
                n_unverified += 1
                continue
            is_off = str(sp.get("type") or "").strip().lower() == "off_topic"
            (off if is_off else wrong).append(text)
        out.append(
            ReadoutVerdict(i, kind, wrong, off, n_unverified, _readout_class(kind, wrong, off))
        )
    for v in out:
        if v.cls != "hallucinated":
            continue
        others = {normalize_ws(t) for o in out if o.idx != v.idx for t in o.wrong}
        v.revoked = all(normalize_ws(t) in others for t in v.wrong)
    return out


CLAIM_KEYS: tuple[str, ...] = (
    "false",
    "true",
    "unverifiable",
    "disputed",
    "off_topic",
    "n_unverified",
)


def tally_claims(
    raw: dict[str, Any] | None, samples: Sequence[str], wrong: Sequence[Sequence[str]]
) -> list[dict[str, int]] | None:
    """Stage-2 output -> one tally per readout, or None for an unjudged cell (re-queued).

    Port of the source pass's ``tally``: ``false`` = the established wrong spans given to the
    judge; a quote must be verbatim in ITS OWN readout (``verify_quote``) or it is dropped and
    counted in ``n_unverified``; a verified quote overlapping an established-false span is
    skipped (already false, never double counted); ``disputed`` counts as ``disputed`` AND
    ``unverifiable``; any other or unknown status is ``unverifiable`` (never inflates ``true``).
    Every readout must be present with a ``claims`` list, else the whole cell is unjudged."""
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        return None
    by_idx: dict[int, dict[str, Any]] = {}
    for s in raw["samples"]:
        if isinstance(s, dict) and isinstance(s.get("idx"), int) and not isinstance(s["idx"], bool):
            by_idx[s["idx"]] = s
    if any(
        i not in by_idx or not isinstance(by_idx[i].get("claims"), list)
        for i in range(len(samples))
    ):
        return None
    out: list[dict[str, int]] = []
    for i, readout in enumerate(samples):
        w = [normalize_ws(x) for x in wrong[i]]
        c = dict.fromkeys(CLAIM_KEYS, 0)
        c["false"] = len(wrong[i])
        for cl in by_idx[i]["claims"]:
            if not isinstance(cl, dict):
                continue
            q = str(cl.get("quote") or "")
            if not q or not verify_quote(q, [readout]):
                c["n_unverified"] += 1
                continue
            nq = normalize_ws(q)
            if any(x in nq or nq in x for x in w):
                continue  # already counted as false
            st = str(cl.get("status") or "").strip().lower()
            if st == "true":
                c["true"] += 1
            elif st == "off_topic":
                c["off_topic"] += 1
            elif st == "disputed":
                c["disputed"] += 1
                c["unverifiable"] += 1
            else:  # unverifiable, or anything unexpected -- never inflates true
                c["unverifiable"] += 1
        out.append(c)
    return out


def fully_judged(raw: dict[str, Any] | None, samples: Sequence[str]) -> bool:
    """A verdict counts only if it judged EVERY readout (a partly judged cell is re-judged, never
    scored partially) — the ``validate`` hook of :func:`run_calls`."""
    v = parse_verdict(raw, samples)
    return v is not None and all(x.cls != "unjudged" for x in v)


@dataclass
class CellRecord:
    item: str
    layer: int
    pos: int
    site_kind: str
    n_samples: int
    verdicts: list[ReadoutVerdict] | None  # None = unjudged (incl. a failed token-lens summary)


# ------------------------------------------------------------------ the family run


def _key(id_: str, layer: int, pos: int) -> str:
    return f"{id_}__L{layer:03d}__p{pos}"


def run(args: JudgeArgs) -> FamilyResult:
    from . import score  # score imports the verdict types from here

    bank = load_bank(FAMILY)["items"]
    scope = item_scope(bank, args)
    by_id = {it["id"]: it for it in scope}
    sites = {it["id"]: {int(s["pos"]): s for s in it["sites"]} for it in scope}
    positions = {iid: list(ps) for iid, ps in sites.items()}
    cells, rep = load_readouts(args.readouts, layers=args.layers, positions=positions)
    layers = sorted(args.layers) if args.layers else rep.layers
    expected = expected_cells(positions, layers)
    missing = missing_cells(cells, expected)
    print(
        f"[{FAMILY}] kind={rep.kind} layers={layers} items={len(scope)} expected={len(expected)} "
        f"cells={len(cells)} missing={len(missing)} | skipped rows: {rep.skipped}",
        flush=True,
    )
    if missing:
        print(
            f"[{FAMILY}] missing cells (first 10): "
            + ", ".join(_key(i, layer, p) for i, layer, p in missing[:10])
        )
        if not args.allow_missing and not args.dry_run:
            raise SystemExit(2)
    kind = rep.kind or "prose"
    k_arm = 1 if kind == "tokens" else HAL_K
    verify = args.extra.get("verify", "1") != "0"  # opts=verify=0 skips stage 2
    nonempty = [c for c in cells if not c.empty]
    spend = Spend()
    pre = Preflighter(args.dry_run)
    samples_of: dict[str, list[str]] = {}  # key -> judged readouts (absent = no readout)
    decoded: dict[str, list[str]] = {}
    with Cache(args.out / "cells.jsonl") as cache:
        if kind == "tokens":
            sjudge = aux_judge(args.judge, args.aux_models, "summarizer")
            bundles: dict[str, str] = {}
            for c in nonempty:
                decoded[c.key] = [decode_token(t) for t in c.tokens or ()]
                bundles[c.key] = render_bag(decoded[c.key], c.scores)
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
            for c in nonempty:
                text = summ.get(c.key)
                if text is not None:
                    samples_of[c.key] = [text]  # a failed summary leaves the cell unjudged
        else:
            for c in nonempty:
                samples_of[c.key] = [s for s in c.samples or () if s.strip()][:HAL_K]
        calls: list[Call] = []
        for c in nonempty:
            if c.key not in samples_of:
                continue
            site = sites[c.id][c.pos]
            system, user = judge_prompt(by_id[c.id], site, samples_of[c.key])
            calls.append(
                Call(
                    c.key,
                    system,
                    user,
                    {
                        "id": c.id,
                        "layer": c.layer,
                        "pos": c.pos,
                        "site_kind": site["kind"],
                        "samples": samples_of[c.key],
                    },
                )
            )
        results = (
            run_calls(
                calls,
                schema=JUDGE_SCHEMA,
                judge=args.judge,
                prompt_version=PROMPT_VERSION,
                cache=cache,
                spend=spend,
                concurrency=args.concurrency,
                rpm=args.rpm,
                dry_run=args.dry_run,
                preflight=pre.for_judge(args.judge),
                validate=lambda call, r: fully_judged(r, call.meta["samples"]),
            )
            if calls
            else {}
        )
        verdicts_of: dict[str, list[ReadoutVerdict] | None] = {}
        for c in nonempty:
            samples = samples_of.get(c.key)
            raw = results.get(c.key) if samples is not None else None
            verdicts_of[c.key] = parse_verdict(raw, samples or []) if raw is not None else None
        # ---- stage 2: claim verification, one call per span-judged cell
        vcalls: list[Call] = []
        if verify:
            for c in nonempty:
                verdicts = verdicts_of.get(c.key)
                if verdicts is None:
                    continue
                site = sites[c.id][c.pos]
                wrong = [list(v.wrong) for v in verdicts]
                system, user = verify_prompt(by_id[c.id], site, samples_of[c.key], wrong)
                vcalls.append(
                    Call(
                        f"{c.key}:V",
                        system,
                        user,
                        {
                            "id": c.id,
                            "layer": c.layer,
                            "pos": c.pos,
                            "site_kind": site["kind"],
                            "samples": samples_of[c.key],
                            "wrong": wrong,
                        },
                    )
                )
        vresults = (
            run_calls(
                vcalls,
                schema=VERIFY_SCHEMA,
                judge=args.judge,
                prompt_version=VERIFY_PROMPT_VERSION,
                cache=cache,
                spend=spend,
                concurrency=args.concurrency,
                rpm=args.rpm,
                dry_run=args.dry_run,
                preflight=pre.for_judge(args.judge),
                validate=lambda call, r: (
                    tally_claims(r, call.meta["samples"], call.meta["wrong"]) is not None
                ),
            )
            if vcalls
            else {}
        )
        for vc in vcalls:
            verdicts = verdicts_of[vc.key.removesuffix(":V")]
            tallies = tally_claims(vresults.get(vc.key), vc.meta["samples"], vc.meta["wrong"])
            if verdicts is not None and tallies is not None:
                for v, t in zip(verdicts, tallies, strict=True):
                    v.claims = t
    records: list[CellRecord] = []
    rows: list[dict[str, Any]] = []
    n_unjudged = 0
    for c in nonempty:
        site = sites[c.id][c.pos]
        samples = samples_of.get(c.key)
        verdicts = verdicts_of.get(c.key)
        n_unjudged += verdicts is None
        n_samples = 1 if kind == "tokens" else len(samples or [])
        records.append(CellRecord(c.id, c.layer, c.pos, site["kind"], n_samples, verdicts))
        row: dict[str, Any] = {
            "key": c.key,
            "id": c.id,
            "layer": c.layer,
            "pos": c.pos,
            "site_kind": site["kind"],
            "samples": samples,
            "verdict": None
            if verdicts is None
            else [
                {
                    "idx": v.idx,
                    "kind": v.kind,
                    "class": v.cls,
                    "revoked": v.revoked,
                    "wrong": v.wrong,
                    "off_topic": v.off_topic,
                    "n_unverified": v.n_unverified,
                    "claims": v.claims,
                }
                for v in verdicts
            ],
        }
        if kind == "tokens":
            row["tokens"] = decoded.get(c.key, [])
        rows.append(row)
    counts = {
        "n_expected_cells": len(expected),
        "n_missing_cells": len(missing),
        "n_unjudged_cells": n_unjudged,
        "n_empty_cells": rep.n_empty,
        "skipped_rows": sum(rep.skipped.values()),
        "spend_usd": spend.usd,
    }
    config = base_config(
        args,
        PROMPT_VERSION,
        kind=kind,
        k=k_arm,
        summary_prompt_version=SUMMARIZER_PROMPT_VERSION if kind == "tokens" else None,
        judged_layers=layers,
        verify=verify,
        verify_prompt_version=VERIFY_PROMPT_VERSION if verify else None,
    )
    return with_readout_count(
        score.score(args, scope, records, rows, counts=counts, config=config, k=k_arm),
        scope,
        cells,
    )
