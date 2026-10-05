"""Bank judge: one call per (item, layer) through ``mcjudge.run_calls``, verbatim-quote gate."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from wsbench.basic.prompts import PROMPT_VERSION, SCHEMA, SYSTEM, cell_samples, render_user
from wsbench.cache import Cache
from wsbench.judge_config import ResolvedJudge
from wsbench.llm import JudgeConfigError, Spend
from wsbench.mcjudge import Call, Preflighter, run_calls
from wsbench.readouts import Cell, Kind

CellKey = tuple[str, int]  # (item id, layer)


@dataclass(frozen=True)
class Verdict:
    expressed: bool
    target: str
    quote: str
    quote_ok: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "expressed": self.expressed,
            "target": self.target,
            "quote": self.quote,
            "quote_ok": self.quote_ok,
        }


EMPTY_VERDICT = Verdict(expressed=False, target="", quote="", quote_ok=False)


@dataclass
class JudgeOutcome:
    verdicts: dict[CellKey, Verdict]
    n_unjudged: int
    n_empty: int
    n_calls: int


_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _fold(s: str) -> str:
    """Lowercase, decode literal ``\\uXXXX`` escapes, strip accents, drop non-alphanumerics."""
    s = _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), s.lower())
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if c.isalnum() and not unicodedata.combining(c))


def _in_sample(quote: str, sample: str) -> bool:
    if quote in sample:
        return True
    norm_q = " ".join(quote.lower().split())
    if norm_q and norm_q in " ".join(sample.lower().split()):
        return True
    folded_q = _fold(quote)
    return bool(folded_q) and folded_q in _fold(sample)


def quote_verified(quote: str, samples: list[str]) -> bool:
    """A positive's quote must be a span of ONE readout sample: exact, then whitespace- and
    case-normalised, then folded (punctuation, markdown and accents ignored). A quote that only
    exists across two samples never verifies; an empty quote never verifies."""
    if not quote.strip():
        return False
    return any(_in_sample(quote, s) for s in samples)


def _verdict(result: dict[str, Any], samples: list[str]) -> Verdict:
    quote = str(result.get("quote") or "")
    quote_ok = quote_verified(quote, samples)
    return Verdict(
        expressed=bool(result.get("expressed")) and quote_ok,
        target=str(result.get("target") or ""),
        quote=quote,
        quote_ok=quote_ok,
    )


def cell_key(key: CellKey) -> str:
    return f"{key[0]}|L{key[1]}"


def judge_cells(
    groups: dict[CellKey, list[Cell]],
    targets: dict[str, list[str]],
    *,
    kind: Kind,
    judge: ResolvedJudge,
    cache: Cache,
    spend: Spend,
    preflight: Preflighter,
    concurrency: int,
    rpm: float,
    dry_run: bool,
) -> JudgeOutcome:
    """Judge every (item, layer) group. A group with no text is a negative without a call.
    Unjudged cells (failed calls, or every call under ``dry_run``) are counted, never scored;
    a run where every call fails raises ``JudgeConfigError``."""
    verdicts: dict[CellKey, Verdict] = {}
    calls: list[Call] = []
    samples_of: dict[str, list[str]] = {}
    n_empty = 0
    for key, cells in groups.items():
        samples = [s for c in cells for s in cell_samples(c)]
        if not any(s.strip() for s in samples):
            verdicts[key] = EMPTY_VERDICT
            n_empty += 1
            continue
        k = cell_key(key)
        samples_of[k] = samples
        calls.append(Call(key=k, system=SYSTEM, user=render_user(targets[key[0]], kind, cells)))
    results = run_calls(
        calls,
        schema=SCHEMA,
        judge=judge,
        prompt_version=PROMPT_VERSION,
        cache=cache,
        spend=spend,
        concurrency=concurrency,
        rpm=rpm,
        dry_run=dry_run,
        preflight=preflight.for_judge(judge),
        temperature=0.0,
    )
    n_unjudged = 0
    for c in calls:
        r = results.get(c.key)
        if r is None:
            n_unjudged += 1
            continue
        item_id, layer = c.key.rsplit("|L", 1)
        verdicts[(item_id, int(layer))] = _verdict(r, samples_of[c.key])
    if calls and not dry_run and n_unjudged == len(calls):
        raise JudgeConfigError(f"all {len(calls)} judge calls failed; nothing was scored")
    return JudgeOutcome(verdicts, n_unjudged=n_unjudged, n_empty=n_empty, n_calls=len(calls))
