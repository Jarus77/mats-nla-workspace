"""Exp 27 helpers (pure, tested): selection of readouts for the blind in-string judge, the judge prompt, verdict
parsing, the append-only judgement cache, Wilson intervals, conditional judge rates, and the table renderers that
merge the judge pass and the exp 23 numbers into the exp 23 analysis tables. The state labels, strict naming regexes
and conditional naming rates come from exp 23's lib23 (imported by sys.path in analyze27.py / judge27.py)."""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

Z95 = 1.96
TRUNC = 1500          # characters of readout shown to the judge (exp 12 used 700 for its batched prompts)
READERS = ("ol", "nla")

PROMPT = """You are judging a short text written by a system that was asked to describe the state of a document at a cursor \
position. You do not see the document, only the text below.

Question: does the text STATE that the current position (the cursor, the end of the text so far, the next token, \
"here") is inside a string literal, that is, inside an unclosed or still-open quoted string? Count only an explicit \
claim about the current position. Do not count: quotation marks that merely appear in the text, strings mentioned as \
a topic or as a data type, a closed or completed string, or a string somewhere else in the document.

Answer on the first line with exactly one word: yes or no.
On the second line write "evidence:" followed by the shortest quote from the text that supports your answer, or \
"evidence: none".

Text:
<<<
{text}
>>>"""


def judge_prompt(readout: str) -> str:
    """The blind prompt: the readout alone (truncated to TRUNC characters), never the document."""
    return PROMPT.format(text=(readout or "")[:TRUNC])


def _truthy(v) -> bool:  # noqa: ANN001
    return v is True or (isinstance(v, int) and not isinstance(v, bool) and v > 0)


def select_judge_items(items: list[dict], texts: dict[str, dict[str, str]], state: str, seed: int,
                       n_neg: int | None = None) -> list[tuple[str, str, str, bool]]:
    """All items where `state` holds plus exactly `n_neg` (default: as many) randomly chosen items where it does
    not, the same negatives for every reader in `texts` ({reader: {id: readout}}), deterministic for `seed`.
    Returns (reader, id, readout, label) tuples."""
    pos = [it["id"] for it in items if _truthy(it["meta"]["labels"][state])]
    neg = [it["id"] for it in items if not _truthy(it["meta"]["labels"][state])]
    n_neg = len(pos) if n_neg is None else n_neg
    if n_neg > len(neg):
        raise ValueError(f"asked for {n_neg} negatives, only {len(neg)} items where {state} does not hold")
    chosen_neg = random.Random(seed).sample(neg, n_neg)
    out = []
    for reader, by_id in texts.items():
        out += [(reader, i, by_id[i], True) for i in pos]
        out += [(reader, i, by_id[i], False) for i in chosen_neg]
    return out


_YESNO = re.compile(r"^(?:answer|verdict|judg(?:e)?ment)?\s*[:\-]?\s*(yes|no)\b", re.IGNORECASE)
_LABELLED = re.compile(r"(?:answer|verdict)\s*[:\-]\s*\**\s*(yes|no)\b", re.IGNORECASE)


def parse_verdict(text: str | None) -> bool | None:
    """yes/no from a judge reply: JSON {"answer": "yes"}, or a leading "Yes." / "no" / "Answer: yes" on the first
    non-empty line, or an "answer: yes|no" label anywhere; None when there is no verdict."""
    if not text or not text.strip():
        return None
    s = text.strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
            for v in (d.values() if isinstance(d, dict) else []):
                if isinstance(v, bool):
                    return v
                if isinstance(v, str) and v.strip().lower() in ("yes", "no"):
                    return v.strip().lower() == "yes"
        except json.JSONDecodeError:
            pass
    first = next(ln for ln in s.splitlines() if ln.strip())
    first = re.sub(r"[*_`#\"'>]+", "", first).strip()
    m = _YESNO.match(first)
    if m:
        return m.group(1).lower() == "yes"
    m = _LABELLED.search(s)
    if m:
        return m.group(1).lower() == "yes"
    return None


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials; (nan, nan) when n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> float | None:
    return None if n == 0 else k / n


def judge_rates(rows: list[dict]) -> dict:
    """Per reader: judged-yes rate when the label holds and when it does not (rows with an unparsed verdict are
    counted in n_unparsed and left out of the judge denominators), Wilson intervals, and the strict-regex rate on
    the same rows (all rows, since the regex does not depend on the judge)."""
    out = {}
    for reader in sorted({r["reader"] for r in rows}, key=lambda r: READERS.index(r) if r in READERS else 99):
        rs = [r for r in rows if r["reader"] == reader]
        parsed = [r for r in rs if r.get("verdict") is not None]
        pos = [r for r in parsed if r["label"]]
        neg = [r for r in parsed if not r["label"]]
        yes_pos, yes_neg = sum(r["verdict"] for r in pos), sum(r["verdict"] for r in neg)
        rx_pos = [bool(r.get("regex")) for r in rs if r["label"]]
        rx_neg = [bool(r.get("regex")) for r in rs if not r["label"]]
        out[reader] = {"n_pos": len(pos), "n_neg": len(neg), "yes_pos": yes_pos, "yes_neg": yes_neg,
                       "rate_pos": _rate(yes_pos, len(pos)), "rate_neg": _rate(yes_neg, len(neg)),
                       "ci_pos": list(wilson(yes_pos, len(pos))), "ci_neg": list(wilson(yes_neg, len(neg))),
                       "regex_rate_pos": _rate(sum(rx_pos), len(rx_pos)), "regex_rate_neg": _rate(sum(rx_neg), len(rx_neg)),
                       "n_unparsed": len(rs) - len(parsed)}
    return out


def cache_key(source: str, reader: str, item_id: str) -> str:
    """exp 23 and exp 27 reuse the same id format (rnd_<kind>_<n>), so the source is part of the key."""
    return f"{source}/{reader}/{item_id}"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return {r["key"]: r for r in rows}


def append_cache(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _f(v: float | None) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def _fci(rate: float | None, ci: list[float]) -> str:
    return "n/a" if rate is None else f"{rate:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def render_judge_table(rates: dict[str, dict]) -> str:
    """rates = {source: judge_rates(...)}; one row per source and reader."""
    lines = ["## Blind judge on the in-string readouts (judge sees the readout only; yes = it states the position is inside a string literal)", "",
             "| source | reader | n in-string / not | judge yes when in string | when not | regex when in string | regex when not | unparsed |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for source, by_reader in rates.items():
        for reader, r in by_reader.items():
            lines.append(f"| {source} | {reader} | {r['n_pos']} / {r['n_neg']} | {_fci(r['rate_pos'], r['ci_pos'])} | "
                         f"{_fci(r['rate_neg'], r['ci_neg'])} | {_f(r['regex_rate_pos'])} | {_f(r['regex_rate_neg'])} | {r['n_unparsed']} |")
    return "\n".join(lines) + "\n"


def _with_ci(rate: float | None, n: int) -> str:
    if rate is None:
        return "n/a"
    lo, hi = wilson(round(rate * n), n)
    return f"{rate:.2f} [{lo:.2f}, {hi:.2f}]"


def render_strict_table(strict: dict, strict23: dict | None) -> str:
    """The exp 23 strict naming table with Wilson intervals on the when-holds cells and, when `strict23` is given, a
    side-by-side column with exp 23's when-holds rates (Oracle Lens / NLA, exp 23's n)."""
    head = "| state | n holds / not | Oracle Lens: named when holds | when not | NLA: named when holds | when not |"
    sep = "|---|---|---:|---:|---:|---:|"
    if strict23 is not None:
        head += " exp 23: OL / NLA when holds (n) |"
        sep += "---:|"
    lines = ["## Naming the state at random read points (strict, state-specific phrasing; share of readouts naming the state when it holds vs when it does not; 95% Wilson)", "", head, sep]
    for st, r in strict.items():
        row = (f"| {st} | {r['ol']['n_true']} / {r['ol']['n_false']} | {_with_ci(r['ol']['named_when_true'], r['ol']['n_true'])} | "
               f"{_f(r['ol']['named_when_false'])} | {_with_ci(r['nla']['named_when_true'], r['nla']['n_true'])} | {_f(r['nla']['named_when_false'])} |")
        if strict23 is not None:
            p = strict23.get(st)
            row += (f" {_f(p['ol']['named_when_true'])} / {_f(p['nla']['named_when_true'])} ({p['ol']['n_true']}) |" if p else " n/a |")
        lines.append(row)
    return "\n".join(lines) + "\n"


def render_carrying_table(carrying: list[dict], carrying23: list[dict] | None) -> str:
    lines = ["## Carrying at random read points (AUROC; readable = probe on the activation; text bag = classifier on the readout alone; "
             "the all-kinds string row is confounded with document kind, the code-only row is the honest number)", "",
             "| kind | state | n yes / no | readable | Oracle Lens readout | NLA readout |" + (" exp 23: readable / OL / NLA |" if carrying23 is not None else ""),
             "|---|---|---|---:|---:|---:|" + ("---:|" if carrying23 is not None else "")]
    prev = {(r["kind"], r["state"]): r for r in (carrying23 or [])}
    for r in carrying:
        row = f"| {r['kind']} | {r['state']} | {r['n_pos']} / {r['n_neg']} | {r['readable']:.2f} | {r['ol_text_bag']:.2f} | {r['nla_text_bag']:.2f} |"
        if carrying23 is not None:
            p = prev.get((r["kind"], r["state"]))
            row += f" {p['readable']:.2f} / {p['ol_text_bag']:.2f} / {p['nla_text_bag']:.2f} |" if p else " n/a |"
        lines.append(row)
    return "\n".join(lines) + "\n"
