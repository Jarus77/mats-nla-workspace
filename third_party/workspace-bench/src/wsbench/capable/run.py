"""Ask a model its own bank questions and grade the answers: the gate every bank was built with,
re-run on a new model. See AGENTS.md."""

import hashlib
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from wsbench.cache import Cache
from wsbench.judge_config import JudgeConfig, ResolvedJudge, resolve
from wsbench.llm import Spend, schema_block
from wsbench.mcjudge import Call, Preflighter, run_calls

from .questions import PARTIAL, Question, build, threshold_for

PROMPT_VERSION = "capable-2026-09-18"
DEFAULT_DRAWS = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_THRESHOLD = 0.8  # the source repo's >= 8/10 gate; two families override it to 10/10
DEFAULT_REASONING_EFFORT = "minimal"
MAX_TOKENS = 4096  # a reasoning model spends most of this before it answers
GREEDY = -1  # the draw index of the temperature-0 answer every bank gate also required

ANSWER_SYSTEM = (
    "You are answering benchmark questions. Answer the question as asked, following any format "
    "it specifies, with no preamble, no explanation and no restatement of the question. If the "
    "question asks you to continue a text, reply with the continuation only."
)
ANSWER_SCHEMA = schema_block("answer", {"answer": {"type": "string"}}, ["answer"])

GRADE_SYSTEM = (
    "You grade a model's ANSWER against the EXPECTED answer to the same question. Mark it "
    "correct when the answer gives the expected one: the same word or value, an inflection of "
    "it, an established synonym or alias, a faithful translation into another language, or the "
    "same quantity written differently. A longer answer that contains the expected one is "
    "correct; an answer that merely mentions the topic, or names something related but "
    "different, is not. Judge only the answer in front of you."
)
GRADE_SCHEMA = schema_block(
    "grade", {"correct": {"type": "boolean"}, "why": {"type": "string"}}, ["correct", "why"]
)


def render_grade(question: Question, answer: str) -> str:
    expected = "\n".join(f"- {g}" for g in question.golds)
    return f"QUESTION:\n{question.ask}\n\nEXPECTED (any one of):\n{expected}\n\nANSWER:\n{answer}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def min_decided(draws: int, threshold: float) -> int:
    """How many draws must come back before an item's rate is reported. Without a floor, an item
    whose calls nearly all failed passes the gate on its one surviving answer."""
    return max(1, math.ceil(threshold * draws))


def run_family(
    family: str,
    *,
    model: str,
    judge: ResolvedJudge,
    out: Path,
    draws: int = DEFAULT_DRAWS,
    temperature: float = DEFAULT_TEMPERATURE,
    threshold: float | None = None,
    greedy: bool = True,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    limit: int = 0,
    concurrency: int = 64,
    rpm: float = 240.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """``draws`` sampled answers per item plus (by default) one greedy answer, each graded by
    ``judge``; identical answers to one item are graded once. An item passes its family's gate
    when the greedy answer is right AND the sampled rate reaches ``threshold``, the rule the banks
    were built with. Cached under ``out/cells.jsonl``, so a re-run resumes."""
    questions = build(family)[: limit or None]
    bar = threshold_for(family, DEFAULT_THRESHOLD) if threshold is None else threshold
    reasoning = {"effort": reasoning_effort} if reasoning_effort else None
    answerer = resolve(
        JudgeConfig(model=model, prompt_version=PROMPT_VERSION, reasoning=reasoning), flag=model
    )
    spend = Spend()
    pre = Preflighter(dry_run)
    draw_ids = ([GREEDY] if greedy else []) + list(range(draws))
    answers: dict[tuple[str, int], str | None] = {}
    grades: dict[str, bool | None] = {}
    n_empty = 0

    with Cache(out / "cells.jsonl") as cache:
        by_temp = {d: (0.0 if d == GREEDY else temperature) for d in draw_ids}
        replies: dict[str, dict | None] = {}
        for temp in sorted(set(by_temp.values())):
            calls = [
                Call(
                    key=f"{family}|{q.id}|d{d}",
                    system=ANSWER_SYSTEM,
                    user=q.ask,
                    meta={"item": q.id, "draw": d},
                )
                for q in questions
                for d in draw_ids
                if by_temp[d] == temp
            ]
            replies |= run_calls(
                calls,
                schema=ANSWER_SCHEMA,
                judge=answerer,
                prompt_version=PROMPT_VERSION,
                cache=cache,
                spend=spend,
                concurrency=concurrency,
                rpm=rpm,
                dry_run=dry_run,
                preflight=pre.for_judge(answerer),
                temperature=temp,
                max_tokens=MAX_TOKENS,
            )
        if dry_run:
            print(f"[{family}] would grade with {judge.model}; grade prompt for the first item:")
            if questions:
                print(render_grade(questions[0], "<the model's answer>"))
            return {
                "family": family,
                "model": model,
                "dry_run": True,
                "n_items": len(questions),
                "n_calls": len(questions) * len(draw_ids),
            }
        for q in questions:
            for d in draw_ids:
                r = replies.get(f"{family}|{q.id}|d{d}")
                if not isinstance(r, dict):
                    answers[(q.id, d)] = None  # the call failed: no answer to grade
                    continue
                text = str(r.get("answer", "")).strip()
                if not text:
                    n_empty += 1  # an empty answer is a wrong answer, not a failed call
                answers[(q.id, d)] = text

        by_id = {q.id: q for q in questions}
        grade_calls = [
            Call(
                key=f"{family}|{q_id}|g{_digest(text)}",
                system=GRADE_SYSTEM,
                user=render_grade(by_id[q_id], text),
                meta={"item": q_id, "answer": text},
            )
            for q_id, text in sorted(
                {(i, t) for (i, _d), t in answers.items() if t}  # one grade per distinct answer
            )
        ]
        verdicts = run_calls(
            grade_calls,
            schema=GRADE_SCHEMA,
            judge=judge,
            prompt_version=PROMPT_VERSION,
            cache=cache,
            spend=spend,
            concurrency=concurrency,
            rpm=rpm,
            dry_run=False,
            preflight=pre.for_judge(judge),
            temperature=0.0,
        )
        for c in grade_calls:
            v = verdicts.get(c.key)
            grades[c.key] = bool(v["correct"]) if isinstance(v, dict) else None

    def graded(q_id: str, text: str | None) -> bool | None:
        if text is None:
            return None  # the answer call failed
        if not text:
            return False  # empty answer: wrong
        return grades.get(f"{family}|{q_id}|g{_digest(text)}")

    floor = min_decided(draws, bar)
    rows = []
    for q in questions:
        sampled = [answers[(q.id, d)] for d in range(draws)]
        verdict_of = [graded(q.id, t) for t in sampled]
        decided = [g for g in verdict_of if g is not None]
        rate = statistics.fmean(float(g) for g in decided) if decided else None
        greedy_ok = graded(q.id, answers[(q.id, GREEDY)]) if greedy else None
        enough = len(decided) >= floor
        passes = None
        if rate is not None and enough and (greedy_ok is not None or not greedy):
            passes = rate >= bar and (greedy_ok is not False)
        rows.append(
            {
                "id": q.id,
                "golds": q.golds,
                "reference": q.reference,
                "answers": sampled,
                "greedy_answer": answers[(q.id, GREEDY)] if greedy else None,
                "greedy_correct": greedy_ok,
                "n_decided": len(decided),
                "n_correct": sum(decided),
                "rate": rate,
                "passes_gate": passes,
                **({"meta": q.meta} if q.meta else {}),
            }
        )

    decided_rows = [r for r in rows if r["passes_gate"] is not None]
    refs = [r["reference"] for r in rows if isinstance(r["reference"], float)]
    return {
        "family": family,
        "model": model,
        "grader": judge.model,
        "prompt_version": PROMPT_VERSION,
        "draws": draws,
        "greedy": greedy,
        "temperature": temperature,
        "reasoning": answerer.reasoning,
        "threshold": bar,
        "min_decided": floor,
        "partial": PARTIAL.get(family),
        "aggregate": {
            "gate_rate": _mean(float(r["passes_gate"]) for r in decided_rows),
            "mean_accuracy": _mean(r["rate"] for r in rows if r["rate"] is not None),
            "greedy_rate": _mean(
                float(r["greedy_correct"])
                for r in rows
                if greedy and r["greedy_correct"] is not None
            ),
            "reference_rate": _mean(refs) if refs else None,
            "n_items": len(rows),
            "n_items_decided": len(decided_rows),
            "n_answers_failed": sum(
                1
                for r in rows
                for a in (*r["answers"], *([r["greedy_answer"]] if greedy else []))
                if a is None
            ),
            "n_answers_empty": n_empty,
            "n_grades_failed": sum(1 for g in grades.values() if g is None),
            "spend_usd": round(spend.usd, 4),
        },
        "rows": rows,
    }


def _mean(values: Any) -> float | None:
    vals = [float(v) for v in values]
    return statistics.fmean(vals) if vals else None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def report_line(result: dict[str, Any]) -> str:
    a = result["aggregate"]
    return (
        f"{result['family']} on {result['model']}: gate={_fmt(a['gate_rate'])} "
        f"accuracy={_fmt(a['mean_accuracy'])} greedy={_fmt(a['greedy_rate'])} "
        f"bank={_fmt(a['reference_rate'])} n={a['n_items_decided']}/{a['n_items']} "
        f"failed={a['n_answers_failed']} empty={a['n_answers_empty']} "
        f"spend=${a['spend_usd']:.2f}"
    )


def subfamily_rates(result: dict[str, Any], key: str) -> dict[str, float]:
    """Mean accuracy split by a ``meta`` key (buggy code's arm, an mt family's bridge leg)."""
    by: dict[str, list[float]] = {}
    for r in result["rows"]:
        value = (r.get("meta") or {}).get(key)
        if value is not None and r["rate"] is not None:
            by.setdefault(str(value), []).append(r["rate"])
    return {k: statistics.fmean(v) for k, v in sorted(by.items())}


def answer_histogram(result: dict[str, Any], top: int = 5) -> list[tuple[str, int]]:
    """The most common answers across the run: a quick read on a model that refuses or loops."""
    counts: Counter[str] = Counter()
    for r in result["rows"]:
        counts.update(a for a in r["answers"] if a)
    return counts.most_common(top)
