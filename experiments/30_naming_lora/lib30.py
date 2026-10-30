"""Pure helpers for exp 30 (no model): the training target, the grouped cluster split, the statement-accuracy
regex check, the FVE definition reproduced from nla.py, and the small path / line helpers. Everything here runs on
the laptop and in the Modal container alike."""
from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is imported lazily: the laptop `modal run` entrypoint imports this module without it
    import torch

TRAIN_ROWS: tuple[str, ...] = ("py_string", "py_arg", "json_depth", "md_heading", "md_list", "code_lang")
TRANSFER_ROWS: tuple[str, ...] = ("js_string", "latex_math")
ALL_ROWS: tuple[str, ...] = TRAIN_ROWS + TRANSFER_ROWS
CLOSE = "\n</explanation>"            # the AV's native closing; the explanation prefix "<explanation>\n" is the head
SEP = "\n\n"                          # the NLA's explanations are paragraphs separated by a blank line
EMPTY = "(empty)"                     # lib25.explanation_text's marker for a failed extraction
ADAPTER_NAME = "iter_000400"          # the adapter continued in training; peft saves it under this sub-directory

_WS = re.compile(r"\s+")


# ---- targets -----------------------------------------------------------------------------------------------------

def append_statement(explanation: str, statement: str) -> str:
    """``explanation`` with ``statement`` as one final sentence: inserted before ``</explanation>`` when the closing
    tag is present, appended otherwise. The statement gets a period unless it already ends with one."""
    stmt = statement.strip()
    if not stmt.endswith("."):
        stmt += "."
    if "</explanation>" in explanation:
        head, tail = explanation.split("</explanation>", 1)
        return head.rstrip() + SEP + stmt + "\n</explanation>" + tail
    return explanation.rstrip() + SEP + stmt


def training_target(explanation: str, statement: str | None) -> str:
    """The teacher-forced target after the ``<explanation>\\n`` prefix: the explanation body, the statement as a final
    sentence (arm S) or nothing (arm C), then the closing tag. Refuses empty explanations."""
    body = explanation.strip()
    if not body or body == EMPTY:
        raise ValueError("empty explanation cannot be a training target")
    if "</explanation>" in body:
        raise ValueError("explanation must be the tag payload, not the raw completion")
    return (append_statement(body, statement) if statement else body) + CLOSE


# ---- split -------------------------------------------------------------------------------------------------------

def grouped_split(items: Sequence[Mapping], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """(train, eval): clusters shuffled with ``seed`` and assigned whole to train until ``frac`` of the items is
    reached; a cluster never straddles. New lists; the input is untouched. Item order within each part follows the
    input order."""
    clusters = sorted({it["cluster"] for it in items})
    rng = random.Random(seed)
    rng.shuffle(clusters)
    size = {c: sum(it["cluster"] == c for it in items) for c in clusters}
    target = frac * len(items)
    train_clusters: set[str] = set()
    n = 0
    for c in clusters:
        if n >= target:
            break
        train_clusters.add(c)
        n += size[c]
    train = [dict(it) for it in items if it["cluster"] in train_clusters]
    ev = [dict(it) for it in items if it["cluster"] not in train_clusters]
    return train, ev


# ---- statement accuracy ------------------------------------------------------------------------------------------

def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip().lower()


def statement_accuracy(readouts: Mapping[str, str | None], items: Sequence[Mapping], fixed_pair: Mapping[str, str]) -> dict:
    """Does each readout name the item's value with the row's fixed statement? ``fixed_pair`` maps both values of the
    row to their filled statement. Per item: "correct" (the true statement, not the complement), "wrong" (the
    complement, or both), "none" (neither, or no readout). accuracy = correct / n; naming_rate = (correct + wrong) / n."""
    pats = {v: _norm(s) for v, s in fixed_pair.items()}
    per: dict[str, str] = {}
    for it in items:
        text = _norm(readouts.get(it["id"]) or "")
        hits = {v for v, p in pats.items() if p in text}
        per[it["id"]] = "none" if not hits else ("correct" if hits == {it["value"]} else "wrong")
    n = len(items)
    counts = {k: sum(v == k for v in per.values()) for k in ("correct", "wrong", "none")}
    return {"n": n, **counts, "accuracy": counts["correct"] / n if n else float("nan"),
            "naming_rate": (counts["correct"] + counts["wrong"]) / n if n else float("nan"), "per_item": per}



def statement_said_by_state(readouts: Mapping[str, str | None], items: Sequence[Mapping], fixed_pair: Mapping[str, str]) -> dict:
    """Which fixed statement each readout says, split by the item's true value. Returns {true value: {"n": count,
    "said": {each value of ``fixed_pair``: rate, "both": rate, "nothing": rate}}}; the rates within a true value sum
    to 1. Uses the same normalised substring test as :func:`statement_accuracy`."""
    pats = {v: _norm(s) for v, s in fixed_pair.items()}
    out: dict[str, dict] = {}
    for true in fixed_pair:
        group = [it for it in items if it["value"] == true]
        counts = {**{v: 0 for v in fixed_pair}, "both": 0, "nothing": 0}
        for it in group:
            text = _norm(readouts.get(it["id"]) or "")
            hits = [v for v, p in pats.items() if p in text]
            counts[hits[0] if len(hits) == 1 else ("nothing" if not hits else "both")] += 1
        n = len(group)
        out[true] = {"n": n, "said": {k: (c / n if n else float("nan")) for k, c in counts.items()}}
    return out

# ---- FVE (nla.py's definition, restated so the eval can score reconstructions with a fixed baseline) ----------------

def fve(pred: torch.Tensor, gold: torch.Tensor, baseline: float | None = None) -> tuple[torch.Tensor, float]:
    """(per-row FVE, baseline). Rows rescaled to norm sqrt(d); mse_i = mean_j (u_pred - u_gold)^2; baseline = the
    rescaled gold set's variance around its mean (computed from ``gold`` unless given); FVE_i = 1 - mse_i / baseline."""
    from mats.readers.nla import fve as _fve
    from mats.readers.nla import predict_mean_baseline

    base = predict_mean_baseline(gold) if baseline is None else float(baseline)
    return _fve(pred, gold, base), base


# ---- P5 baseline (within-item FVE of naming vs non-naming samples) --------------------------------------------------

def within_item_fve_diff(samples: Mapping[str, Sequence[tuple[bool, float]]]) -> dict:
    """``samples``: item id -> [(names_state, fve), ...]. Mean over items having both kinds of sample of
    (mean FVE of naming samples - mean FVE of non-naming samples)."""
    diffs = []
    for rows in samples.values():
        a = [f for names, f in rows if names and not math.isnan(f)]
        b = [f for names, f in rows if not names and not math.isnan(f)]
        if a and b:
            diffs.append(sum(a) / len(a) - sum(b) / len(b))
    return {"n_items": len(diffs), "mean_diff": (sum(diffs) / len(diffs)) if diffs else None}


# ---- small helpers -------------------------------------------------------------------------------------------------

def trained_adapter_path(out: str, arm: str) -> str:
    """Absolute path of the trained adapter of ``arm``; absolute so that ``load_nla(adapter=path)`` resolves it as is
    (``Path(root) / "/abs"`` is ``"/abs"``) and derives the adapter name ``iter_000400`` from its last component."""
    return f"{out}/adapter_{arm}/{ADAPTER_NAME}"


def within(x: float, centre: float, tol: float) -> bool:
    """|x - centre| <= tol, False on NaN / None."""
    if x is None or centre is None or math.isnan(x) or math.isnan(centre):
        return False
    return abs(x - centre) <= tol + 1e-12
