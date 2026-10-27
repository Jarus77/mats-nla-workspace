"""Pure helpers for exp 29 (naming and carrying along the NLA's RL trajectory, on the exp 22 public bank). No model
and no torch at import time; the GPU stages in modal29.py, the judge (judge29.py) and the laptop analysis
(analyze29.py) all import this file.

Provenance: the selection split, the explanation payload rule and exp 14's frozen probe are lib25's (exp 25, which
measured the same critic AUROC for iter_000400); the strict naming regexes are lib26's (`NAME_RE`, exp 23's phrasing
for the structural rows); the text-bag classifier is exp 23's call of lib19's `bag_of_tokens` (whole readout, min_df
3, grouped 5-fold, lam 1.0); the judge rows follow exp 12's `Readout` shape so recall_lib's prompt and aggregate
run unchanged. `strip_expl_tail` is modal22's `_strip_expl`; `strip_analysis` is exp 06's `_strip`."""
from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXPS = _HERE.parent
for _d in (_EXPS / "22_public_bank", _EXPS / "23_random_points", _EXPS / "25_public_exhibits", _EXPS / "26_layer_sweep",
           _EXPS / "19_natural_bank"):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))                     # appended, not inserted: this directory's env.py must win
from feature_spec22 import FEATURES, LAYER  # noqa: E402,F401
from lib25 import MARKED, explanation_text, frozen_probe_auroc, select_split  # noqa: E402,F401
from lib26 import NAME_RE, names_rate, names_state  # noqa: E402,F401

ROWS: tuple[str, ...] = tuple(FEATURES)
STRUCTURAL_ROWS: tuple[str, ...] = tuple(r for r in ROWS if r != "code_lang")
CONTROL_ROW = "code_lang"
STRING_ROWS: tuple[str, ...] = ("py_string", "js_string", "latex_math")
READABLE_ROWS: tuple[str, ...] = ("py_string", "js_string", "latex_math", "py_arg", "json_depth")   # exp 22 readable >= 0.95
CKPT_TEXT = "textnla"
CKPT_BASE = "av_base"
FINAL_CKPT = "iter_000400"                       # the adapter every sprint experiment used (nla.DEFAULT_ADAPTER)
K_SEL = 1                                        # readouts per selection item (naming, carrying, text bag)
K_MINE = 8                                       # readouts per mining item (the judge-scored "states" cell, exp 22)
MEASURES: tuple[str, ...] = ("states", "names", "carries", "textbag")
USD_PER_READOUT = 1.17 / 4096                    # exp 22's judge log: $1.17 for 16 cells x 320 readouts
JUDGE_SAMPLES_FALLBACK = 4

_ITER = re.compile(r"^iter_(\d{6})$")
_ANALYSIS = re.compile(r"<analysis>(.*?)(</analysis>|$)", re.DOTALL)


# ---- checkpoints --------------------------------------------------------------------------------------------------

def iter_step(name: str) -> int | None:
    """RL step of a checkpoint name: av_base -> 0, iter_000400 -> 400, textnla -> None; ValueError otherwise."""
    if name == CKPT_TEXT:
        return None
    if name == CKPT_BASE:
        return 0
    m = _ITER.match(name)
    if not m:
        raise ValueError(f"not a checkpoint name: {name!r}")
    return int(m.group(1))


def checkpoint_order(names: Iterable[str]) -> list[str]:
    """Training order: textnla (no NLA training at all), av_base (warm start, step 0), then iter_* by step."""
    uniq = list(dict.fromkeys(names))
    for n in uniq:
        iter_step(n)                                 # validates
    key = lambda n: (-1, 0) if n == CKPT_TEXT else (0, iter_step(n))  # noqa: E731
    return sorted(uniq, key=key)


def discover_checkpoints(adapter_dir_names: Iterable[str]) -> list[str]:
    """av_base plus every `iter_*` directory name found under av_rl_adapters, in training order."""
    iters = [n for n in adapter_dir_names if _ITER.match(n)]
    return checkpoint_order([CKPT_BASE, *iters])


def adapter_path(ckpt: str) -> str | None:
    """Repo-relative adapter path for load_nla; None for av_base (no adapter); textnla has no adapter at all."""
    if ckpt == CKPT_BASE:
        return None
    if ckpt == CKPT_TEXT:
        raise ValueError("textnla is not an NLA checkpoint")
    iter_step(ckpt)
    return f"av_rl_adapters/{ckpt}"


def OUT_FILE(kind: str, ckpt: str, row: str) -> str:  # noqa: N802
    return f"{kind}29_{ckpt}_{row}.json"


# ---- readout text -----------------------------------------------------------------------------------------------------

def strip_analysis(raw: str) -> str:
    """exp 06 `_strip`: the payload of <analysis> (closed or not); the whole text when there is no tag."""
    m = _ANALYSIS.search(raw or "")
    return m.group(1).strip() if m else (raw or "").strip()


def strip_expl_tail(t: str) -> str:
    """modal22 `_strip_expl`: what exp 22 stored for the judge (raw AV text, cut at a stray </br>)."""
    return t.split("</br>")[0].strip() if "</br>" in t else (t or "").strip()


def readout_text(raw: str, ckpt: str) -> str:
    """The readout payload used for naming, carrying and the text bag: <explanation> body for NLA checkpoints
    (lib25.explanation_text), <analysis> body for TextNLA; "(empty)" when nothing is left."""
    t = strip_analysis(raw) if ckpt == CKPT_TEXT else explanation_text(raw or "")
    return t if t else "(empty)"


# ---- tables -----------------------------------------------------------------------------------------------------------

def rate_table(values: Mapping[str, Mapping[str, float]], rows: Sequence[str], ckpts: Sequence[str] | None = None) -> str:
    """Markdown rows x checkpoints (training order; the union of the value keys unless `ckpts` is given), 2 d.p."""
    cols = list(ckpts) if ckpts is not None else checkpoint_order(values)
    out = ["| row | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for r in rows:
        cells = [values.get(c, {}).get(r) for c in cols]
        out.append(f"| {r} | " + " | ".join("" if v is None else f"{v:.2f}" for v in cells) + " |")
    return "\n".join(out)


def merge(a: Mapping, b: Mapping) -> dict:
    """A new dict with a's keys and b's keys (b wins on a shared key); neither input is touched."""
    return {**a, **b}


def merge_nested(a: Mapping[str, Mapping], b: Mapping[str, Mapping]) -> dict:
    """Two {ckpt: {row: value}} tables merged per checkpoint, as new dicts."""
    return {c: merge(a.get(c, {}), b.get(c, {})) for c in dict.fromkeys([*a, *b])}


# ---- text bag (exp 23's call of lib19) ---------------------------------------------------------------------------------

def word_tokens(text: str) -> list[str]:
    """Deterministic fallback tokeniser (case-folded words and single punctuation marks) when the Qwen tokenizer is
    not cached locally; analyze29 records which tokeniser was used."""
    return re.findall(r"[a-z0-9_]+|[^\sa-z0-9_]", (text or "").lower())


def textbag_auroc(token_ids: Sequence[Sequence], y, groups) -> float:  # noqa: ANN001
    """Grouped 5-fold AUROC of a bag-of-tokens logistic probe on the WHOLE readout (exp 23 analyze23: lib19's
    `bag_of_tokens` with window 10**6, min_df 3; `cv_auroc` with lam 1.0). Tokens may be ints or strings."""
    import numpy as np

    from lib19 import bag_of_tokens
    from mats.probes import cv_auroc

    vocab = {t: i for i, t in enumerate(sorted({t for ids in token_ids for t in ids}, key=str))}
    ids_int = [[vocab[t] for t in ids] for ids in token_ids]
    X = bag_of_tokens(ids_int, window=10**6, min_df=3)
    groups = np.asarray(groups)
    k = min(5, len(np.unique(groups)))
    return float(cv_auroc(X, np.asarray(y), groups, k=k, lam=1.0)["auroc"])


# ---- judge ---------------------------------------------------------------------------------------------------------------

def judge_rows(gen: Mapping, value_of: Mapping[str, str], samples: int = K_MINE) -> list[dict]:
    """exp 12 `Readout` fields (item, value, k, text) for the first `samples` mining readouts of every item that
    has a value; file order kept."""
    out = []
    for item, texts in gen.get("mine", {}).items():
        if item not in value_of:
            continue
        for k, t in enumerate(list(texts)[:samples]):
            out.append({"item": item, "value": value_of[item], "k": k, "text": t})
    return out


def judge_cost_estimate(n_readouts: int, per_readout: float = USD_PER_READOUT) -> float:
    return float(n_readouts) * per_readout


def samples_for_budget(n_items: int, per_ckpt_cap: float, full: int = K_MINE, fallback: int = JUDGE_SAMPLES_FALLBACK) -> int:
    """8 samples per item unless one checkpoint's judge bill would exceed `per_ckpt_cap` dollars, then 4."""
    return full if judge_cost_estimate(n_items * full) <= per_ckpt_cap else fallback


def stated_rate(judged: Sequence[Mapping]) -> float | None:
    """recall_lib.aggregate's `stated_rate` (share of judged readouts the judge marks as stating the property)."""
    n = len(judged)
    return None if n == 0 else sum(bool(r["stated"]) for r in judged) / n


# ---- predictions (RESULTS.md "Predictions written before the run") -------------------------------------------------------

P1_STATES_MAX = 0.25
P2_CODE_LANG_RISE = 0.15
P3_CARRY_RISE = 0.05
P3_TEXTNLA_TOL = 0.05
P4_TEXTBAG_TOL = 0.05


def mean_delta(values: Mapping[str, Mapping[str, float]], a: str, b: str, rows: Sequence[str]) -> float | None:
    """Mean over `rows` of value[b] - value[a]; None when no row has both cells."""
    ds = [values[b][r] - values[a][r] for r in rows if r in values.get(a, {}) and r in values.get(b, {})]
    return sum(ds) / len(ds) if ds else None


def check_predictions(summary: Mapping[str, Mapping[str, Mapping[str, float]]], final: str = FINAL_CKPT) -> dict:
    """The four pre-registered checks on {states, carries, textbag}[ckpt][row]; None when the cells are missing."""
    states, carries, textbag = (summary.get(k, {}) for k in ("states", "carries", "textbag"))
    cells = [v[r] for v in states.values() for r in STRUCTURAL_ROWS if r in v]
    p1 = all(c <= P1_STATES_MAX for c in cells) if cells else None
    d2 = mean_delta(states, CKPT_BASE, final, [CONTROL_ROW])
    p2 = (d2 >= P2_CODE_LANG_RISE) if d2 is not None else None
    d3 = mean_delta(carries, CKPT_BASE, final, READABLE_ROWS)
    t3 = mean_delta(carries, CKPT_BASE, CKPT_TEXT, READABLE_ROWS)
    p3 = (d3 >= P3_CARRY_RISE and abs(t3) <= P3_TEXTNLA_TOL) if (d3 is not None and t3 is not None) else None
    spans = []
    for r in STRING_ROWS:
        vals = [v[r] for v in textbag.values() if r in v]
        if len(vals) >= 2:
            spans.append(max(vals) - min(vals))
    p4 = all(s < P4_TEXTBAG_TOL for s in spans) if spans else None
    return {"p1_structural_states_flat": p1, "p2_code_lang_states_rise": p2,
            "p3_carrying_rises_and_textnla_near_base": p3, "p4_textbag_stable_on_strings": p4}
