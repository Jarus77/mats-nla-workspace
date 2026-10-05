"""Round-5 item generators (DECLARED PILOT SPLIT; ids recorded in pilot_split.json).

digit: numbered lists "1. <sentence>\\n2. <sentence>\\n...N. <sentence>\\n" with N in {4, 6}; read token =
       the final token (contains the newline), where the model is about to emit the next item number
       (5 or 7). Sentences (5-16 words) come from fresh wikitext paragraphs (pool seed DIGIT_SEED),
       excluding 02, 07, the col pilot and the calibration passages. TRAIN (direction/gates): 200 per
       class; EVAL (reader cells): 30 per class; disjoint sentences.
tense120: calib_items.build_tense with seed TENSE_SEED, 60 Past + 60 Pres, excluding the calibration tense
       passages as well.
"""
from __future__ import annotations

import random
import re

import calib_items as CI
import pilot_items as PI

DIGIT_SEED = 10010
TENSE_SEED = 11011


def _sentences(paras: list[str]) -> list[str]:
    out = []
    for p in paras:
        for s in re.split(r"(?<=[.!?])\s+", p):
            n = len(s.split())
            if 5 <= n <= 16 and s.endswith(".") and not re.search(r"\d", s) and "\n" not in s:
                out.append(s)
    return out


def build_digit(tok, out02: str, out07: str, exclude_sha: set) -> dict:  # noqa: ANN001
    s1 = PI.load_02("s1_data", out02)
    pool = s1._wikitext_paragraphs(300, 60000, DIGIT_SEED)
    used = PI._used_prefixes(out02, out07)
    by_head: dict[str, list[str]] = {}
    for u in used:
        by_head.setdefault(u[:30], []).append(u)
    fresh = [p for p in pool[:8000] if not PI._overlaps(p, by_head) and CI._sha(p) not in exclude_sha]
    sents = _sentences(fresh)
    rng = random.Random(DIGIT_SEED)
    rng.shuffle(sents)
    it = iter(sents)
    out = {"train": [], "eval": []}
    for split, n_per in (("eval", 30), ("train", 200)):
        for N in (4, 6):
            for k in range(n_per):
                items = [next(it) for _ in range(N)]
                text = "".join(f"{i + 1}. {s}\n" for i, s in enumerate(items))
                ids = tok(text, add_special_tokens=False)["input_ids"]
                out[split].append({"id": f"digit_{split}_{N + 1}_{k:03d}", "next_digit": N + 1,
                                   "y": 1 if N == 6 else 0, "ids": ids, "text": text,
                                   "sent_heads": [s[:40] for s in items]})
    return out


def build_tense120(tok, out02: str, out07: str, col_items: list[dict], exclude_sha: set) -> list[dict]:  # noqa: ANN001
    return CI.build_tense(tok, out02, out07, col_items, seed=TENSE_SEED, n_per_class=60,
                          exclude_sha=exclude_sha, id_prefix="r5_tense", pool_slice=12000)
