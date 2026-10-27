"""Exp 21: does the state change the text that follows the read point? A text-only classifier (bag of ALL tokens in the
next N tokens of the true source, grouped 5-fold by source file) scored by AUROC. The candidate predictor for how much
Oracle Lens's readout exhibits a state (exp 20's `ol_readout` column)."""
from __future__ import annotations

import numpy as np

N_NEXT = 32
TAIL = 200          # characters of the prefix used to locate the read point in the source


def locate(src: str, prefix: str, tail: int = TAIL) -> int | None:
    """Offset in `src` just after the read point, found by the last `tail` characters of the item's prefix."""
    key = prefix[-tail:]
    i = src.rfind(key)
    return None if i < 0 else i + len(key)


def continuation_ids(src: str, prefix_ids: list[int], n: int, encode) -> list[int]:  # noqa: ANN001
    """Token ids of the next n tokens of `src` (already cut at the read point); `prefix_ids` unused but kept so
    callers can pass the item for symmetry."""
    return encode(src)[:n]


def continuation_auroc(next_ids: list[list[int]], y: np.ndarray, groups: np.ndarray, n: int = N_NEXT) -> float:
    """Bag of all tokens in the first n continuation tokens, grouped-CV logistic probe, AUROC (exp 19's cue check with
    every token counted: punctuation and markers are the point here)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "19_natural_bank"))
    from lib19 import bag_of_tokens

    from mats.probes import cv_auroc

    X = bag_of_tokens([ids[:n] for ids in next_ids], window=10**6, min_df=3)
    k = min(5, len(np.unique(groups)))
    return cv_auroc(X, np.asarray(y), np.asarray(groups), k=k, lam=1.0)["auroc"]
