import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib21 as L  # noqa: E402


def test_locate_prefix_finds_the_read_point_in_the_source():
    src = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    prefix = "gamma delta epsilon zeta eta"
    assert L.locate(src, prefix) == src.index(" eta") + 4
    assert L.locate(src, "not in the file at all") is None


def test_continuation_ids_take_the_first_n_after_the_read_point():
    ids = L.continuation_ids("abc", list(range(10)), 3, lambda s: list(range(100, 112)))
    assert len(ids) == 3


def test_continuation_auroc_separates_a_cue_and_not_noise():
    rng = np.random.default_rng(0)
    n = 120
    y = np.array([0, 1] * (n // 2))
    cued = [list(rng.integers(100, 400, 31)) + ([7] if y[i] else [9]) for i in range(n)]
    noise = [list(rng.integers(100, 400, 32)) for i in range(n)]
    g = np.arange(n)
    assert L.continuation_auroc(cued, y, g) > 0.9
    assert abs(L.continuation_auroc(noise, y, g) - 0.5) < 0.15
