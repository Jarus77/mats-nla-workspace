import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib15 as L  # noqa: E402
from spec15 import DIR_NAMES, K_NULL, REL_DOSE  # noqa: E402


def test_conditions_count_and_shape():
    c = L.conditions()
    assert c[0] == L.ZERO
    assert len(c) == 1 + len(DIR_NAMES) * 2 * 2 + 2 * K_NULL * 2
    assert all(len(x) == 4 for x in c)
    assert sum(1 for x in c if x[0] == "null") == 2 * K_NULL * 2


def test_class_mean_direction_is_unit_and_points_to_class_one():
    X = torch.zeros(6, 4)
    X[3:, 0] = 2.0
    y = np.array([0, 0, 0, 1, 1, 1])
    v = L.class_mean_direction(X, y)
    assert torch.allclose(v, torch.tensor([1.0, 0, 0, 0]))
    with pytest.raises(ValueError):
        L.class_mean_direction(X, np.zeros(6, int))


def test_build_centred_sets_projection_to_signed_dose():
    torch.manual_seed(0)
    h = torch.randn(5, 8) * 3
    v = L.unit(torch.randn(8))
    hp = L.build(h, v, ("real", "x", "r11", 1))
    hm = L.build(h, v, ("real", "x", "r11", -1))
    assert torch.allclose(hp @ v, REL_DOSE * h.norm(dim=1), atol=1e-5)
    assert torch.allclose(hm @ v, -REL_DOSE * h.norm(dim=1), atol=1e-5)
    assert torch.allclose(hp.norm(dim=1), hm.norm(dim=1), atol=1e-4)  # equal norms both sides
    hq = L.build(h, v, ("real", "x", "q90", 1), q90=0.5)
    assert torch.allclose(hq @ v, torch.full((5,), 0.5), atol=1e-5)
    assert torch.equal(L.build(h, v, L.ZERO), h)
    with pytest.raises(ValueError):
        L.build(h, v, ("real", "x", "q90", 1))


def test_q90_dose_is_quantile_minus_median():
    h = torch.arange(11, dtype=torch.float32)[:, None] * torch.tensor([[1.0, 0.0]])
    assert L.q90_dose(h, torch.tensor([1.0, 0.0])) == pytest.approx(9.0 - 5.0)


def test_continuation_metrics():
    assert L.cells_to_newline("Lyon | staff | north |\n| Hugo |") == (3, True)
    assert L.cells_to_newline("a | b") == (1, False)
    assert L.tokens_to_close(["young", " woman", ")", " and"]) == (2, True)
    assert L.tokens_to_close(["a", "b"]) == (2, False)
    assert L.chars_to_close("ab) c") == (2, True)
    assert L.chars_to_close("abc") == (3, False)


def test_contrast_vs_nulls_z_and_band():
    rng = np.random.default_rng(0)
    n = 50
    minus = rng.normal(size=n)
    real_plus = minus + 1.0
    nulls_p = [minus + rng.normal(scale=0.05, size=n) for _ in range(20)]
    nulls_m = [minus for _ in range(20)]
    out = L.contrast_vs_nulls(real_plus, minus, nulls_p, nulls_m)
    assert out["real"]["mean_shift"] == pytest.approx(1.0)
    assert out["real"]["paired_win"] == 1.0
    assert out["band"]["z"] > 10 and out["band"]["outside"]
    null_like = L.contrast_vs_nulls(nulls_p[0], minus, nulls_p[1:], nulls_m[1:])
    assert abs(null_like["band"]["z"]) < 3


def test_cosines_symmetric_unit_diagonal():
    d = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0, 1.0])}
    c = L.cosines(d)
    assert c["a"]["a"] == pytest.approx(1.0) and c["a"]["b"] == pytest.approx(c["b"]["a"])
    assert c["a"]["b"] == pytest.approx(2 ** -0.5)
