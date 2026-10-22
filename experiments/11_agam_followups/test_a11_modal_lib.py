"""Unit tests for a11_modal_lib (condition construction, splits, punctuation token sets)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import a11_modal_lib as ML  # noqa: E402


def test_median_split_balanced_and_binary():
    y = ML.median_split(np.array([1, 5, 9, 2, 8, 3]))
    assert set(y.tolist()) <= {0, 1}
    assert y.sum() == 3


def test_unit_mean_diff_direction_points_from_0_to_1():
    X = torch.zeros(6, 4)
    X[:3, 0] = 1.0
    y = np.array([1, 1, 1, 0, 0, 0])
    v = ML.class_mean_direction(X, y)
    assert torch.allclose(v, torch.tensor([1.0, 0, 0, 0]))
    assert abs(float(v.norm()) - 1) < 1e-6


def test_q90_dose_is_q90_minus_median_of_projection():
    X = torch.arange(10, dtype=torch.float32)[:, None] * torch.tensor([[1.0, 0.0]])
    v = torch.tensor([1.0, 0.0])
    d = ML.q90_dose(X, v)
    proj = X @ v
    assert d == pytest.approx(float(torch.quantile(proj, 0.9) - proj.median()))


def test_centred_conditions_have_equal_norms_and_expected_count():
    torch.manual_seed(0)
    H = torch.randn(5, 8)
    v = torch.randn(8)
    v = v / v.norm()
    nulls = [torch.randn(8) for _ in range(3)]
    nulls = [n / n.norm() for n in nulls]
    conds = ML.build_conditions(H, {"jrem": v, "raw": v}, {"jrem": 2.0, "raw": 2.5}, {"n": nulls}, null_ref="jrem")
    names = [c[0] for c in conds]
    assert names[0] == "zero" and "jrem+" in names and "raw-" in names and "n0+" in names
    assert len(conds) == 1 + 2 * 2 + 2 * 3
    plus = dict(conds)["jrem+"]
    minus = dict(conds)["jrem-"]
    assert torch.allclose(plus.norm(dim=1), minus.norm(dim=1), atol=1e-5)
    assert torch.allclose((plus - minus) @ v, torch.full((5,), 4.0), atol=1e-4)


def test_punct_token_ids_picks_strings_starting_with_punctuation():
    vocab = {0: " the", 1: ".", 2: ",\n", 3: " (", 4: "abc", 5: "\n", 6: " ?"}
    sel = ML.punct_token_ids(lambda i: vocab[i], len(vocab))
    assert sel["punct"] == [1, 2, 3, 6]
    assert sel["newline"] == [2, 5]


def test_template_sets_are_pairs_of_strings():
    for feat in ("col", "tense"):
        for reader in ("ol", "nla"):
            pairs = ML.TEMPLATES[feat][reader]
            assert pairs and all(len(p) == 3 and isinstance(p[1], str) for p in pairs)
