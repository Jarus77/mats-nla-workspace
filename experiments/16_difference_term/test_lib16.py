import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib16 as L  # noqa: E402


def test_partners_opposite_class_other_cluster():
    y = np.array([0, 0, 1, 1, 0, 1])
    cl = ["a", "b", "a", "b", "c", "c"]
    j = L.partners(y, cl, seed=0)
    assert all(y[j[i]] != y[i] and cl[j[i]] != cl[i] for i in range(6))
    with pytest.raises(ValueError):
        L.partners(np.array([0, 1]), ["a", "a"], seed=0)


def test_construct_identities():
    torch.manual_seed(0)
    h, r_ol, r_nla = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
    j = np.array([1, 0, 3, 2])
    v = L.construct(h, r_ol, r_nla, j)
    assert torch.allclose(v["swap_res"], r_ol + (h - r_ol)[j])
    assert torch.allclose(v["swap_ol"], r_ol[j] + (h - r_ol))
    assert torch.allclose(v["swap_resn"], r_ol + (r_nla - r_ol)[j])
    assert torch.allclose(v["swap_res"] + v["swap_ol"], h + h[j], atol=1e-5)  # the two swaps split h_i + h_j
    assert torch.equal(v["ol_only"], r_ol)


def test_follow_is_complementary_for_opposite_partners():
    y = np.array([0, 0, 1, 1])
    j = np.array([2, 3, 0, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8])
    f = L.follow(s, y, j)
    assert f["follows_i"] == 1.0 and f["follows_j"] == 0.0
    f2 = L.follow(s[j], y, j)  # readout that tracks the partner
    assert f2["follows_j"] == 1.0 and f2["follows_i"] == 0.0


def test_fve_rows():
    g = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    assert L.fve_rows(g, g) == pytest.approx(1.0)
    assert L.fve_rows(g.mean(0, keepdim=True).expand_as(g), g) == pytest.approx(0.0)
