import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import ar_lib as L  # noqa: E402


def test_span_crop_and_pad():
    assert L.span_crop([1, 2, 3, 4, 5], 3) == [3, 4, 5]
    assert L.span_crop([1, 2], 5) == [1, 2]
    with pytest.raises(ValueError):
        L.span_crop([1], 0)
    ids, mask = L.right_pad([[7, 8, 9], [5]], pad_id=0)
    assert ids.tolist() == [[7, 8, 9], [5, 0, 0]] and mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert (mask.sum(1) - 1).tolist() == [2, 0]


def test_rename_lora_keys():
    sd = {"base_model.model.layers.3._orig_mod.mlp.up_proj.lora_A.weight": torch.zeros(1)}
    assert list(L.rename_lora_keys(sd)) == ["base_model.model.layers.3.mlp.up_proj.lora_A.weight"]


def test_fve_and_retrieval():
    g = torch.randn(50, 16)
    assert L.fve_raw(g, g) == pytest.approx(1.0)
    assert L.fve_raw(g.mean(0).expand_as(g), g) == pytest.approx(0.0, abs=1e-6)
    assert L.retrieval_at_1(g, g) == 1.0
    assert L.retrieval_at_1(g[torch.randperm(50)], g) < 0.2


def _toy(n=200, d=24, strength=3.0, seed=0):
    rng = torch.Generator().manual_seed(seed)
    y = np.array([i % 2 for i in range(n)])
    groups = np.arange(n) // 2
    X = torch.randn(n, d, generator=rng)
    X[:, 0] += strength * torch.as_tensor(y, dtype=torch.float32)
    return X, y, groups


def test_retention_kept_vs_dropped():
    X, y, g = _toy()
    kept = X.clone()
    dropped = X.clone()
    dropped[:, 0] = 0.0  # the ruler erases the concept dimension
    r_kept = L.frozen_probe_retention(X, kept, y, g)
    r_drop = L.frozen_probe_retention(X, dropped, y, g)
    assert r_kept["true"] > 0.9 and r_kept["recon"] > 0.9
    assert r_drop["recon"] < 0.65 and r_drop["err"] > 0.9
    assert L.retrained_probe_auroc(dropped, y, g) < 0.65
    assert L.gap_retention(X, kept, y)["ratio"] == pytest.approx(1.0)
    assert abs(L.gap_retention(X, dropped, y)["ratio"]) < 0.2


def test_statement_shift():
    X, y, _ = _toy()
    a, b = X[y == 1].mean(0), X[y == 0].mean(0)
    full = L.statement_shift(X, y, a, b)
    assert full["ratio"] == pytest.approx(1.0) and full["cosine"] == pytest.approx(1.0)
    assert L.statement_shift(X, y, b, a)["ratio"] == pytest.approx(-1.0)
    assert abs(L.statement_shift(X, y, a, a)["ratio"]) < 1e-6
