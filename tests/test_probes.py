"""Offline tests for src/mats/probes.py."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mats import probes as P  # noqa: E402


def _data(n=120, d=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = np.repeat([1, 0], n // 2)
    X = torch.randn(n, d, generator=g)
    X[:, 0] += torch.as_tensor(y * 3.0, dtype=torch.float32)
    groups = np.tile(np.arange(n // 2), 2)
    return X, y, groups


def test_group_folds_disjoint_groups():
    _, _, g = _data()
    folds = P.group_folds(g, k=4)
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(g)))
    for f in folds:
        others = np.setdiff1d(np.arange(len(g)), f)
        assert not set(g[f]) & set(g[others])


def test_cv_auroc_and_removal():
    X, y, g = _data()
    assert P.cv_auroc(X, y, g)["auroc"] > 0.9
    kill = lambda Z: torch.cat([torch.zeros(len(Z), 1), Z[:, 1:]], 1)  # noqa: E731
    assert P.cv_auroc(X, y, g, transform=kill, retrain=True)["auroc"] < 0.7
    assert P.cv_auroc(X, y, g, transform=kill, retrain=False)["auroc"] < 0.7


def test_directions_point_along_signal():
    X, y, _ = _data()
    assert float(P.mean_diff_direction(X, y)[0]) > 0.9
    m = P.fit_logreg(X, torch.as_tensor(y))
    assert float(m.direction()[0]) > 0.8
