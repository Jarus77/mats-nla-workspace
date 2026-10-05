"""Offline unit tests for src/mats/metrics.py (small synthetic data, no GPU, no network)."""

import math

import numpy as np
import pytest

from mats import metrics as M


# ----------------------------------------------------------------------------- AUROC


def test_auroc_perfect_and_reversed():
    assert M.auroc([2, 3, 4], [0, 1]) == 1.0
    assert M.auroc([0, 1], [2, 3, 4]) == 0.0


def test_auroc_ties_count_half():
    assert M.auroc([1, 1], [1, 1]) == 0.5
    # one tie out of 4 comparisons: pos {1, 2} vs neg {1, 0} -> (0.5 + 1 + 1 + 1) / 4
    assert M.auroc([1, 2], [1, 0]) == pytest.approx(3.5 / 4)


def test_auroc_matches_bruteforce():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(0.5, 1, 37), rng.normal(0, 1, 23)
    pos[:5] = neg[:5]  # force ties
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert M.auroc(pos, neg) == pytest.approx(brute)


def test_auroc_from_labels():
    s = np.array([0.1, 0.9, 0.4, 0.8])
    y = np.array([0, 1, 0, 1])
    assert M.auroc_labels(s, y) == 1.0


def test_auroc_empty_class_is_nan():
    assert math.isnan(M.auroc([], [1.0]))


# ----------------------------------------------------------------------------- bootstrap


def test_cluster_bootstrap_ci_contains_point_and_is_deterministic():
    rng = np.random.default_rng(1)
    n = 60
    clusters = np.repeat(np.arange(n), 2)
    labels = np.tile([1, 0], n)
    scores = labels * 0.8 + rng.normal(0, 1, 2 * n)
    r1 = M.auroc_ci(scores, labels, clusters, n_boot=400, seed=3)
    r2 = M.auroc_ci(scores, labels, clusters, n_boot=400, seed=3)
    assert r1 == r2
    assert r1.lo <= r1.auroc <= r1.hi
    assert 0.5 < r1.auroc < 1.0
    assert r1.n_clusters == n


def test_cluster_bootstrap_wider_than_naive_when_clusters_are_duplicated():
    """Duplicating every row (perfectly correlated within cluster) must not shrink the CI."""
    rng = np.random.default_rng(2)
    n = 40
    labels = rng.integers(0, 2, n)
    scores = labels * 0.7 + rng.normal(0, 1, n)
    single = M.auroc_ci(scores, labels, np.arange(n), n_boot=600, seed=0)
    dup = M.auroc_ci(np.repeat(scores, 4), np.repeat(labels, 4), np.repeat(np.arange(n), 4),
                     n_boot=600, seed=0)
    naive = M.auroc_ci(np.repeat(scores, 4), np.repeat(labels, 4), np.arange(4 * n),
                       n_boot=600, seed=0)
    assert dup.auroc == pytest.approx(single.auroc)
    assert (dup.hi - dup.lo) > 1.5 * (naive.hi - naive.lo)
    assert (dup.hi - dup.lo) == pytest.approx(single.hi - single.lo, rel=0.35)


def test_cluster_bootstrap_generic_statistic():
    vals = np.arange(10.0)
    lo, hi, boots = M.cluster_bootstrap(lambda idx: vals[idx].mean(), np.arange(10), n_boot=300,
                                        seed=0, return_samples=True)
    assert lo < vals.mean() < hi
    assert len(boots) == 300


# ----------------------------------------------------------------------------- FollowFeature


def test_follow_feature_detects_shift_and_null():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 2, 80)  # item-level prior, shared by +/- versions
    plus, minus = base + 1.0 + rng.normal(0, 0.3, 80), base - 1.0 + rng.normal(0, 0.3, 80)
    r = M.follow_feature(plus, minus, n_boot=300, seed=0)
    assert r["auroc"] > 0.6 and r["lo"] > 0.5
    assert r["paired_win"] > 0.95
    assert r["mean_shift"] == pytest.approx(2.0, abs=0.2)
    # zero-dose baseline subtraction removes item priors -> near-perfect separation
    zero = base + rng.normal(0, 0.3, 80)
    r0 = M.follow_feature(plus, minus, lo_zero=zero, n_boot=300, seed=0)
    assert r0["auroc"] > r["auroc"]
    null = M.follow_feature(base + rng.normal(0, 1, 80), base + rng.normal(0, 1, 80), n_boot=300,
                            seed=0)
    assert null["lo"] < 0.5 < null["hi"]


def test_follow_feature_clusters_default_to_items():
    r = M.follow_feature(np.ones(5), np.zeros(5), n_boot=50, seed=0)
    assert r["n_clusters"] == 5
    assert r["auroc"] == 1.0


def test_follow_feature_respects_explicit_clusters():
    plus, minus = np.arange(8.0) + 0.5, np.arange(8.0)
    r = M.follow_feature(plus, minus, clusters=np.repeat(np.arange(4), 2), n_boot=50, seed=0)
    assert r["n_clusters"] == 4


# ----------------------------------------------------------------------------- FollowText


def test_follow_text_pairs():
    rng = np.random.default_rng(5)
    a, b = rng.normal(1, 1, 50), rng.normal(0, 1, 50)
    r = M.follow_text(a, b, n_boot=200, seed=0)
    assert 0.6 < r["auroc"] < 0.9
    assert r["n_clusters"] == 50


# ----------------------------------------------------------------------------- fraction of probe


def test_fraction_of_probe_point():
    assert M.fraction_of_probe(0.75, 1.0) == pytest.approx(0.5)
    assert M.fraction_of_probe(0.5, 0.9) == pytest.approx(0.0)
    assert math.isnan(M.fraction_of_probe(0.7, 0.5))


def test_fraction_of_probe_ci_joint_bootstrap():
    rng = np.random.default_rng(6)
    n = 60
    labels = np.tile([1, 0], n)
    clusters = np.repeat(np.arange(n), 2)
    probe = labels * 5.0 + rng.normal(0, 1, 2 * n)  # AUROC ~1
    reader = labels * 0.6 + rng.normal(0, 1, 2 * n)
    r = M.fraction_of_probe_ci(reader, probe, labels, clusters, n_boot=300, seed=0)
    assert r["lo"] <= r["fraction"] <= r["hi"]
    assert 0.0 < r["fraction"] < 1.0
    assert r["auroc_probe"] > 0.99


# ----------------------------------------------------------------------------- paired null


def test_paired_null_auroc_difference():
    rng = np.random.default_rng(7)
    n = 80
    base = rng.normal(0, 1, n)
    real = (base + 0.8 + rng.normal(0, 0.5, n), base - 0.8 + rng.normal(0, 0.5, n))
    null = (base + rng.normal(0, 0.5, n), base + rng.normal(0, 0.5, n))
    r = M.paired_null_auroc(real, null, n_boot=300, seed=0)
    assert r["auroc_real"] > r["auroc_null"]
    assert r["diff_lo"] > 0
    assert r["p_one_sided"] < 0.05
    same = M.paired_null_auroc(null, null, n_boot=200, seed=0)
    assert same["diff"] == pytest.approx(0.0)


# ----------------------------------------------------------------------------- Page trend


def test_page_trend_matches_scipy_asymptotic():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(8)
    data = rng.normal(0, 1, (25, 4)) + np.array([0.0, 0.2, 0.4, 0.6])
    ours = M.page_trend_test(data)
    ref = scipy_stats.page_trend_test(data, method="asymptotic")
    assert ours["L"] == pytest.approx(ref.statistic)
    assert ours["p"] == pytest.approx(ref.pvalue, rel=1e-6)


def test_page_trend_direction():
    inc = np.tile(np.arange(4.0), (12, 1)) + np.random.default_rng(0).normal(0, 0.1, (12, 4))
    assert M.page_trend_test(inc)["p"] < 1e-3
    assert M.page_trend_test(inc[:, ::-1])["p"] > 0.99


def test_page_trend_handles_ties_within_block():
    data = np.array([[1.0, 1.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    r = M.page_trend_test(data)
    assert np.isfinite(r["L"]) and 0 <= r["p"] <= 1


# ----------------------------------------------------------------------------- summaries


def test_void_rule_fve_guard():
    nat = np.linspace(0.45, 0.95, 20)  # realistic per-example spread (SD ~0.15)
    ok = M.fve_guard(nat, nat - 0.02, margin=0.05)
    bad = M.fve_guard(nat, nat - 0.10, margin=0.05)
    assert not ok["void"] and bad["void"] and bad["void_margin"]
    floor = M.fve_guard(np.full(20, 0.75), np.full(20, 0.73), margin=0.05)
    assert floor["void_p5"] and not floor["void_margin"]  # degenerate spread: p5 rule bites
    assert ok["mean_diff"] == pytest.approx(-0.02)
    assert bad["lo"] <= bad["mean_diff"] <= bad["hi"]


def test_fve_guard_ignores_nan_pairs():
    nat = np.array([0.7, 0.8, np.nan, 0.75])
    st = np.array([0.69, np.nan, 0.6, 0.74])
    r = M.fve_guard(nat, st, margin=0.05)
    assert r["n_pairs"] == 2


# ----------------------------------------------------------------------------- null band


def test_paired_stats_fast():
    r = M.paired_stats([2.0, 3.0, 1.0], [1.0, 1.0, 2.0])
    assert r["paired_win"] == pytest.approx(2 / 3)
    assert r["mean_shift"] == pytest.approx(2 / 3)
    assert r["auroc"] == pytest.approx(M.auroc([2.0, 3.0, 1.0], [1.0, 1.0, 2.0]))


def test_null_band_percentile_and_z():
    nulls = np.linspace(-1, 1, 21)
    r = M.null_band(2.0, nulls)
    assert r["percentile"] == 1.0 and r["outside_two_sided"]
    assert r["z"] == pytest.approx(2.0 / np.std(nulls, ddof=1))
    mid = M.null_band(0.0, nulls)
    assert 0.4 < mid["percentile"] < 0.6 and not mid["outside_two_sided"]
    assert mid["band_lo"] == pytest.approx(np.quantile(nulls, 0.025))
    low = M.null_band(-5.0, nulls)
    assert low["percentile"] == 0.0 and low["outside_two_sided"] and low["z"] < 0


def test_null_band_from_contrasts():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 40)
    real = (base + 1, base - 1)
    nulls = [(base + rng.normal(0, .2, 40), base + rng.normal(0, .2, 40)) for _ in range(20)]
    out = M.null_band_contrast(real, nulls)
    for k in ("paired_win", "mean_shift", "auroc"):
        assert out[k]["percentile"] == 1.0
    assert out["mean_shift"]["z"] > 5
