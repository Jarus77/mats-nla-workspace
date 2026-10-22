"""Unit tests for a11_lib (pure functions; no torch, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import a11_lib as L  # noqa: E402


# ----------------------------------------------------------------------------- consensus
def test_consensus_counts_items_by_support():
    hit = lambda t: "line" in t  # noqa: E731
    rows = [["line a", "line b", "x", "y"],      # 2 of 4 hits
            ["line", "x", "y", "z"],              # 1 of 4
            ["x", "y", "z", "w"],                 # 0 of 4
            ["line", "line", "line", "line"]]     # 4 of 4
    r = L.consensus_rate(rows, hit, min_count=2)
    assert r["n_items"] == 4 and r["k"] == 4
    assert r["single"] == pytest.approx(7 / 16)
    assert r["items_any"] == pytest.approx(3 / 4)
    assert r["consensus"] == pytest.approx(2 / 4)
    assert r["support"] == [2, 1, 0, 4]


def test_consensus_ignores_none_samples_and_ragged_rows():
    hit = lambda t: t == "h"  # noqa: E731
    rows = [["h", None, "h"], ["x"]]
    r = L.consensus_rate(rows, hit, min_count=2)
    assert r["support"] == [2, 0]
    assert r["single"] == pytest.approx(2 / 3)
    assert r["k"] == 3  # max row length


def test_self_consistency_is_majority_share():
    hit = lambda t: t == "y"  # noqa: E731
    rows = [["y", "y", "y", "n"], ["y", "n", "y", "n"], ["n"] * 4]
    s = L.self_consistency(rows, hit)
    assert s["per_item"] == pytest.approx([0.75, 0.5, 1.0])
    assert s["mean"] == pytest.approx((0.75 + 0.5 + 1.0) / 3)


# ----------------------------------------------------------------------------- punctuation context
@pytest.mark.parametrize("text,ends,sent_start", [
    ("He went home.", "stop", True),
    ("He went home. The", "word", True),
    ("He went home,", "punct", False),
    ("He went home and then", "word", False),
    ("Was it good? Yes", "word", True),
    ("Was it good?", "stop", True),
    ("the (first", "word", False),
])
def test_punct_context(text, ends, sent_start):
    c = L.punct_context(text, window=6)
    assert c["ends"] == ends
    assert c["sentence_start"] is sent_start


def test_punct_context_window_limits_sentence_start():
    assert L.punct_context("End. a b c d e f g h", window=4)["sentence_start"] is False
    assert L.punct_context("End. a", window=4)["sentence_start"] is True


# ----------------------------------------------------------------------------- strata
def test_strata_stats_groups_paired_shifts():
    p = np.array([1.0, 2.0, 3.0, 0.0])
    m = np.array([0.0, 0.0, 0.0, 1.0])
    strata = ["a", "a", "b", "b"]
    s = L.strata_stats(p, m, strata)
    assert set(s) == {"a", "b"}
    assert s["a"]["n"] == 2 and s["a"]["mean_shift"] == pytest.approx(1.5)
    assert s["b"]["n"] == 2 and s["b"]["mean_shift"] == pytest.approx(1.0)
    assert 0.0 <= s["a"]["auroc"] <= 1.0


# ----------------------------------------------------------------------------- map
def test_assemble_map_and_rank_corr():
    cells = [{"layer": 40, "pos": 0, "feature": "col", "probe_auroc": 0.9, "z": {"OL": 1.0}},
             {"layer": 42, "pos": 0, "feature": "col", "probe_auroc": 0.95, "z": {"OL": 2.0}},
             {"layer": 44, "pos": 0, "feature": "col", "probe_auroc": 0.8, "z": {"OL": 0.5}}]
    m = L.assemble_map(cells)
    assert m["col"][(42, 0)]["probe_auroc"] == 0.95
    rc = L.rank_corr([c["probe_auroc"] for c in cells], [c["z"]["OL"] for c in cells])
    assert rc["rho"] == pytest.approx(1.0) and rc["n"] == 3


# ----------------------------------------------------------------------------- catalogue
def test_shared_contradictions_pairs_readers_on_the_same_item():
    rows = [
        {"reader": "NLA@42", "family": "f", "id": "i1",
         "claims": [{"claim": "wrong A", "label": "contradicted_by_text"}, {"claim": "ok", "label": "prompt_inferable"}]},
        {"reader": "OL@42", "family": "f", "id": "i1",
         "claims": [{"claim": "wrong B", "label": "contradicted_by_text"}]},
        {"reader": "NLA@42", "family": "f", "id": "i2", "claims": [{"claim": "fine", "label": "prompt_restatement"}]},
        {"reader": "OL@42", "family": "f", "id": "i2", "claims": [{"claim": "wrong C", "label": "contradicted_by_text"}]},
    ]
    out = L.shared_contradictions(rows, readers=("NLA@42", "OL@42"))
    assert [o["id"] for o in out] == ["i1"]
    assert out[0]["NLA@42"] == ["wrong A"] and out[0]["OL@42"] == ["wrong B"]
    assert out[0]["family"] == "f"
