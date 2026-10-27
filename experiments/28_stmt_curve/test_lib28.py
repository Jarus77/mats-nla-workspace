"""Tests for lib28 (laptop, no model): the statement-shift formula against exp 14's, the loaders on synthetic blobs,
the permutation correlations, the binomial logistic fit, the band table, the paraphrase parser and collect_rows on a
tiny synthetic repo."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import lib28 as L  # noqa: E402


def _ar_lib():
    spec = importlib.util.spec_from_file_location("ar_lib14", HERE.parent / "14_ar_blindspot" / "ar_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ the formula
def test_statement_shift_hand_case():
    d = torch.tensor([2.0, 0.0, 0.0])                       # gap 2 along the first axis
    rec_a = torch.tensor([1.0, 5.0, 0.0])
    rec_b = torch.tensor([0.0, 0.0, 0.0])
    # ((rec_a - rec_b) . d_hat) / gap = (1 * 1) / 2
    assert L.statement_shift(rec_a, rec_b, d) == pytest.approx(0.5)
    assert L.statement_shift(rec_b, rec_a, d) == pytest.approx(-0.5)
    assert L.statement_shift(torch.tensor([0.0, 9.0, 0.0]), rec_b, d) == pytest.approx(0.0)


def test_statement_shift_equals_exp14_and_exp22_formula_on_random_data():
    g = torch.Generator().manual_seed(28)
    X = torch.randn(40, 16, generator=g)
    y = np.arange(40) % 2
    X[:, 3] += 3.0 * torch.as_tensor(y, dtype=torch.float32)
    ra, rb = torch.randn(16, generator=g), torch.randn(16, generator=g)
    d, gap = L.class_direction(X, y)
    ref = _ar_lib().statement_shift(X, y, ra, rb)["ratio"]
    assert L.statement_shift(ra, rb, d) == pytest.approx(ref, abs=1e-5)
    # exp 22's modal22 c7lx form: ((rs0 - rs1) @ u) / ||d_true||
    u = d / d.norm()
    assert L.statement_shift(ra, rb, d) == pytest.approx(float(((ra - rb) @ u) / d.norm()), abs=1e-6)
    assert gap == pytest.approx(float(d.norm()))


def test_class_direction_is_marked_minus_unmarked_mean():
    X = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    y = np.array([1, 1, 0, 0])
    d, gap = L.class_direction(X, y)
    assert torch.allclose(d, torch.tensor([2.0, -2.0]))
    assert gap == pytest.approx(np.sqrt(8))
    with pytest.raises(ValueError):
        L.class_direction(X, np.array([1, 1, 1, 1]))


def test_shift_summary_reports_fixed_mean_min_max_sd_range():
    s = L.shift_summary(0.10, [0.0, 0.1, 0.2, 0.3])
    assert s["fixed"] == 0.10 and s["para_mean"] == pytest.approx(0.15)
    assert s["para_min"] == 0.0 and s["para_max"] == pytest.approx(0.3) and s["para_range"] == pytest.approx(0.3)
    assert s["para_sd"] == pytest.approx(np.std([0.0, 0.1, 0.2, 0.3], ddof=1))
    assert s["n_para"] == 4


# ------------------------------------------------------------------------------------------------ loaders
def _blob(keys, n=6, d=4):
    g = torch.Generator().manual_seed(1)
    return {"ids": [f"it{i}" for i in range(n)], "acts": {k: torch.randn(n, d, generator=g) for k in keys}}


def test_acts_from_blob_accepts_int_or_str_layer_keys_and_checks_ids():
    for keys in ((42, 40, 44), ("42", "36")):
        blob = _blob(keys)
        A = L.acts_from_blob(blob, 42, blob["ids"])
        assert A.shape == (6, 4) and A.dtype == torch.float32
    with pytest.raises(KeyError):
        L.acts_from_blob(_blob((40, 44)), 42, None)
    with pytest.raises(ValueError):
        L.acts_from_blob(_blob((42,)), 42, ["other"] * 6)


def test_items_from_gen_ol_and_items_base_formats():
    items = [{"id": "a", "feature": "f1", "split": "select", "cluster": "c1", "value": "x"},
             {"id": "b", "feature": "f2", "split": "select", "cluster": "c2", "value": "x"}]
    assert [it["id"] for it in L.items_from({"feature": "f1", "items": items[:1]}, "gen_ol", "f1")] == ["a"]
    assert [it["id"] for it in L.items_from(items, "items_base", "f2")] == ["b"]
    with pytest.raises(ValueError):
        L.items_from({"feature": "f9", "items": items}, "gen_ol", "f1")
    with pytest.raises(ValueError):
        L.items_from(items, "unknown", "f1")


def test_select_split_labels_marked_value_one_and_keeps_clusters():
    items = [{"id": "a", "split": "mine", "cluster": "c0", "value": "in"},
             {"id": "b", "split": "select", "cluster": "c1", "value": "in"},
             {"id": "c", "split": "select", "cluster": "c1", "value": "out"},
             {"id": "d", "split": "select", "cluster": "c2", "value": "out"}]
    idx, y, groups = L.select_split(items, "in")
    assert idx == [1, 2, 3] and y.tolist() == [1, 0, 0] and groups.tolist() == ["c1", "c1", "c2"]


# ------------------------------------------------------------------------------------------------ statistics
def test_spearman_perm_detects_monotone_and_is_null_on_noise():
    rng = np.random.default_rng(0)
    x = np.arange(30, dtype=float)
    r = L.spearman_perm(x, x ** 2 + rng.normal(0, 1, 30), n_perm=500, seed=0)
    assert r["rho"] > 0.95 and r["p"] < 0.01 and r["n"] == 30
    r0 = L.spearman_perm(x, rng.normal(0, 1, 30), n_perm=500, seed=0)
    assert abs(r0["rho"]) < 0.5 and r0["p"] > 0.05
    assert np.isnan(L.spearman_perm([1, 2], [1, 2], n_perm=10, seed=0)["rho"])


def test_pearson_perm_matches_numpy_corrcoef():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=25), rng.normal(size=25)
    r = L.pearson_perm(x, y, n_perm=200, seed=0)
    assert r["r"] == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_auroc_threshold_separates_the_high_rows():
    shift = np.array([0.0, 0.1, 0.2, 0.5, 0.7])
    stated = np.array([0.0, 0.1, 0.4, 0.8, 0.9])
    assert L.auroc_threshold(shift, stated, 0.30) == pytest.approx(1.0)
    assert L.auroc_threshold(shift, stated[::-1], 0.30) == pytest.approx(0.0)
    assert np.isnan(L.auroc_threshold(shift, np.zeros(5), 0.30))


def test_logistic_fit_recovers_a_planted_slope_and_aic_prefers_the_true_model():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 200)
    p = 1 / (1 + np.exp(-(-0.5 + 3.0 * x)))
    n = np.full(200, 320)
    k = rng.binomial(n, p)
    fit = L.logistic_fit(x[:, None], k, n)
    assert fit["converged"] and fit["coef"][0] == pytest.approx(-0.5, abs=0.15) and fit["coef"][1] == pytest.approx(3.0, abs=0.3)
    null = L.logistic_fit(np.zeros((200, 0)), k, n)
    assert null["aic"] - fit["aic"] > 100
    assert fit["k_params"] == 2 and null["k_params"] == 1
    # a separate-intercept-only null fit is reproducible
    assert L.logistic_fit(np.zeros((200, 0)), k, n)["coef"][0] == pytest.approx(null["coef"][0])


def test_band_table_counts_rows_per_shift_band():
    rows = [{"x": -0.05, "stated": 0.0}, {"x": 0.05, "stated": 0.4}, {"x": 0.15, "stated": 0.1},
            {"x": 0.30, "stated": 0.7}, {"x": 0.72, "stated": 0.9}]
    t = L.band_table(rows, "x", "stated")
    assert [b["n"] for b in t] == [2, 1, 2]
    assert t[0]["mean_stated"] == pytest.approx(0.2) and t[2]["min_stated"] == pytest.approx(0.7)
    assert t[0]["band"] == "< 0.10" and t[1]["band"] == "0.10 to 0.25" and t[2]["band"] == ">= 0.25"


# ------------------------------------------------------------------------------------------------ paraphrases
def test_parse_pairs_accepts_eight_aligned_pairs_and_rejects_bad_output():
    good = {"pairs": [{"register": f"r{i}", "a": f"Value A phrasing {i}", "b": f"Value B phrasing {i}"} for i in range(8)]}
    pairs = L.parse_pairs(json.dumps(good), ("fixed A", "fixed B"))
    assert len(pairs) == 8 and pairs[0] == ["Value A phrasing 0", "Value B phrasing 0"]
    assert L.parse_pairs(None, ("a", "b")) is None
    assert L.parse_pairs("not json", ("a", "b")) is None
    seven = {"pairs": good["pairs"][:7]}
    assert L.parse_pairs(json.dumps(seven), ("a", "b")) is None
    same = {"pairs": [{"register": "r", "a": "x", "b": "x"}] * 8}
    assert L.parse_pairs(json.dumps(same), ("a", "b")) is None
    dup = {"pairs": [{"register": "r", "a": "x", "b": "y"}] * 8}
    assert L.parse_pairs(json.dumps(dup), ("a", "b")) is None


def test_paraphrase_prompt_carries_the_fixed_pair_and_eight_registers():
    row = {"experiment": "22", "feature": "py_string", "values": ["inside", "outside"],
           "desc": "Python source: inside a string literal vs outside", "fixed": ["The current position is inside a string literal",
                                                                                   "The current position is outside a string literal"]}
    p = L.paraphrase_prompt(row)
    assert "inside a string literal" in p and "outside a string literal" in p
    assert len(L.REGISTERS) == 8 and all(r in p for r in L.REGISTERS)


# ------------------------------------------------------------------------------------------------ collect_rows
def _mini_repo(tmp_path: Path) -> dict:
    spec = tmp_path / "spec_x.py"
    spec.write_text(
        'FEATURES = {"row_a": {"values": ["in", "out"], "desc": "a toy row"}, "row_b": {"values": ["p", "q"], "desc": "b"}}\n'
        'FIXED = {"nla": {"row_a": {"template": "The text is {v}", "fillers": {"in": "inside", "out": "outside"}}}}\n')
    recall = tmp_path / "recall.json"
    recall.write_text(json.dumps({"row_a": {"nla": {"stated_rate": 0.25, "n_samples": 320}},
                                  "row_b": {"nla": {"stated_rate": 0.0, "n_samples": 320}}}))
    score = tmp_path / "score.json"
    score.write_text(json.dumps({"rows": [{"feature": "row_a", "observed": {"nla_states": 0.5}}]}))
    (tmp_path / "gen_nla_row_a.json").write_text(json.dumps({"nla_gens": {f"i{i}": ["t"] * 8 for i in range(40)}}))
    instr = tmp_path / "instr.json"
    instr.write_text(json.dumps([{"feature": "row_a", "instr": {"readable": 0.9, "span": 0.8, "span_gap": 0.2, "stmt": 0.1, "used_z": float("nan")}}]))
    return {
        "s1": {"spec": spec, "score": (recall, "recall"), "vol": "/results/s1", "items_fmt": "gen_ol",
               "acts_name": "acts_{f}.pt", "instr": None, "gens": None},
        "s2": {"spec": spec, "score": (score, "score"), "vol": "/results/s2", "items_fmt": "items_base",
               "acts_name": "acts2_{f}.pt", "instr": instr, "gens": tmp_path / "gen_nla_{f}.json"},
    }


def test_collect_rows_on_a_synthetic_repo(tmp_path):
    rows, missing = L.collect_rows(sources=_mini_repo(tmp_path), types={("s1", "row_a")})
    assert [(r["experiment"], r["feature"]) for r in rows] == [("s1", "row_a"), ("s2", "row_a")]
    a, b = rows
    assert a["fixed"] == ["The text is inside", "The text is outside"] and a["values"] == ["in", "out"]
    assert a["stated"] == 0.25 and a["n_readouts"] == 320 and a["type"] == "self_inducing"
    assert a["acts"] == {"vol": "/results/s1/acts_row_a.pt", "layer": L.LAYER}
    assert a["items"] == {"vol": "/results/s1/gen_ol_row_a.json", "fmt": "gen_ol"}
    assert a["prior"] == {}
    assert b["stated"] == 0.5 and b["n_readouts"] == 320 and b["type"] == "structural"
    assert b["items"] == {"vol": "/results/s2/items_base.json", "fmt": "items_base"}
    assert b["prior"]["stmt"] == 0.1 and b["prior"]["used_z"] is None
    # row_b has a stated rate but no fixed NLA pair: reported as missing, not silently dropped
    assert missing == [{"experiment": "s1", "feature": "row_b", "reason": "no FIXED['nla'] statement pair in the spec"}]


def test_collect_rows_real_repo_has_the_memo_rows():
    rows, missing = L.collect_rows()
    keys = {(r["experiment"], r["feature"]) for r in rows}
    for k in L.COUNTER_ROWS | L.LANGUAGE_ROWS:
        assert k in keys, k
    assert all(r["fixed"][0] != r["fixed"][1] for r in rows)
    assert all(0.0 <= r["stated"] <= 1.0 for r in rows)
    assert len(rows) >= 40
