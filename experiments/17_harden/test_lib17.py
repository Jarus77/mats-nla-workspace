import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))  # mats package, as run_local.py does
import lib17 as L  # noqa: E402
import spec17 as S  # noqa: E402


def test_candidates_fill_in_slot_order():
    t = {"id": "ol:mc:fixed", "template": "- column {v}", "slots": ["one", "two"]}
    assert L.candidates(t) == ["- column one", "- column two"]


def test_candidates_returns_word_lists_for_jlens():
    t = {"id": "jlens:mc:fixed", "template": None, "slots": [["first", "one"], ["second", "two"]]}
    assert L.candidates(t) == [["first", "one"], ["second", "two"]]


def test_true_index_table_column_and_drift():
    assert L.true_index("tb_tablecol", {"meta": {"col": 5}}) == 4
    assert L.true_index("tb_tablecol", {"meta": {"col": 1}}) == 0
    with pytest.raises(ValueError):
        L.true_index("tb_tablecol", {"meta": {"col": 2}})
    assert L.true_index("tb_paren", {"value": "inside"}) == 0
    assert L.true_index("tb_paren", {"value": "outside"}) == 1


def test_paren_item_filter():
    assert L.paren_item_ok("He said (as before) that")
    assert not L.paren_item_ok('He said "as before (that')
    assert not L.paren_item_ok("He said [as before (that")


def test_metrics_multi_perfect_and_chance():
    lp = np.full((6, 6), -5.0)
    truth = np.array([0, 0, 0, 3, 4, 5])
    y = np.array([0, 0, 0, 1, 1, 1])
    lp[np.arange(6), truth] = 0.0
    m = L.metrics_multi(lp, truth, y, S.CLASS_SETS["tb_tablecol"], S.ROWS["tb_tablecol"]["values"],
                        np.arange(6), n_boot=50, seed=0)
    assert m["top1"]["point"] == 1.0 and m["top1"]["chance"] == pytest.approx(1 / 6)
    assert m["auroc_lse"]["point"] == 1.0 and m["mrr"]["point"] == 1.0
    assert m["n"] == 6
    flat = np.zeros((6, 6))
    m0 = L.metrics_multi(flat, truth, y, S.CLASS_SETS["tb_tablecol"], S.ROWS["tb_tablecol"]["values"],
                         np.arange(6), n_boot=50, seed=0)
    assert m0["auroc_lse"]["point"] == pytest.approx(0.5)


def test_metrics_multi_keeps_template_axis_out():
    # one template at a time: a [N, C] matrix, never a stacked [T, N, C]
    with pytest.raises(ValueError):
        L.metrics_multi(np.zeros((2, 3, 4)), np.zeros(3, int), np.zeros(3, int), {"a": [0], "b": [1]}, ["a", "b"],
                        np.arange(3), n_boot=5, seed=0)


def test_lse_pair_lo_matches_two_candidate_when_classes_are_singletons():
    lp = np.array([[0.0, -1.0], [-2.0, 0.0]])
    out = L.lse_pair_lo(lp, {"inside": [0], "outside": [1]}, ["inside", "outside"])
    assert np.allclose(out, lp[:, 1] - lp[:, 0])


def test_spec_candidate_counts_match_rows():
    for row, per_reader in S.MC_TEMPLATES.items():
        for reader, temps in per_reader.items():
            for t in temps:
                assert len(t["slots"]) == S.ROWS[row]["n_cands"], (row, reader, t["id"])


def test_pick_unembedding_prefers_lm_head_and_falls_back():
    assert L.pick_unembedding({"lm_head.weight": "a", "model.norm.weight": "b"}) == ("lm_head.weight", "a", "b")
    assert L.pick_unembedding({"model.embed_tokens.weight": "a"}) == ("model.embed_tokens.weight", "a", None)
    assert L.pick_unembedding({"lm_head.weight": "a", "model.language_model.norm.weight": "c"}) == ("lm_head.weight", "a", "c")
    with pytest.raises(KeyError):
        L.pick_unembedding({"model.layers.0.x": "a"})


def test_vocab_stats_peak_ratio_is_large_for_one_hot():
    x = np.zeros(1000)
    x[7] = 10.0
    x[3] = -4.0
    s = L.vocab_stats(x, top_k=3)
    assert s["peak_ratio"] > 20 and s["top_pos"][0][0] == 7 and s["top_neg"][0][0] == 3
    assert len(s["top_pos"]) == 3 and 0 < s["top_mass"] <= 1
    flat = np.ones(1000)
    assert L.vocab_stats(flat, top_k=3)["peak_ratio"] == float("inf")


def test_vocab_markdown_verdict_uses_null_band_and_content_words():
    import report17 as R

    def entry(peak, toks):
        return {"peak_ratio": peak, "top_mass": 0.5, "top_pos_tokens": toks, "top_neg_tokens": [], "rank_of_interest": {}}
    res = {"feature": "tb_tablecol", "gain": "1+w", "dirs": {"cm_raw": entry(8.0, [" the"]), "arstmt_raw": entry(30.0, [" fifth", " column"])},
           "nulls": {f"iso_{k}": entry(8.0 + 0.1 * k, []) for k in range(20)},
           "content_words": ["first", "fifth", "column"]}
    md = R.vocab_markdown([res])
    assert "lexical: yes" in md and "arstmt_raw" in md
    res["dirs"]["arstmt_raw"] = entry(8.5, [" the"])
    assert "lexical: no" in R.vocab_markdown([res])


def test_mc_table_reads_pt_and_keeps_template_axis(tmp_path):
    import torch

    import analyze17 as A

    blob = {"ids": ["a", "b", "c", "d"], "truth": np.array([0, 3, 0, 4]), "y": np.array([0, 1, 0, 1]),
            "clusters": ["a", "b", "c", "d"], "excluded": [],
            "lp": {"ol:mc:fixed": torch.zeros(4, 6), "ol:mc:m3": torch.zeros(4, 6)}}
    torch.save(blob, tmp_path / "mc_ol_tb_tablecol.pt")
    out = A.mc_table("tb_tablecol", tmp_path, n_boot=20, reference=None)
    assert set(out["cells"]["ol"]) == {"ol:mc:fixed", "ol:mc:m3"}
    assert "nla" not in out["cells"]  # missing reader file is omitted, not an error
    assert out["n"] == 4 and out["cells"]["ol"]["ol:mc:fixed"]["top1"]["chance"] == pytest.approx(1 / 6)


def test_confusion_counts_argmax_per_class():
    lp = np.array([[0, -1, -2], [0, -1, -2], [-2, 0, -1], [-2, -1, 0]])
    y = np.array([0, 0, 1, 1])
    assert L.confusion(lp, y, ["a", "b"]) == {"a": [2, 0, 0], "b": [0, 1, 1]}
