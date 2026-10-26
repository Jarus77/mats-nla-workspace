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


def test_load_guiv_items_validates_records(tmp_path):
    import json

    good = [{"id": "g1", "text": "| a | b |", "value": "first", "cluster": "c1"},
            {"id": "g2", "text": "| a | b | c | d |", "value": "late", "cluster": "c2"}]
    p = tmp_path / "guiv_items_tb_tablecol.json"
    p.write_text(json.dumps(good))
    assert [it["id"] for it in L.load_guiv_items(p, "tb_tablecol")] == ["g1", "g2"]
    p.write_text(json.dumps(good + [{"id": "g3", "text": "x", "value": "middle", "cluster": "c3"}]))
    with pytest.raises(ValueError, match="g3"):
        L.load_guiv_items(p, "tb_tablecol")
    p.write_text(json.dumps([{"id": "g4", "text": "x", "value": "first"}]))
    with pytest.raises(ValueError, match="g4"):
        L.load_guiv_items(p, "tb_tablecol")


def test_guiv_pairs_returns_id_a_b_triples():
    phr = {"tb_tablecol": {"nla": [{"id": "g:nla:0", "template": "Cell in column {v}", "fillers": {"first": "one", "late": "five"}}],
                           "ol": []}}
    assert L.guiv_pairs(phr, "tb_tablecol", "nla") == [["g:nla:0", "Cell in column one", "Cell in column five"]]
    assert L.guiv_pairs(phr, "tb_tablecol", "ol") == []
    assert L.guiv_pairs(phr, "tb_paren", "nla") == []


def test_xs_table_two_by_two_auroc(tmp_path):
    import torch

    import analyze17 as A

    his_y = [0, 0, 1, 1]
    lo_track = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    blob = {"his_ids": list("abcd"), "his_y": his_y, "his_clusters": list("abcd"),
            "our_ids": list("wxyz"), "our_y": [0, 1, 0, 1], "our_clusters": list("wxyz"),
            "pairs": {"ours": [["o1", "A", "B"]], "his": [["h1", "A", "B"]]},
            "lo": {"his_x_ours": {"o1": lo_track}, "his_x_his": {"h1": -lo_track},
                   "ours_x_his": {"h1": torch.tensor([-1.0, 1.0, -1.0, 1.0])}}}
    torch.save(blob, tmp_path / "xs_ol_tb_tablecol.pt")
    out = A.xs_table("tb_tablecol", tmp_path, n_boot=20)
    c = out["cells"]["ol"]
    assert c["his_x_ours"]["point"] == 1.0 and c["his_x_his"]["point"] == 0.0 and c["ours_x_his"]["point"] == 1.0
    assert "nla" not in out["cells"]


def test_analyze17_xs_entry_point_runs_without_name_error(tmp_path):
    import subprocess

    r = subprocess.run([sys.executable, str(HERE / "analyze17.py"), "--xs", "--rows", "tb_tablecol", "--vol", str(tmp_path), "--out", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "NameError" not in r.stderr, r.stderr[-500:]


def test_confusion_centred_removes_per_candidate_prior():
    lp = np.array([[0.0, -8.0], [0.0, -8.0], [0.0, -7.0], [0.0, -7.0]])  # candidate 1 has a strong prior penalty
    y = np.array([0, 0, 1, 1])
    assert L.confusion(lp, y, ["a", "b"]) == {"a": [2, 0], "b": [2, 0]}
    assert L.confusion(lp, y, ["a", "b"], centre=True) == {"a": [2, 0], "b": [0, 2]}


def test_mc_verdict_applies_declared_thresholds():
    import report17 as R

    def cell(top1_lo, top1, lse_lo, lse):
        return {"top1": {"point": top1, "lo": top1_lo, "hi": 1.0, "chance": 1 / 6},
                "auroc_lse": {"point": lse, "lo": lse_lo, "hi": 1.0}, "mrr": {"point": 0, "lo": 0, "hi": 0}}
    res = {"row": "tb_tablecol", "cells": {"nla": {"nla:mc:fixed": cell(0.39, 0.47, 0.84, 0.89)},
                                           "ol": {"ol:mc:m3": cell(0.07, 0.12, 0.73, 0.80), "ol:mc:fixed": cell(0.01, 0.03, 0.53, 0.62)}}}
    assert "survives" in R.mc_verdict(res)
    res["cells"]["nla"]["nla:mc:fixed"] = cell(0.30, 0.47, 0.84, 0.89)   # top-1 lower bound at or below 0.33
    assert "artefact" in R.mc_verdict(res)
    res["cells"]["nla"]["nla:mc:fixed"] = cell(0.39, 0.47, 0.84, 0.79)   # AUROC point below OL's best (0.80)
    assert "artefact" in R.mc_verdict(res)
    res["cells"]["ol"] = {}
    assert "pending" in R.mc_verdict(res)                                 # no OL cell: not a pass
    paren = {"row": "tb_paren", "cells": {"nla": {"nla:mc:state": cell(0.51, 0.6, 0.8, 0.9)}, "ol": {"ol:mc:state": cell(0.2, 0.28, 0.3, 0.4)}}}
    assert "survives" in R.mc_verdict(paren)


def test_vocab_markdown_reports_rank_of_interest():
    import report17 as R

    e = {"peak_ratio": 6.0, "top_mass": 0.001, "top_pos_tokens": [" x"], "top_neg_tokens": [" y"],
         "rank_of_interest": {" fifth": 345, " first": 248298}}
    res = {"feature": "tb_tablecol", "gain": "1+w", "vocab": 248320, "dirs": {"arstmt_raw": e},
           "nulls": {f"iso_{k}": {**e, "peak_ratio": 5.0 + 0.01 * k} for k in range(20)}, "content_words": ["first", "fifth", "column"]}
    md = R.vocab_markdown([res])
    assert "345" in md and "248298" in md
