"""chain_intermediates: the decoy set, the verdict, the item rule, a scripted run and a dry run."""

import json
from pathlib import Path

import pytest

from wsbench import registry
from wsbench.banks import load_bank
from wsbench.cli import main
from wsbench.evals.chain_intermediates import judge
from wsbench.evals.chain_intermediates.prompts import PROMPT_VERSION, SYSTEM
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.registry import JudgeArgs
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "evals/chain_intermediates/items.json"
EXAMPLE = REPO / "examples/readouts/chain_intermediates.jsonl"


@pytest.fixture(autouse=True)
def _registered():
    from wsbench.evals.chain_intermediates import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)


def test_bank():
    header, items = load_bank(BANK)
    assert header["family"] == "chain_intermediates" and len(items) == 120
    assert {it["depth"] for it in items} == {2, 3}
    for it in items:
        gold = it["intermediates"]
        assert len(gold) == it["depth"] - 1
        # the leak rule: no intermediate is a numeral literal of the prompt
        assert all(v not in it["literals"] for v in gold)
        assert it["start"] not in gold and it["answer"] not in gold


def test_near_set_excludes_gold_and_stays_positive():
    assert judge.near_set([11, 5]) == [8, 9, 10, 12, 13, 14, 2, 3, 4, 6, 7]
    assert judge.near_set([11, 5], exclude=(23, 2)) == [8, 9, 10, 12, 13, 14, 3, 4, 6, 7]
    assert judge.near_set([2]) == [1, 3, 4, 5]
    assert judge.names_number(5, "then 5.") and not judge.names_number(5, "was 15 or 5.0")
    assert judge.names_number(11, "Ġ11 | Ġthe") and not judge.names_number(83, "Ġ11 (10.83)")
    assert 5 not in judge.near_set([5, 6]) and 6 not in judge.near_set([5, 6])


def test_verdict_kinds():
    gold, near = [11, 5], judge.near_set([11, 5])
    readout = "- the running value is 11 after the first step\n- then 5"
    hit = judge.verdict(
        {"values": [11, 5], "basis": "arithmetic", "quote": "11"}, gold, near, readout
    )
    assert hit["kind"] == "hit" and hit["hit"] and hit["any_of_3"]
    second = judge.verdict(
        {"values": [23, 11], "basis": "numeral_bag", "quote": "23"}, gold, near, "23 then 11"
    )
    assert second["kind"] == "other" and not second["hit"] and second["any_of_3"]
    near_v = judge.verdict(
        {"values": [12], "basis": "stated_result", "quote": "12"}, gold, near, "it is 12"
    )
    assert near_v["kind"] == "near" and not near_v["hit"]
    # a value the readout never shows, with a quote that is not verbatim: not credited
    ghost = judge.verdict(
        {"values": [11], "basis": "arithmetic", "quote": "eleven-ish"}, gold, near, "nothing here"
    )
    assert ghost["kind"] == "unverified" and not ghost["hit"]
    # a Chinese numeral is credited through the verbatim quote
    cjk = judge.verdict(
        {"values": [11], "basis": "numeral_bag", "quote": "十一"}, gold, near, "结果是 十一"
    )
    assert cjk["kind"] == "hit"
    # a digit-only quote inside a longer number does not verify
    sub = judge.verdict({"values": [5], "basis": "stated_result", "quote": "5"}, gold, near, "15")
    assert sub["kind"] == "unverified"
    none = judge.verdict(
        {"values": [], "states_value": False, "basis": "none", "quote": ""}, gold, near, readout
    )
    assert none["kind"] == "none" and not none["hit"]
    assert (
        judge.verdict({"values": "11", "basis": "none", "quote": ""}, gold, near, readout)["kind"]
        == "none"
    )
    assert judge.verdict(
        {"values": [True, 11], "basis": "none", "quote": "11"}, gold, near, readout
    )["values"] == [11]
    capped = judge.verdict(
        {"values": [1, 2, 3, 11], "basis": "numeral_bag", "quote": "1"}, gold, near, "1 2 3 11"
    )
    assert capped["values"] == [1, 2, 3] and not capped["any_of_3"]
    assert judge.verdict(None, gold, near, readout)["judged"] is False


def _v(item, layer, kind, values, judged=True):
    return {
        "item": item,
        "layer": layer,
        "kind": kind,
        "values": values,
        "hit": kind == "hit",
        "judged": judged,
        "any_of_3": kind == "hit",
        "basis": "arithmetic",
    }


def test_item_row_tri_state_and_null():
    it = {"id": "a", "depth": 3, "intermediates": [11, 5], "answer": 2, "start": 23}
    gold, near = [11, 5], judge.near_set([11, 5])
    vs = [_v("a", 20, "other", [23]), _v("a", 24, "hit", [11]), _v("a", 28, "near", [12])]
    row = judge._item_row(it, gold, near, vs, [], set())
    assert row["pass"] is True and row["earliest_layer"] == 24 and row["hitting_layers"] == [24]
    assert row["null_top1_near"] == pytest.approx(1 * 2 / len(near) / 3)
    vs = [_v("a", 20, "other", [23]), _v("a", 24, "none", [])]
    assert judge._item_row(it, gold, near, vs, [], set())["pass"] is False
    assert judge._item_row(it, gold, near, vs, [("a", 28)], set())["pass"] is None
    vs = [_v("a", 20, "unavailable", [], judged=False)]
    assert judge._item_row(it, gold, near, vs, [], set())["pass"] is None
    # all cells empty: judged negative, null denominator counts them
    row = judge._item_row(it, gold, near, [], [], {("a", 20), ("a", 24)})
    assert row["pass"] is False and row["null_top1_near"] == 0.0


def test_scripted_run_on_the_toy_file(tmp_path, monkeypatch):
    _h, items = load_bank(BANK)
    by_id = {it["id"]: it for it in items}
    lines = [x for x in EXAMPLE.read_text().splitlines() if x.strip()]
    ids = sorted({json.loads(x)["id"] for x in lines})
    assert len(ids) == 3
    a, b, c = ids

    def fake_run_calls(calls, **kw):
        out = {}
        for call in calls:
            i, layer = call.meta["item"], call.meta["layer"]
            gold = by_id[i]["intermediates"]
            if i == a and layer == 44:
                out[call.key] = {
                    "states_value": True,
                    "values": [gold[0]],
                    "basis": "arithmetic",
                    "quote": str(gold[0]),
                }
            elif i == b:
                out[call.key] = {
                    "states_value": True,
                    "values": [gold[0] + 1],
                    "basis": "stated_result",
                    "quote": str(gold[0] + 1),
                }
            elif i == c and layer == 20:
                out[call.key] = None
            else:
                out[call.key] = {"states_value": False, "values": [], "basis": "none", "quote": ""}
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    args = JudgeArgs(
        readouts=EXAMPLE,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=[a, b, c],
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
    )
    r = judge.run(args)
    rows = {row["id"]: row for row in r.rows}
    assert rows[a]["pass"] is True and rows[a]["earliest_layer"] == 44
    assert rows[b]["pass"] is False and rows[b]["kind_by_layer"]["20"] == "near"
    assert rows[b]["null_top1_near"] > 0
    assert rows[c]["pass"] is None  # one unavailable verdict and no hit
    assert r.value == 0.5 and r.extras["n_items_decided"] == 2
    assert r.extras["n_items_undecided"] == 1 and r.n_items == 3
    assert r.counts["n_unjudged_cells"] == 1 and r.counts["n_missing_cells"] == 0
    assert r.extras["null_top1_near"] > 0 and r.extras["per_depth"]
    assert r.extras["kinds"]["hit"] == 1 and r.extras["kinds"]["near"] == 11


def test_missing_cells_are_fatal_without_allow_missing(tmp_path):
    args = JudgeArgs(
        readouts=EXAMPLE,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=None,
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
    )
    with pytest.raises(SystemExit) as e:
        judge.run(args)
    assert e.value.code == 2


def test_registered_readme_and_dry_run(tmp_path, capsys, monkeypatch):
    from wsbench.evals.chain_intermediates import SPEC

    assert SPEC.group == "computational" and SPEC.judge.prompt_version == PROMPT_VERSION
    assert SYSTEM in (REPO / "evals/chain_intermediates/README.md").read_text(encoding="utf-8")
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    argv = ["judge", f"family={SPEC.name}", f"readouts={EXAMPLE}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] == 117
    assert "Lens output:" in capsys.readouterr().out


def test_all_cells_mode_scores_every_row(tmp_path, monkeypatch):
    """``opts=cells=all``: every (layer, pos) row is a cell; an item passes on any cell, and the
    per-cell hits are listed."""
    _h, items = load_bank(BANK)
    it = items[0]
    gold = it["intermediates"][0]
    rows = [
        {"id": it["id"], "layer": 20, "pos": 5, "samples": ["nothing"]},
        {"id": it["id"], "layer": 20, "pos": 6, "samples": [f"the value is {gold}"]},
        {"id": it["id"], "layer": 44, "pos": 5, "samples": ["nothing"]},
        {"id": it["id"], "layer": 44, "pos": 6, "samples": ["nothing"]},
    ]

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            if str(gold) in c.user:
                out[c.key] = {
                    "states_value": True,
                    "values": [gold],
                    "basis": "a",
                    "quote": str(gold),
                }
            else:
                out[c.key] = {"states_value": False, "values": [], "basis": "none", "quote": ""}
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    args = JudgeArgs(
        readouts=path,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=[it["id"]],
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
        extra={"cells": "all"},
    )
    r = judge.run(args)
    row = r.rows[0]
    assert row["pass"] is True and row["hits_at"] == [(20, 6)] and row["earliest_layer"] == 20
    assert r.extras["n_calls"] == 4 and r.extras["n_rows_not_last_token"] == 0
    assert r.extras["cells_per_item"] == 4.0
    assert r.extras["per_layer_hit_rate"] == {"20": 0.5, "44": 0.0}
    assert r.config["cells"] == "all"
