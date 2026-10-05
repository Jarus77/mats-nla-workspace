"""arithmetic_intermediates: the bank, tolerance and verification rules, the verdict with its
permutation null, a scripted run and a dry run."""

import json
import re
from pathlib import Path

import pytest

from wsbench import registry
from wsbench.banks import load_bank
from wsbench.cli import main
from wsbench.evals.arithmetic_intermediates import judge
from wsbench.evals.arithmetic_intermediates.prompts import (
    PROMPT_VERSION,
    PROMPT_VERSION_BATCH,
    SYSTEM,
    SYSTEM_BATCH,
)
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.registry import JudgeArgs
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "evals/arithmetic_intermediates/items.json"
EXAMPLE = REPO / "examples/readouts/arithmetic_intermediates.jsonl"
KEPT = {
    "absval", "addmul", "floordiv", "frac", "fracadd", "fraccomp", "fracint", "fracsmall",
    "muladd", "mulmid", "sign", "signpair", "subsub", "subsubx",
}  # fmt: skip


@pytest.fixture(autouse=True)
def _registered():
    from wsbench.evals.arithmetic_intermediates import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)


def test_bank():
    header, items = load_bank(BANK)
    assert header["family"] == "arithmetic_intermediates" and len(items) == 596
    assert set(header["variants"]) == KEPT and "maxsel" in header["removed_variants"]
    names = {it["name"] for it in items}
    for it in items:
        v = header["variants"][it["variant"]]
        assert it["tolerance"] == v["tolerance"] and it["cell"] == v["cell"]
        assert it["cell"]["layer"] in (56, 60) and it["cell"]["pos"] in (-7, -8)
        assert it["intermediates"] and float(it["intermediates"][0]) == float(
            it["intermediates"][0]
        )
        assert it["null_set"] and all(n in names for n in it["null_set"])
        assert all(n != it["name"] for n in it["null_set"])
    assert sum(v["n_items"] for v in header["variants"].values()) == 596


def test_tolerance_and_quantities():
    assert judge.tolerance_ok(-51.0, -51.0, "exact") and not judge.tolerance_ok(
        51.0, -51.0, "exact"
    )
    assert judge.tolerance_ok(74.0, 73.0, "rel2pct") and not judge.tolerance_ok(
        75.0, 73.0, "rel2pct"
    )
    assert judge.quantities("- 2. The total is 4,816 and -51.\n- 17,000") == [
        4816.0,
        -51.0,
        17000.0,
    ]
    assert judge.quantities("Step 3: 0.777778 \u2248 \uff17.\uff15") == [0.777778, 7.5]
    assert judge.quantities("29.") == [29.0]  # a bare number on its own line is a value
    assert judge.quantities("-51. Then times 14") == [-51.0, 14.0]  # a negative is not a bullet
    assert judge.quantities("- 2. The answer is 9") == [9.0]  # a bulleted list marker is
    assert judge.quantities("\u221251 and \u2212 3") == [-51.0, 3.0]  # the real minus sign


def test_verified_and_verdict():
    text = "- 271 - 322 = -51, then times 14\n- 结果是 负五十一"
    assert judge.verified(-51.0, text, "") and not judge.verified(-52.0, text, "")
    assert judge.verified(-52.0, text, "负五十一")  # a CJK-numeral quote vouches for the top value
    assert not judge.verified(-52.0, text, "负五十一", top=False)  # only for the top value
    assert not judge.verified(-52.0, text, "-52")  # a bare-number quote does not
    assert not judge.verified(-52.0, text, "4,816") and not judge.verified(
        999.0, text, "then times"
    )
    assert not judge.verified(-52.0, text, "not here 五")  # must be verbatim
    nulls = [-40.0, -60.0, 12.0]
    v = judge.verdict(
        {"values": [-51, 14], "basis": "arithmetic", "quote": "-51"}, -51.0, "exact", nulls, text
    )
    assert (
        v["kind"] == "hit"
        and v["hit"]
        and v["top1_hit"]
        and v["cross"] == 0.0
        and v["kept"] == [-51.0, 14.0]
    )
    v = judge.verdict(
        {"values": [-40, -51], "basis": "numeral_bag", "quote": ""},
        -51.0,
        "exact",
        nulls,
        "-40 and -51",
    )
    assert v["hit"] and not v["top1_hit"] and v["cross"] == pytest.approx(1 / 3)
    v = judge.verdict(
        {"values": [99], "basis": "stated_result", "quote": "ninety-nine"},
        -51.0,
        "exact",
        nulls,
        text,
    )
    assert v["kind"] == "unverified" and not v["hit"]
    v = judge.verdict(
        {"values": [14], "basis": "stated_result", "quote": "14"}, -51.0, "exact", nulls, text
    )
    assert v["kind"] == "other" and v["cross"] == 0.0
    assert (
        judge.verdict({"values": [], "basis": "none", "quote": ""}, -51.0, "exact", nulls, text)[
            "kind"
        ]
        == "none"
    )
    assert judge.verdict(
        {"values": [True, 1, 2, 3, 4], "basis": "none", "quote": ""}, 1.0, "exact", [], "1 2 3 4"
    )["values"] == [1.0, 2.0, 3.0]
    assert judge.verdict(None, -51.0, "exact", nulls, text)["judged"] is False
    # relative tolerance: 4.30 counts for a 4.333 intermediate at 2%
    v = judge.verdict(
        {"values": [4.303030303030303], "basis": "arithmetic", "quote": ""},
        4.333333,
        "rel2pct",
        [],
        "142/33 = 4.303030303030303",
    )
    assert v["hit"]


def test_scripted_run(tmp_path, monkeypatch):
    _h, items = load_bank(BANK)
    by_id = {it["id"]: it for it in items}
    ids = sorted({json.loads(x)["id"] for x in EXAMPLE.read_text().splitlines() if x.strip()})
    assert len(ids) == 4
    pick = {by_id[i]["variant"]: i for i in ids}
    a, b, c, d = pick["sign"], pick["addmul"], pick["frac"], pick["subsub"]

    def fake_run_calls(calls, **kw):
        out = {}
        for call in calls:
            i = call.meta["item"]
            target = float(by_id[i]["intermediates"][0])
            if i == a:  # names the intermediate
                out[call.key] = {
                    "states_value": True,
                    "values": [target],
                    "basis": "arithmetic",
                    "quote": "",
                }
            elif i == b:  # names a null item's intermediate (a cross hit, not a hit)
                other = float(by_id[by_id[i]["null_set"][0]]["intermediates"][0])
                out[call.key] = {
                    "states_value": True,
                    "values": [other],
                    "basis": "stated_result",
                    "quote": "",
                }
            elif i == c:
                out[call.key] = None
            else:
                out[call.key] = {"states_value": False, "values": [], "basis": "none", "quote": ""}
        return out

    # item b's readout writes a null item's intermediate, so the value verifies and crosses
    other = by_id[by_id[b]["null_set"][0]]["intermediates"][0]
    rows = [json.loads(x) for x in EXAMPLE.read_text().splitlines() if x.strip()]
    for row in rows:
        if row["id"] == a:  # the real rows do not write -51; a scripted hit must verify
            row["samples"] = [f"- 271 - 322 = {by_id[a]['intermediates'][0]}, then times 14"]
        if row["id"] == b:
            row["samples"] = [f"- the sum comes to {other}"]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    args = JudgeArgs(
        readouts=path,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=ids,
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
    )
    r = judge.run(args)
    rows = {row["id"]: row for row in r.rows}
    assert rows[a]["pass"] is True and rows[a]["kind"] == "hit" and rows[a]["cross"] == 0.0
    assert rows[b]["pass"] is False and rows[b]["kind"] == "other" and rows[b]["cross"] > 0
    assert rows[c]["pass"] is None and rows[d]["pass"] is False
    assert r.value == pytest.approx(1 / 3) and r.extras["n_items_decided"] == 3
    assert r.extras["cross"] > 0 and r.extras["net"] < r.value
    assert r.counts["n_unjudged_cells"] == 1 and r.counts["n_missing_cells"] == 0
    assert set(r.extras["per_variant"]) <= KEPT and r.extras["per_role"]


def test_rows_off_the_frozen_cell_are_ignored_and_missing_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "run_calls", lambda calls, **kw: {})
    _h, items = load_bank(BANK)
    it = items[0]
    rows = [
        {
            "id": it["id"],
            "layer": it["cell"]["layer"],
            "pos": it["cell"]["pos"] + 1,
            "samples": ["x"],
        },
        {"id": it["id"], "layer": 20, "pos": it["cell"]["pos"], "samples": ["x"]},
    ]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    base = {
        "readouts": path,
        "out": tmp_path / "out",
        "judge": resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        "layers": None,
        "items": [it["id"]],
        "limit": 0,
        "allow_missing": False,
        "concurrency": 1,
        "rpm": 1.0,
        "dry_run": False,
    }
    with pytest.raises(SystemExit) as e:
        judge.run(JudgeArgs(**base))
    assert e.value.code == 2
    r = judge.run(JudgeArgs(**{**base, "allow_missing": True}))
    assert r.counts["n_missing_cells"] == 1 and r.extras["n_rows_off_cell"] == 2
    with pytest.raises(SystemExit):
        judge.run(JudgeArgs(**{**base, "layers": [56, 60]}))


def test_registered_readme_and_dry_run(tmp_path, capsys, monkeypatch):
    from wsbench.evals.arithmetic_intermediates import SPEC

    assert SPEC.group == "computational" and SPEC.judge.prompt_version == PROMPT_VERSION
    readme = (REPO / "evals/arithmetic_intermediates/README.md").read_text(encoding="utf-8")
    assert SYSTEM in readme and SYSTEM_BATCH in readme and PROMPT_VERSION_BATCH in readme
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    argv = ["judge", f"family={SPEC.name}", f"readouts={EXAMPLE}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] == 592
    assert "Lens output:" in capsys.readouterr().out


_ENTRY = re.compile(r'^\[(\d+)\] pos=(-?\d+) token=("(?:\\.|[^"\\])*"): ', re.M)


def batch_entries_of(user: str) -> list[tuple[int, int, str]]:
    """(k, pos, text) of every entry in a batched user message, as the test judge reads it."""
    heads = list(_ENTRY.finditer(user))
    tail = user.rindex("\n\nFor each entry")
    out = []
    for n, m in enumerate(heads):
        stop = heads[n + 1].start() if n + 1 < len(heads) else tail
        out.append((int(m.group(1)), int(m.group(2)), user[m.end() : stop].strip()))
    return out


def test_all_cells_mode_uses_every_row_and_reports_bands(tmp_path, monkeypatch):
    """``opts=cells=all``: every (layer, pos) row is a cell, ONE call per (item, layer) lists
    them as numbered entries, an item passes on any cell, and the accuracy bands are computed
    over the same kept values."""
    _h, items = load_bank(BANK)
    it = next(x for x in items if x["tolerance"] == "exact")
    target = float(it["intermediates"][0])
    off = target * 1.03  # inside 5%, outside exact
    rows = [
        {"id": it["id"], "layer": 20, "pos": 0, "token": "a", "samples": ["nothing here"]},
        {"id": it["id"], "layer": 20, "pos": 1, "token": "\n", "samples": [f"maybe {off:g}"]},
        {"id": it["id"], "layer": 60, "pos": 1, "samples": [f"the value is {target:g}"]},
    ]
    seen: list[dict] = []

    def fake_run_calls(calls, **kw):
        seen.append(kw)
        out = {}
        for c in calls:
            entries = batch_entries_of(c.user)
            assert [k for k, _p, _t in entries] == list(range(1, len(entries) + 1))
            assert [p for _k, p, _t in entries] == c.meta["positions"]
            reply = []
            for k, _pos, text in entries:
                if f"{target:g}" in text:
                    reply.append({"k": k, "values": [target], "quote": f"{target:g}", "basis": "t"})
                elif f"{off:g}" in text:
                    reply.append({"k": k, "values": [off], "quote": f"{off:g}", "basis": "t"})
                else:
                    reply.append({"k": k, "values": [], "quote": "", "basis": "none"})
            out[c.key] = {"entries": reply}
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
    assert row["n_cells"] == 3 and row["pass"] is True and row["hits_at"] == [(60, 1)]
    assert row["bands"] == {"exact": True, "rel5pct": True}
    assert r.extras["n_rows_off_cell"] == 0 and r.extras["n_calls"] == 2  # (item, layer) pairs
    assert r.extras["n_calls_failed"] == 0 and r.counts["n_unjudged_cells"] == 0
    assert r.extras["cell_hit_rate"] == pytest.approx(1 / 3)
    assert r.extras["per_layer_hit_rate"] == {"20": 0.0, "60": 1.0}
    assert r.config["cells"] == "all" and r.config["layers_judged"] == [20, 60]
    assert r.config["prompt_version"] == PROMPT_VERSION_BATCH
    assert seen[0]["prompt_version"] == PROMPT_VERSION_BATCH and seen[0]["validate"] is not None
    # without the exact cell, only the 5% band is reached
    path.write_text("".join(json.dumps(x) + "\n" for x in rows[:2]))
    r2 = judge.run(JudgeArgs(**{**args.__dict__, "out": tmp_path / "out2"}))
    assert r2.rows[0]["pass"] is False and r2.extras["n_calls"] == 1
    assert r2.rows[0]["bands"] == {"exact": False, "rel5pct": True}
    assert r2.extras["bands"]["rel5pct"] == 1.0 and r2.extras["bands"]["exact"] == 0.0


def test_batched_replies_are_read_per_entry(tmp_path, monkeypatch):
    """A batched reply is mapped back to cells by ``k``: the first well-formed entry per k
    wins, malformed and missing entries leave their cells unjudged, a failed call leaves every
    cell of its (item, layer) unjudged, a reply without any usable entry is invalid."""
    good = {"k": 2, "values": [1.0], "basis": "t", "quote": ""}
    res = {
        "entries": [
            good,
            {"k": 2, "values": [2.0]},  # duplicate k: ignored
            {"k": "x", "values": []},  # non-integer k
            {"k": 5, "values": []},  # out of range
            {"k": 1.0, "values": [3.0]},  # an integral float is fine
            {"k": 3, "values": "no"},  # values not a list
            {"k": True, "values": []},
        ]
    }
    parsed = judge.batch_entries(res, 4)
    assert set(parsed) == {1, 2} and parsed[2] is good and parsed[1]["values"] == [3.0]
    assert judge.batch_entries(None, 3) == {} and judge.batch_entries({"entries": "x"}, 3) == {}
    call = judge.Call(key="k", system="", user="", meta={"n_entries": 4})
    assert judge.valid_batch(call, res) and not judge.valid_batch(call, {"entries": []})

    _h, items = load_bank(BANK)
    it = next(x for x in items if x["tolerance"] == "exact")
    target = float(it["intermediates"][0])
    rows = [
        {"id": it["id"], "layer": 56, "pos": -3, "samples": [f"it comes to {target + 1:g}"]},
        {"id": it["id"], "layer": 56, "pos": -2, "samples": [f"so {target:g}"]},
        {"id": it["id"], "layer": 56, "pos": -1, "samples": ["nothing"]},
        {"id": it["id"], "layer": 60, "pos": -1, "samples": ["nothing"]},
    ]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    base = {
        "readouts": path,
        "out": tmp_path / "out",
        "judge": resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        "layers": None,
        "items": [it["id"]],
        "limit": 0,
        "allow_missing": False,
        "concurrency": 1,
        "rpm": 1.0,
        "dry_run": False,
        "extra": {"cells": "all"},
    }
    # the layer-56 reply skips entry 2 (the hit), the layer-60 call fails
    monkeypatch.setattr(
        judge,
        "run_calls",
        lambda calls, **kw: {
            c.key: (
                None
                if c.meta["layer"] == 60
                else {
                    "entries": [
                        {"k": 1, "values": [target + 1], "basis": "t", "quote": ""},
                        {"k": 3, "values": [], "basis": "none", "quote": ""},
                    ]
                }
            )
            for c in calls
        },
    )
    r = judge.run(JudgeArgs(**base))
    row = r.rows[0]
    assert row["pass"] is None and row["n_cells"] == 4 and row["n_unjudged"] == 2
    assert r.counts["n_unjudged_cells"] == 2 and r.extras["n_calls_failed"] == 1
    assert r.extras["n_calls"] == 2 and r.complete is False
    assert row["unjudged_at"] == [(56, -2), (60, -1)]
    # a hit in one entry decides the item even with unjudged cells beside it
    monkeypatch.setattr(
        judge,
        "run_calls",
        lambda calls, **kw: {
            c.key: {
                "entries": [
                    {"k": 2, "values": [target], "basis": "t", "quote": ""},
                ]
            }
            if c.meta["layer"] == 56
            else None
            for c in calls
        },
    )
    r = judge.run(JudgeArgs(**{**base, "out": tmp_path / "out2"}))
    assert r.rows[0]["pass"] is True and r.rows[0]["hits_at"] == [(56, -2)]
    assert r.rows[0]["n_unjudged"] == 3 and r.value == 1.0


def test_example_file_carries_the_every_position_regime(tmp_path, monkeypatch):
    """Since 2026-09-23 the readouts of record hold every position at layers 56/60, judged
    ``opts=cells=all``: the toy file shows several positions of both layers per item, each item's
    frozen cell among them, ``pos`` a negative offset from the end of the render."""
    header, items = load_bank(BANK)
    assert "every position" in header["read"] and "cells=all" in header["read"]
    by_id = {it["id"]: it for it in items}
    rows = [json.loads(x) for x in EXAMPLE.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert {r["layer"] for r in rows} == {56, 60} and all(r["pos"] < 0 for r in rows)
    for i in {r["id"] for r in rows}:
        cells = {(r["layer"], r["pos"]) for r in rows if r["id"] == i}
        assert len(cells) >= 3 and len({p for _l, p in cells}) >= 2
        assert (by_id[i]["cell"]["layer"], by_id[i]["cell"]["pos"]) in cells
    assert any(not r["samples"] for r in rows)  # one real empty cell: a negative, not a gap
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    argv = [
        "judge",
        "family=arithmetic_intermediates",
        f"readouts={EXAMPLE}",
        f"out={out}",
        "dry_run=True",
        "opts=cells=all",
    ]
    assert main(argv) == 0
    r = read_results(out)
    assert r.config["cells"] == "all" and r.config["layers_judged"] == [56, 60]
    # a dry run reports the 592 items without readouts as missing rather than failing
    assert r.counts["n_expected_cells"] - r.counts["n_missing_cells"] == len(rows)
    assert r.counts["n_empty_cells"] == 1
    assert r.extras["n_rows_off_cell"] == 0
