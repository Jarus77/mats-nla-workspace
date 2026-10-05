"""multi_concept_directed_modulation: option lists, the region rule, the verdict, the read.json
converter, a scripted toy run and a dry run."""

import json
from pathlib import Path

import pytest

from wsbench import registry
from wsbench.banks import load_bank
from wsbench.cli import main
from wsbench.evals.multi_concept_directed_modulation import judge
from wsbench.evals.multi_concept_directed_modulation.options import option_sets, partner_of
from wsbench.evals.multi_concept_directed_modulation.prompts import (
    LETTERS,
    N_OPTIONS,
    PROMPT_VERSION,
)
from wsbench.evals.multi_concept_directed_modulation.regions import (
    TARGET_SENTENCE,
    Region,
    classify_regions,
    label_cells,
    write_compliance,
)
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.readouts import convert_read_json, load_readouts
from wsbench.registry import JudgeArgs
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "evals/multi_concept_directed_modulation/items.json"
EXAMPLE = REPO / "examples/readouts/multi_concept_directed_modulation.jsonl"

# the shape every binding item's completion had under the pre-2026-09-23 rollout regime: sentence,
# blank line, empty think block, then the model re-writing the prompt (the region rule is kept)
WITH_THINK = [
    "review",
    " the",
    " annual",
    " budget",
    '."',
    "\n\n",
    "<think>",
    "\n\n",
    "</think>",
    "\n\n",
    "Think",
    " about",
]
# d-cranes: no think block, a self-added continuation as the tail
NO_THINK = ["review", " the", " annual", " budget", '."', "\n\n", "Now", " write", " this"]
# d-petrichor: the model ignored the instruction and wrote its own sentence
OFF_TASK = ["is", " called", " petrichor", ".", " It", " is", " a", " scent", " that", " many"]


def test_bank_and_partners():
    header, items = load_bank(BANK)
    assert header["family"] == "multi-concept-directed-modulation" and len(items) == 27
    controls = [it["name"] for it in items if not it["concepts"]]
    assert controls == ["d-none", "b-none"]
    assert partner_of("d-fox-a") == "d-fox-b" and partner_of("b-ange-mix-ba") == "b-ange-mix-ab"
    assert partner_of("d-plumber") is None and partner_of("d-none") is None
    names = {it["name"] for it in items}
    with_partner = [it for it in items if partner_of(it["name"]) in names]
    assert len(with_partner) == 18  # d-fox-a/b and the sixteen binding items


def test_options_match_the_golden_and_carry_gold_and_partner():
    _h, items = load_bank(BANK)
    opts = option_sets(items)
    golden = json.loads(
        (REPO / "tests/golden/multi_concept_directed_modulation_options.json").read_text("utf-8")
    )
    assert {i: (v["options"], v["gold_indices"]) for i, v in golden["options"].items()} == opts
    by_name = {it["name"]: it for it in items}
    for it in items:
        o, g = opts[it["id"]]
        assert len(o) == N_OPTIONS and len(set(o)) == N_OPTIONS
        assert [o[i] for i in g] == it["concepts"]
        partner = by_name.get(partner_of(it["name"]) or "")
        if partner:
            for c in partner["concepts"]:
                assert c in o
    assert opts["d-none"][1] == []


def test_regions_with_and_without_a_think_block():
    got = classify_regions(WITH_THINK)
    assert got[:5] == [Region.IN_SENTENCE] * 5
    assert got[5:9] == [Region.IN_THINK] * 4  # the blank line before <think> is not in-sentence
    assert got[9:] == [Region.POST_THINK] * 3
    assert classify_regions(NO_THINK) == [Region.IN_SENTENCE] * 5 + [Region.SELF_ADDED_META] * 4
    assert classify_regions(["meet", " on", " Thursday"]) == [Region.IN_SENTENCE] * 3


def test_write_compliance_and_off_task():
    assert write_compliance(WITH_THINK).complied
    bad = write_compliance(OFF_TASK)
    assert not bad.complied and bad.run < 3
    labels, comp = label_cells(OFF_TASK)
    assert not comp.complied and set(labels) == {Region.OFF_TASK}


def test_verdict_quote_gate_and_direction():
    options = [
        "Adam being angry at Betty",
        "wet umbrella",
        "Betty being angry at Adam",
        "x",
        "y",
        "z",
    ]
    readout = "- Adam is furious with Betty about the meeting\n- an umbrella, dripping wet"
    v = judge.verdict(
        {
            "picks": [
                {"choice": "A", "quote": "Adam is furious with Betty"},
                {"choice": "B", "quote": "umbrella, dripping wet"},
            ]
        },
        options,
        [0],
        readout,
    )
    assert v["kind"] == "hit" and v["own"] == [options[0]] and v["false"] == ["wet umbrella"]
    # a pick without a verbatim quote is not credited; free text is not a letter
    v = judge.verdict(
        {
            "picks": [
                {"choice": "A", "quote": "not in the readout"},
                {"choice": "none", "quote": ""},
            ]
        },
        options,
        [0],
        readout,
    )
    assert v["kind"] == "none" and v["n_unverified"] == 1 and v["n_invalid"] == 1
    # the reversed binding is a false pick, not a hit
    v = judge.verdict(
        {"picks": [{"choice": "C", "quote": "Adam is furious"}]}, options, [0], readout
    )
    assert v["kind"] == "false_pick" and v["false"] == [options[2]]
    assert judge.verdict({"picks": []}, options, [0], readout)["kind"] == "none"
    assert judge.verdict({"picks": "A"}, options, [0], readout)["kind"] == "invalid"
    assert judge.verdict(None, options, [0], readout)["judged"] is False


def test_convert_read_json_round_trip(tmp_path):
    read = {
        "layers": [44, 52],
        "records": [
            {
                "name": "d-x*",
                "rels": [-1, -2],
                "tokens": [" budget", " annual"],
                "ao": {"44": [["a"], ["b"]], "52": [["c"], [""]]},
            },
            {"name": "bad", "rels": [-1], "tokens": [], "ao": {}},
        ],
    }
    src = tmp_path / "read.json"
    src.write_text(json.dumps(read))
    out = tmp_path / "r.jsonl"
    rep = convert_read_json(src, out)
    cells, _ = load_readouts(out)
    assert rep.skipped["malformed"] == 1 and len(cells) == 4 and rep.n_empty == 1
    assert {c.id for c in cells} == {"d-x_"}
    assert sorted((c.layer, c.pos, c.token) for c in cells) == [
        (44, -2, " annual"),
        (44, -1, " budget"),
        (52, -2, " annual"),
        (52, -1, " budget"),
    ]


def test_toy_file_regions():
    """The toy file is three items of the s3d read of record (prefill regime, 2026-09-23): d-plumber
    (novel), b-ange-mix-ab (binding) and d-none (control); the window is the 12 prefilled sentence
    tokens, so every cell is in-sentence."""
    cells, rep = load_readouts(EXAMPLE)
    assert rep.kind == "prose" and rep.layers == [44, 52, 56, 60]
    assert all(c.token is not None for c in cells)
    for item in ("d-plumber", "b-ange-mix-ab", "d-none"):
        toks = {c.pos: c.token for c in cells if c.id == item}
        assert sorted(toks) == list(range(-12, 0))
        assert "".join(toks[p] for p in sorted(toks)) == TARGET_SENTENCE
        labels, comp = label_cells([toks[p] for p in sorted(toks)])
        assert comp.complied and labels == [Region.IN_SENTENCE] * 12


def test_toy_run_with_a_scripted_judge(tmp_path, monkeypatch):
    _h, items = load_bank(BANK)
    opts = option_sets(items)
    by_name = {it["name"]: it for it in items}
    plumber, binding, control = "d-plumber", "b-ange-mix-ab", "d-none"
    reverse = by_name["b-ange-mix-ba"]["concepts"][0]

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            item, layer = c.meta["item"], c.meta["layer"]
            o, g = opts[item]
            readout = c.user.split('"""\n', 1)[1].split('\n"""', 1)[0]
            if item == plumber and layer == 44:
                # credit two dictated concepts at the first cell, one at the others
                own = [LETTERS[i] for i in g]
                picks = [{"choice": own[0], "quote": readout[:12]}]
                if c.meta["pos"] == min(
                    m["pos"] for m in (x.meta for x in calls) if m["item"] == item
                ):
                    picks.append({"choice": own[1], "quote": readout[:12]})
                out[c.key] = {"picks": picks}
            elif item == binding:
                out[c.key] = {
                    "picks": [{"choice": LETTERS[o.index(reverse)], "quote": readout[:12]}]
                }
            elif item == control and layer == 60:
                out[c.key] = {"picks": [{"choice": "A", "quote": readout[:12]}]}
            else:
                out[c.key] = {"picks": []}
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    args = JudgeArgs(
        readouts=EXAMPLE,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=[plumber, binding, control],
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
    )
    r = judge.run(args)
    rows = {row["id"]: row for row in r.rows}
    assert rows[plumber]["pass"] is True and rows[plumber]["capacity"]["per_activation_max"] == 2
    assert rows[plumber]["capacity"]["per_item_union"] == 2
    assert rows[binding]["pass"] is False and rows[binding]["partner_picked"] is True
    assert rows[binding]["false_picks"] == [reverse]
    assert rows[control]["pass"] is None and rows[control]["reason"] == "control"
    assert r.value == 0.5 and r.n_items == 3
    assert r.extras["n_controls"] == 1 and r.extras["control_pick_rate"] == pytest.approx(0.25)
    assert r.extras["partner_confusion_rate"] == 1.0
    assert r.extras["per_stratum"] == {"mix": 0.0, "novel": 1.0}
    assert r.counts["n_missing_cells"] == 0 and r.counts["n_unjudged_cells"] == 0
    assert r.extras["n_calls"] == r.counts["n_expected_cells"] - r.counts["n_empty_cells"]
    assert not r.complete  # a subset run


def test_rows_without_tokens_are_fatal(tmp_path, monkeypatch):
    rows = [{"id": "d-plumber", "layer": 44, "pos": -1, "samples": ["x"]}]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = JudgeArgs(
        readouts=path,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=None,
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=True,
    )
    with pytest.raises(SystemExit) as e:
        judge.run(args)
    assert e.value.code == 2


def test_registered_and_dry_run(tmp_path, capsys, monkeypatch):
    from wsbench.evals.multi_concept_directed_modulation import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    assert SPEC.group == "basic_mt" and SPEC.judge.prompt_version == PROMPT_VERSION
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    argv = ["judge", f"family={SPEC.name}", f"readouts={EXAMPLE}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] == 24
    assert "[user]" in capsys.readouterr().out
