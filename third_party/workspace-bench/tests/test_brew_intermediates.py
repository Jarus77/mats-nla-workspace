"""brew_intermediates: the bank's cells and regions, the screen, the rule, a scripted run."""

import json
from pathlib import Path

import pytest

from wsbench import registry
from wsbench.banks import load_bank
from wsbench.cli import main
from wsbench.evals.brew_intermediates import judge
from wsbench.evals.brew_intermediates.prompts import (
    COLOURS,
    PROMPT_VERSION,
    SYSTEM,
    mentions_any_colour,
    render_user,
)
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.registry import JudgeArgs
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "evals/brew_intermediates/items.json"
EXAMPLE = REPO / "examples/readouts/brew_intermediates.jsonl"


@pytest.fixture(autouse=True)
def _registered():
    from wsbench.evals.brew_intermediates import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)


def test_bank_cells_regions_and_options():
    header, items = load_bank(BANK)
    assert header["family"] == "brew_intermediates" and len(items) == 100
    for it in items:
        regions = it["regions"]
        assert len(regions) == 24 and set(map(int, regions)) == set(it["eval_positions"])
        counts = {r: sum(1 for v in regions.values() if v == r) for r in set(regions.values())}
        assert counts == {
            "stir": 6,
            "start": 6,
            "question": 9,
            "emit_asst": 1,
            "emit_think": 1,
            "emit_colon": 1,
        }
        opts = it["options_adjacent"][0]
        gold, start, answer = it["intermediates"][0], it["start"], it["answer"]
        assert len(opts) == 5 and len(set(opts)) == 5 and {gold, start, answer} <= set(opts)
        assert all(c in COLOURS for c in opts) and len({gold, start, answer}) == 3
        assert len([c for c in opts if c not in (gold, start, answer)]) == 2


def test_screen_and_prompt():
    assert mentions_any_colour("the potion turns Greenish") and mentions_any_colour("颜色是红")
    assert not mentions_any_colour("the committee meets on Thursday")
    assert not mentions_any_colour("a goldfinch")  # word boundary: not a colour mention
    user = render_user("x" * 5000, ["red", "blue"])
    assert "Candidate colours: red, blue" in user and "x" * 4001 not in user  # truncated


def test_shuffle_is_seeded_and_option_preserving():
    opts = ["red", "black", "green", "purple", "blue"]
    a = judge.shuffled(opts, "item|20|131")
    assert sorted(a) == sorted(opts) and a == judge.shuffled(opts, "item|20|131")
    assert a != judge.shuffled(opts, "item|24|131") or a != judge.shuffled(opts, "other|20|131")


def test_verdict_filters_to_options():
    options, gold, offs = ["red", "black", "green", "purple", "blue"], "green", ["purple", "red"]
    v = judge.verdict(
        {"named": ["green", "teal", "red"], "primary": "teal", "basis": "explicit"},
        options,
        gold,
        offs,
    )
    assert v["named"] == ["green", "red"] and v["gold"] and v["offs"] == [False, True]
    assert v["primary"] == "none"
    v = judge.verdict(
        {"named": "green", "primary": "green", "basis": "explicit"}, options, gold, offs
    )
    assert v["named"] == [] and not v["gold"] and v["primary"] == "green"
    assert judge.verdict(None, options, gold, offs)["judged"] is False


def _cell(gold, offs, region="emit_asst"):
    return {"gold": gold, "offs": offs, "region": region, "judged": True}


def test_rule_pass_null_baseline():
    # gold named in 3 cells, offs in 1 and 2: 2*3 > 3 -> pass; each off in the gold's role fails
    cells = [_cell(True, [True, False]), _cell(True, [False, True]), _cell(True, [False, True])]
    r = judge.rule(cells)
    assert r["G"] == 3 and r["O"] == [1, 2] and r["pass"] and r["null"] == 0.0
    assert r["baseline"] == pytest.approx(1 / 3)
    # a tie fails: gold 2, offs 2 and 2 -> 2*2 > 4 is False
    tie = [_cell(True, [True, True]), _cell(True, [True, True])]
    assert not judge.rule(tie)["pass"] and judge.rule(tie)["null"] == 0.0
    # spraying every colour: nobody clears the rule
    spray = [_cell(True, [True, True])] * 5
    assert not judge.rule(spray)["pass"] and judge.rule(spray)["baseline"] == 0.0
    # one off colour dominates: pass False, null 0.5 (one of two offs clears it)
    off = [_cell(False, [True, False])] * 4 + [_cell(True, [False, False])]
    r = judge.rule(off)
    assert not r["pass"] and r["null"] == 0.5 and r["baseline"] == pytest.approx(1 / 3)
    assert judge.rule([]) is None


def test_scripted_run_headline_regions(tmp_path, monkeypatch):
    _h, items = load_bank(BANK)
    by_id = {it["id"]: it for it in items}
    ids = sorted({json.loads(x)["id"] for x in EXAMPLE.read_text().splitlines() if x.strip()})
    a, b = ids

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            it = by_id[c.meta["item"]]
            gold, start = it["intermediates"][0], it["start"]
            region = it["regions"][str(c.meta["pos"])]
            if c.meta["item"] == a and region.startswith("emit"):
                out[c.key] = {"named": [gold, start], "primary": gold, "basis": "explicit"}
            elif c.meta["item"] == a:
                out[c.key] = {"named": [start], "primary": start, "basis": "explicit"}
            else:
                offs = [
                    x for x in it["options_adjacent"][0] if x not in (gold, start, it["answer"])
                ]
                out[c.key] = {"named": offs[:1], "primary": offs[0], "basis": "implied"}
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    args = JudgeArgs(
        readouts=EXAMPLE,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=[a, b],
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
    )
    r = judge.run(args)
    rows = {row["id"]: row for row in r.rows}
    assert rows[a]["pass"] is True and rows[a]["emit"]["G"] > 0 and rows[a]["stir"]["G"] == 0
    assert rows[b]["pass"] is False and rows[b]["emit"]["G"] == 0
    assert r.value == 0.5 and r.counts["n_expected_cells"] == 2 * 9 * 11
    assert r.extras["regions_judged"] == ["emit_asst", "emit_colon", "emit_think", "stir"]
    assert r.extras["n_screened_cells"] > 0  # the toy's colour-free lines never reach the judge
    assert r.extras["n_calls"] + r.extras["n_screened_cells"] == r.counts["n_expected_cells"]
    assert r.config["regions"] == "headline"


def test_regions_option_and_bad_value(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "run_calls", lambda calls, **kw: {c.key: None for c in calls})
    ids = sorted({json.loads(x)["id"] for x in EXAMPLE.read_text().splitlines() if x.strip()})
    base = {
        "readouts": EXAMPLE,
        "out": tmp_path / "out",
        "judge": resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        "layers": None,
        "items": ids,
        "limit": 0,
        "allow_missing": False,
        "concurrency": 1,
        "rpm": 1.0,
        "dry_run": False,
    }
    r = judge.run(JudgeArgs(**base, extra={"regions": "all"}))
    assert r.counts["n_expected_cells"] == 2 * 24 * 11 and r.value is None  # every call failed
    with pytest.raises(SystemExit) as e:
        judge.run(JudgeArgs(**base, extra={"regions": "emit"}))
    assert e.value.code == 2


def test_registered_readme_and_dry_run(tmp_path, capsys, monkeypatch):
    from wsbench.evals.brew_intermediates import SPEC

    assert SPEC.group == "computational" and SPEC.judge.prompt_version == PROMPT_VERSION
    assert SYSTEM in (REPO / "evals/brew_intermediates/README.md").read_text(encoding="utf-8")
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    argv = ["judge", f"family={SPEC.name}", f"readouts={EXAMPLE}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] == 98
    assert "Candidate colours:" in capsys.readouterr().out


def test_blank_emission_cell_counts_and_failed_call_undecides(tmp_path, monkeypatch):
    _h, items = load_bank(BANK)
    it = items[0]
    emit_pos = [int(p) for p, r in it["regions"].items() if r.startswith("emit")]
    stir_pos = [int(p) for p, r in it["regions"].items() if r == "stir"]
    rows = []
    for layer in range(20, 61, 4):
        for p in emit_pos + stir_pos:
            text = "" if (layer == 20 and p == emit_pos[0]) else f"- the potion is {it['start']}"
            rows.append({"id": it["id"], "layer": layer, "pos": p, "samples": [text]})
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    calls_seen = []

    def fake_run_calls(calls, **kw):
        calls_seen.extend(calls)
        out = {
            c.key: {"named": [it["start"]], "primary": it["start"], "basis": "explicit"}
            for c in calls
        }
        emit_calls = [c for c in calls if c.meta["pos"] in emit_pos]
        first = min(emit_calls, key=lambda c: (c.meta["layer"], c.meta["pos"]))
        out[first.key] = None  # one emission call failed
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
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
    )
    r = judge.run(args)
    row = r.rows[0]
    assert r.counts["n_empty_cells"] == 1 and row["emit"]["n_cells"] == 33  # the blank counts
    assert row["pass"] is None and r.counts["n_unjudged_cells"] == 1  # a failed emission call
    assert row["stir"]["G"] == 0 and r.value is None
    # allow_missing scores what is there when a layer is absent
    short = [x for x in rows if x["layer"] != 60]
    path.write_text("".join(json.dumps(x) + "\n" for x in short))
    monkeypatch.setattr(
        judge,
        "run_calls",
        lambda calls, **kw: {
            c.key: {"named": [], "primary": "none", "basis": "none"} for c in calls
        },
    )
    full = JudgeArgs(**{**args.__dict__, "layers": list(range(20, 61, 4))})
    with pytest.raises(SystemExit):
        judge.run(full)
    r = judge.run(JudgeArgs(**{**full.__dict__, "allow_missing": True}))
    assert r.counts["n_missing_cells"] == 9 and r.rows[0]["pass"] is None
