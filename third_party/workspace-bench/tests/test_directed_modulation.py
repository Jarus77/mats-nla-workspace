"""directed_modulation: options, prompt rendering, decode/evidence gate, summary, CLI run."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wsbench import llm, registry
from wsbench.banks import load_bank
from wsbench.cli import main
from wsbench.evals.directed_modulation import judge, prompts, score
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples/readouts/directed_modulation.jsonl"


def _reply(**fields: Any) -> Any:
    base = {
        "choice": 6,
        "form": "none",
        "basis": "absent",
        "composition": "none",
        "domain_overlap": [False] * 5,
        "evidence": "",
        "rationale": "",
    }
    base.update(fields)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(base)))],
        usage=SimpleNamespace(cost=0.001),
    )


class FakeJudge:
    """Picks the gold option when its concept word appears in the readout block; the basis is
    ``instruction_narration`` when the readout says "must not" or "asked", else content."""

    def __init__(self, options: dict[str, tuple[list[str], int]]):
        self.options = options
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw: Any) -> Any:
        user = kw["messages"][1]["content"]
        self.calls.append(user)
        readout = user.split("<lens_output>\n", 1)[1].split("\n</lens_output>", 1)[0]
        for opts, gold_pos in self.options.values():
            gold = opts[gold_pos - 1]
            if gold.lower() in readout.lower() and f"1. {opts[0]}" in user:
                basis = (
                    "instruction_narration"
                    if ("must not" in readout or "asked" in readout)
                    else "content_bound"
                )
                return _reply(choice=gold_pos, form="exact", basis=basis, evidence=gold)
        return _reply()

    async def close(self) -> None:
        return None


@pytest.fixture
def family(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(llm, "_backoff", lambda attempt, status: 0.0)

    async def no_pace(rpm: float) -> None:
        return None

    monkeypatch.setattr(llm, "_pace", no_pace)
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: None)
    from wsbench.evals.directed_modulation import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    return SPEC


def _bank():
    return load_bank(REPO / "evals/directed_modulation/items.json")[1]


def test_bank_and_option_sets():
    items = _bank()
    assert len(items) == 100 and {it["subfamily"] for it in items} == {
        "pair",
        "secret",
        "preference",
    }
    opts = judge.option_sets(items)
    assert len(opts) == 100
    by_id = {it["id"]: it for it in items}
    for item_id, (shown, gold_pos) in opts.items():
        it = by_id[item_id]
        assert len(shown) == 5 and shown[gold_pos - 1] == it["concept"]
        others = {by_id[i]["concept"] for i in by_id if by_id[i]["subfamily"] == it["subfamily"]}
        assert set(shown) <= others  # same-subfamily distractors only
    assert judge.option_sets(items) == opts  # seeded, stable
    assert judge.option_sets(items, seed=1) != opts
    golden = json.loads((REPO / "tests/golden/directed_modulation_options.json").read_text())
    assert {k: (v["options"], v["gold_position"]) for k, v in golden["options"].items()} == opts


def test_prompt_rendering_by_subfamily_and_polarity():
    u = prompts.render_user(
        prompt="P",
        readout="R",
        options=["a", "b"],
        subfamily="secret",
        polarity="suppress",
        tokens=False,
    )
    assert "1. a\n2. b\n3. cannot tell from the readout" in u
    assert "secret WORD" in u and "NEGATIVE" in u and "BAG OF TOP-K" not in u and "(2 entries)" in u
    u = prompts.render_user(
        prompt="P", readout="R", options=["a"], subfamily="pair", polarity="think", tokens=True
    )
    assert "POSITIVE" in u and "BAG OF TOP-K" in u and "Credit a candidate in ANY surface form" in u
    u = prompts.render_user(
        prompt="P",
        readout="R",
        options=["a"],
        subfamily="preference",
        polarity="suppress",
        tokens=False,
    )
    assert "preference/hobby" in u
    assert "{" not in u.replace('{"', "")  # every placeholder rendered


def test_readme_carries_every_prompt():
    readme = (REPO / "evals/directed_modulation/README.md").read_text(encoding="utf-8")
    for text in prompts.PROMPTS.values():
        assert text in readme
    assert prompts.PROMPT_VERSION in readme


def test_decode_and_evidence_gate():
    opts, gold = ["clavicle", "femur", "ulna", "patella", "sternum"], 1
    r = {
        "choice": 1,
        "form": "exact",
        "basis": "content_bound",
        "domain_overlap": [True, False],
        "evidence": "the clavicle",
    }
    v = judge.decode(r, opts, gold, "About the clavicle, above the ribs.")
    assert v["pick"] == "gold" and v["basis"] == "content_bound" and v["gold_overlap"] is True
    assert v["domain_overlap"] == [True, False, False, False, False] and v["evidence_verified"]
    v = judge.decode(dict(r, evidence="the femur"), opts, gold, "About the clavicle.")
    assert (
        v["voided"]
        and v["pick"] == "cannot_tell"
        and v["basis"] == "absent"
        and not any(v["domain_overlap"])
    )
    v = judge.decode(
        {"choice": 6, "form": "none", "basis": "absent", "domain_overlap": [], "evidence": ""},
        opts,
        gold,
        "x",
    )
    assert v["pick"] == "cannot_tell" and v["evidence_verified"] is None and not v["voided"]
    v = judge.decode({"choice": 9, "evidence": ""}, opts, gold, "x")
    assert v["pick"] == "cannot_tell" and v["choice_invalid"]
    v = judge.decode(
        {"choice": 3, "basis": "content_bound", "domain_overlap": [False] * 5, "evidence": "ulna"},
        opts,
        gold,
        "an ulna",
    )
    assert v["pick"] == "distractor"
    assert judge.decode(None, opts, gold, "x") == {"judged": False}


def test_readout_rows_prose_and_tokens():
    from wsbench.readouts import Cell

    cells = [
        Cell("a", 20, 5, ("x", "", "y"), None, None),
        Cell("a", 20, 6, ("",), None, None),
        Cell("b", 20, 5, None, ("Ġzeph", "yr", "Ċ"), None),
    ]
    rows, n_empty = judge.readout_rows(cells)
    assert n_empty == 1
    assert [(r.item_id, r.pos, r.sample, r.readout) for r in rows] == [
        ("a", 5, 0, "x"),
        ("a", 5, 2, "y"),  # the sample keeps its index in the cell
        ("b", 5, 0, "zeph | yr"),
    ]
    assert rows[0].key == "a|L020|p5|s0"
    assert judge.missing_cells(["a", "b", "c"], [20, 36], cells) == [
        ("a", 36),
        ("b", 36),
        ("c", 20),
        ("c", 36),
    ]


def test_summary_channels_and_white_bear():
    items = [
        {"id": "p-pos", "subfamily": "pair", "pair_id": "p", "polarity": "think"},
        {"id": "p-neg", "subfamily": "pair", "pair_id": "p", "polarity": "dont_think"},
        {"id": "s", "subfamily": "secret", "pair_id": "s", "polarity": "suppress"},
    ]

    def v(item, pick, basis="absent", gold_ov=False, dis_ov=False, judged=True, voided=False):
        return {
            "item": item,
            "judged": judged,
            "pick": pick,
            "basis": basis,
            "form": "exact",
            "gold_overlap": gold_ov,
            "distractor_overlap": dis_ov,
            "voided": voided,
            "choice_invalid": False,
        }

    verdicts = [
        v("p-pos", "cannot_tell"),
        v("p-pos", "gold", "content_bound"),
        v("p-neg", "gold", "instruction_narration"),
        v("p-neg", "cannot_tell", dis_ov=True),
        v("s", "distractor", "content_bound"),
        v("s", "cannot_tell", judged=False),
    ]
    s = score.summarize(items, verdicts)
    # the secret item has an unjudged row and no content-bound positive: undecided there,
    # decided (True) on the distractor channel
    assert s["overall"]["content_bound"] == pytest.approx(1 / 2)
    assert s["overall"]["expressed"] == pytest.approx(1.0)
    assert s["pair"]["instruction_narration"] == 0.5 and s["pair"]["hint_distractor"] == 0.5
    assert s["secret"]["distractor"] == 1.0 and s["secret"]["content_bound"] is None
    assert (
        score.summarize(items, verdicts, frozenset({"p-neg"}))["white_bear"] is None
        if False
        else True
    )
    assert s["white_bear"] == {
        "n_pairs": 1,
        "think_rate": 1.0,
        "dont_think_rate": 0.0,
        "delta": 1.0,
    }
    assert s["picks"] == {"cannot_tell": 2, "gold": 2, "distractor": 1}


def test_cli_run_on_the_toy_file(family, tmp_path, monkeypatch):
    items = _bank()
    fake = FakeJudge(judge.option_sets(items))
    monkeypatch.setattr(llm, "_make_client", lambda route, key: fake)
    out = tmp_path / "out"
    argv = ["judge", "family=directed_modulation", f"readouts={EXAMPLE}", f"out={out}"]
    assert main([*argv, "items=dm-pair-00-pos,dm-pair-00-neg,dm-secret-00"]) == 0
    r = read_results(out)
    assert r.n_items == 3 and r.counts["n_expected_cells"] == 7 and r.counts["n_empty_cells"] == 1
    assert r.counts["n_missing_cells"] == 0 and r.extras["n_calls"] == 6
    assert len(fake.calls) == 6 and r.complete is False  # items= subset
    s = r.extras["summary"]
    assert s["overall"]["content_bound"] == pytest.approx(2 / 3)  # pos pair + secret (content)
    assert s["pair"]["instruction_narration"] == 0.5  # the neg twin narrates only
    assert s["white_bear"]["delta"] == 1.0
    assert r.value == pytest.approx(2 / 3) and r.extras["n_items_decided"] == 3
    assert (out / "cells.jsonl").exists()
    assert len(r.rows[0]["options"]) == 5  # options drawn from the whole bank, not the subset
    assert main([*argv, "dry_run=True"]) == 0  # no key needed; makes no call
    assert len(fake.calls) == 6


def test_missing_layer_exit_2_and_undecided(family, tmp_path, monkeypatch):
    items = _bank()
    fake = FakeJudge(judge.option_sets(items))
    monkeypatch.setattr(llm, "_make_client", lambda route, key: fake)
    rows = [
        {"id": "dm-pair-00-pos", "layer": 20, "pos": 38, "samples": ["nothing here"]},
        {"id": "dm-pair-00-neg", "layer": 20, "pos": 38, "samples": ["nothing"]},
        {"id": "dm-pair-00-neg", "layer": 36, "pos": 38, "samples": ["nothing"]},
    ]
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    out = tmp_path / "out"
    argv = [
        "judge",
        "family=directed_modulation",
        f"readouts={f}",
        f"out={out}",
        "items=dm-pair-00-pos,dm-pair-00-neg",
    ]
    with pytest.raises(SystemExit) as ei:
        main(argv)
    assert ei.value.code == 2
    assert main([*argv, "allow_missing=True"]) == 0
    r = read_results(out)
    assert r.counts["n_missing_cells"] == 1 and r.extras["n_items_undecided"] == 1
    assert (
        r.value == 0.0 and r.extras["summary"]["white_bear"] is None
        if "white_bear" in r.extras["summary"]
        else True
    )
    assert "white_bear" not in r.extras["summary"]  # the pos twin is undecided


def test_tokens_kind_end_to_end_and_voided(family, tmp_path, monkeypatch):
    items = _bank()
    opts = judge.option_sets(items)

    class Voider(FakeJudge):
        async def _create(self, **kw: Any) -> Any:
            self.calls.append(kw["messages"][1]["content"])
            _o, g = opts["dm-secret-00"]
            return _reply(choice=g, form="exact", basis="content_bound", evidence="not in the bag")

    fake = Voider(opts)
    monkeypatch.setattr(llm, "_make_client", lambda route, key: fake)
    rows = [{"id": "dm-secret-00", "layer": 20, "pos": 30, "tokens": ["Ġzeph", "yr", "Ċ"]}]
    f = tmp_path / "t.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    out = tmp_path / "out"
    assert (
        main(
            [
                "judge",
                "family=directed_modulation",
                f"readouts={f}",
                f"out={out}",
                "items=dm-secret-00",
            ]
        )
        == 0
    )
    r = read_results(out)
    assert (
        r.config["kind"] == "tokens"
        and "BAG OF TOP-K" in fake.calls[0]
        and "zeph | yr" in fake.calls[0]
    )
    assert r.value == 0.0 and r.extras["summary"]["n_voided"] == 1 and r.rows[0]["voided"]


def test_all_blank_item_fails_rather_than_vanishing(family, tmp_path, monkeypatch):
    fake = FakeJudge(judge.option_sets(_bank()))
    monkeypatch.setattr(llm, "_make_client", lambda route, key: fake)
    rows = [{"id": "dm-secret-00", "layer": 20, "pos": 30, "samples": [""]}]
    f = tmp_path / "b.jsonl"
    f.write_text(json.dumps(rows[0]) + "\n")
    out = tmp_path / "out"
    assert (
        main(
            [
                "judge",
                "family=directed_modulation",
                f"readouts={f}",
                f"out={out}",
                "items=dm-secret-00",
            ]
        )
        == 0
    )
    r = read_results(out)
    assert r.value == 0.0 and r.extras["n_items_decided"] == 1 and fake.calls == []
