import json
import sys
from pathlib import Path

import pytest
from conftest import tag_of, write_jsonl

from wsbench.evals.user_modeling import judge, prompts
from wsbench.evals.user_modeling.judge import build_options, decode_choice
from wsbench.evals.user_modeling.prompts import CANNOT_TELL, UM_OPTION_SUPPLEMENT
from wsbench.mcjudge import load_bank

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/golden/user_modeling_options.json").read_text())
EXAMPLE = ROOT / "examples/readouts/user_modeling.jsonl"
A, B, C = "um-sd-country-00", "um-sd-country-01", "um-ss-agegroup-05"  # C is synthsys


def _items() -> list[dict]:
    return load_bank("user_modeling")["items"]


def test_bank_loads():
    bank = load_bank("user_modeling")
    assert set(bank) == {"family", "gate", "read", "items"} and bank["family"] == "user-modeling"
    items = bank["items"]
    assert len(items) == 100
    for it in items:
        assert {"name", "subfamily", "attr_class", "attr", "messages"} <= set(it)
        assert it["messages"][0]["role"] == "system"
    assert sum(it["subfamily"] == "synthsys" for it in items) == 53
    assert len({it["name"] for it in items}) == 100


def test_options_match_source_golden():
    items = _items()
    opts = build_options(items)
    assert len(GOLDEN) == 100 and set(opts) == set(GOLDEN)
    for it in items:
        options, gold_position = opts[it["name"]]
        assert {"options": options, "gold_position": gold_position} == GOLDEN[it["name"]], it
        assert len(options) == 5 and options[gold_position - 1] == it["attr"].replace("_", " ")
        assert "_" not in "".join(options)
    # sparse classes are topped up from the supplement, never from another class
    c = next(it for it in items if it["name"] == C)
    assert set(opts[C][0]) - {c["attr"].replace("_", " ")} <= set(UM_OPTION_SUPPLEMENT["AgeGroup"])
    block = prompts.option_lines(opts[A][0])
    assert block.splitlines()[-1] == f"6. {CANNOT_TELL}" and len(block.splitlines()) == 6
    assert CANNOT_TELL not in opts[A][0]


def test_rendered_prompt_matches_golden():
    items = _items()
    options, _ = build_options(items)[A]
    user = prompts.render_user(
        prompts.render_user_turn(items[0]["messages"]), "The user is Danish.", options
    )
    assert user == (ROOT / "tests/golden/user_modeling_prompt.txt").read_text(encoding="utf-8")
    assert user.startswith('You are analyzing the output of an "oracle lens"')
    assert "[system] You are a helpful assistant." in user
    assert user.count(f'"{CANNOT_TELL}"') == 2 and "{" not in user
    assert prompts.PROMPT_VERSION == "um-v2"
    assert prompts.UM_ATTRIBUTE_SCHEMA["name"] == "um_attribute"
    assert prompts.UM_ATTRIBUTE_SCHEMA["schema"]["required"] == [
        "choice",
        "basis",
        "evidence",
        "rationale",
    ]


def test_source_repo_never_imported():
    assert not any(m.startswith("global_workspace") for m in sys.modules)
    assert not any("global-workspace" in p for p in sys.path)


def test_decode_choice_rules():
    options = ["a", "b", "c", "d", "e"]
    assert decode_choice(3, options, 3) == ("gold", "c", False)
    assert decode_choice(1, options, 3) == ("distractor", "a", False)
    assert decode_choice(6, options, 3) == ("cannot_tell", CANNOT_TELL, False)
    for bad in (0, 7, -1, "3", None, 2.0, True):
        assert decode_choice(bad, options, 3) == ("cannot_tell", CANNOT_TELL, True), bad


def _scripted(tmp_path):
    rows = [
        {"id": A, "layer": 20, "pos": 0, "samples": ["R:A:L20:s0", "", "R:A:L20:s2"]},
        {"id": A, "layer": 44, "pos": 0, "samples": ["R:A:L44:s0"]},
        {"id": B, "layer": 20, "pos": 0, "samples": ["R:B:L20:s0", "R:B:L20:s1"]},
        {"id": B, "layer": 44, "pos": 0, "samples": ["   "]},  # empty cell
        {"id": C, "layer": 20, "pos": 0, "samples": ["R:C:L20:s0"]},
        {"id": C, "layer": 44, "pos": 0, "samples": ["R:C:L44:s0"]},
        {"id": "not-in-bank", "layer": 20, "pos": 0, "samples": ["x"]},
    ]
    return write_jsonl(tmp_path / "r.jsonl", rows)


def _verdict(choice, basis="absent"):
    return {"choice": choice, "basis": basis, "evidence": "ev", "rationale": "why"}


def _responder(system, user):
    if user.startswith("Return"):
        return {"a": 1}
    assert system == prompts.UM_ATTRIBUTE_SYSTEM
    tag = tag_of(user)
    gold = {
        "A": GOLDEN[A]["gold_position"],
        "B": GOLDEN[B]["gold_position"],
        "C": GOLDEN[C]["gold_position"],
    }
    return {
        "A:L20:s0": _verdict(gold["A"], "verbatim_echo"),
        "A:L20:s2": _verdict(6),
        "A:L44:s0": _verdict(gold["A"], "inferred_characterization"),
        "B:L20:s0": _verdict(1 if gold["B"] != 1 else 2, "inferred_characterization"),
        "B:L20:s1": _verdict(99),
        "C:L20:s0": None,  # api failure
        "C:L44:s0": _verdict(gold["C"], "verbatim_echo"),
    }[tag]


def test_run_verdicts_and_numbers(tmp_path, fake_llm, mk_args):
    fake = fake_llm(_responder)
    args = mk_args(_scripted(tmp_path), items=[A, B, C])
    res = judge.run(args)
    assert res.family == "user_modeling" and res.metric == "pass_rate"
    assert res.n_items == 3 and res.value == pytest.approx(1 / 3)
    assert res.chance == pytest.approx(1 / 6) and "do not quote" in res.chance_label
    assert res.ci95 is not None and res.higher_is_better
    c = res.counts
    assert c["n_expected_cells"] == 6 and c["n_empty_cells"] == 1 and c["n_missing_cells"] == 0
    assert c["n_unjudged_cells"] == 1 and c["skipped_rows"] == 1 and c["spend_usd"] > 0
    by_key = {r["key"]: r for r in res.rows}
    assert set(by_key) == {
        f"{A}__L020__p0:s0", f"{A}__L020__p0:s2", f"{A}__L044__p0:s0",
        f"{B}__L020__p0:s0", f"{B}__L020__p0:s1", f"{C}__L044__p0:s0",
    }  # fmt: skip
    r = by_key[f"{A}__L020__p0:s2"]
    assert set(r) == {
        "key", "id", "subfamily", "attr_class", "layer", "pos", "sample_idx", "options",
        "gold_position", "choice", "basis", "evidence", "rationale", "pick", "pick_value",
        "choice_invalid",
    }  # fmt: skip
    assert r["sample_idx"] == 2 and r["pick"] == "cannot_tell" and r["choice_invalid"] is False
    assert r["pick_value"] == CANNOT_TELL and r["basis"] == "absent" and r["evidence"] == "ev"
    assert r["options"] == GOLDEN[A]["options"] and r["gold_position"] == GOLDEN[A]["gold_position"]
    assert r["subfamily"] == "selfdescribe" and r["attr_class"] == "Country"
    assert by_key[f"{A}__L020__p0:s0"]["pick"] == "gold"
    assert by_key[f"{A}__L020__p0:s0"]["basis"] == "verbatim_echo"
    assert by_key[f"{A}__L020__p0:s0"]["pick_value"] == "paraguay"
    assert by_key[f"{A}__L044__p0:s0"]["basis"] == "inferred_characterization"
    assert by_key[f"{B}__L020__p0:s0"]["pick"] == "distractor"
    b1 = by_key[f"{B}__L020__p0:s1"]
    assert b1["pick"] == "cannot_tell" and b1["choice_invalid"] is True and b1["choice"] == 99
    assert by_key[f"{C}__L044__p0:s0"]["subfamily"] == "synthsys"
    e = res.extras
    assert e["correct"] == pytest.approx(2 / 3) and e["inferred"] == pytest.approx(1 / 3)
    assert e["distractor"] == pytest.approx(1 / 3) and e["seed"] == 0 and e["n_api_failed"] == 1
    assert e["by_subfamily"] == {
        "selfdescribe": {"n_items": 2, "correct": 0.5, "inferred": 0.5, "distractor": 0.5},
        "synthsys": {"n_items": 1, "correct": 1.0, "inferred": 0.0, "distractor": 0.0},
    }
    assert e["picks"] == {"gold": 3, "distractor": 1, "cannot_tell": 1, "invalid": 1}
    assert res.complete is False and res.pinned_instrument is True
    assert res.config["prompt_version"] == "um-v2" and res.config["items"] == [A, B, C]
    judge_calls = [u for _s, u in fake.calls if not u.startswith("Return")]
    assert len(judge_calls) == 7  # k samples -> k calls; blank samples make no call
    assert all("<lens_output>\nR:" in u for u in judge_calls)
    assert (args.out / "cells.jsonl").exists()

    # resume: only the failed call is retried, then nothing at all
    fake2 = fake_llm(lambda s, u: {"a": 1} if u.startswith("Return") else _verdict(6))
    res2 = judge.run(args)
    assert [u for _s, u in fake2.calls if not u.startswith("Return")] == [
        u for u in judge_calls if "R:C:L20:s0" in u
    ]
    assert res2.counts["n_unjudged_cells"] == 0 and res2.value == pytest.approx(1 / 3)
    fake3 = fake_llm(lambda s, u: pytest.fail("no call expected"))
    judge.run(args)
    assert fake3.calls == []


def test_one_item_scope_reproduces_golden_options(tmp_path, fake_llm, mk_args):
    fake_llm(lambda s, u: {"a": 1} if u.startswith("Return") else _verdict(6))
    res = judge.run(mk_args(_scripted(tmp_path), items=[B]))
    assert res.n_items == 1 and len(res.rows) == 2
    for r in res.rows:
        assert r["options"] == GOLDEN[B]["options"]
        assert r["gold_position"] == GOLDEN[B]["gold_position"]
    assert res.extras["by_subfamily"]["synthsys"] == {
        "n_items": 0,
        "correct": None,
        "inferred": None,
        "distractor": None,
    }


def test_inferred_needs_gold_and_basis(tmp_path, fake_llm, mk_args):
    def echo_only(system, user):
        if user.startswith("Return"):
            return {"a": 1}
        return _verdict(GOLDEN[A]["gold_position"], "verbatim_echo")

    fake_llm(echo_only)
    res = judge.run(mk_args(_scripted(tmp_path), items=[A]))
    assert res.value == 0.0 and res.extras["correct"] == 1.0 and res.extras["inferred"] == 0.0


def test_tokens_kind_routes_through_summarizer(tmp_path, fake_llm, mk_args):
    rows = [
        {"id": A, "layer": 20, "pos": 0, "tokens": ["ĠParaguay", "Ġsopa"], "scores": [2.0, 1.0]},
        {"id": A, "layer": 44, "pos": 0, "tokens": ["Ġnoise"]},
    ]
    seen: list[str] = []

    def responder(system, user):
        if user.startswith("Return"):
            return {"a": 1}
        if user.startswith("TOKEN READOUTS:"):
            seen.append(user)
            if "noise" in user:
                return None  # summary failure -> cell unjudged
            return {"interpretation": "R:interp the user is Paraguayan"}
        assert "<lens_output>\nR:interp the user is Paraguayan\n</lens_output>" in user
        assert "ĠParaguay" not in user
        return _verdict(GOLDEN[A]["gold_position"], "inferred_characterization")

    fake_llm(responder)
    res = judge.run(mk_args(write_jsonl(tmp_path / "t.jsonl", rows), items=[A]))
    assert seen == ["TOKEN READOUTS:\nĠParaguay (2.00) | Ġsopa (1.00)", "TOKEN READOUTS:\nĠnoise"]
    assert res.value == 1.0 and len(res.rows) == 1 and res.rows[0]["sample_idx"] == 0
    assert res.rows[0]["key"] == f"{A}__L020__p0:s0"
    assert res.counts["n_unjudged_cells"] == 1 and res.extras["n_api_failed"] == 1


def test_dry_run_prints_and_makes_no_calls(tmp_path, mk_args, capsys, monkeypatch):
    monkeypatch.setattr("wsbench.llm._make_client", lambda r, k: pytest.fail("no client"))
    res = judge.run(mk_args(EXAMPLE, dry_run=True))
    out = capsys.readouterr().out
    assert prompts.UM_ATTRIBUTE_SYSTEM in out and "<lens_output>\nThe user is Danish." in out
    assert f"6. {CANNOT_TELL}" in out
    assert res.value is None and res.ci95 is None and res.rows == []
    assert res.config["dry_run"] is True and res.complete is False
    assert res.counts["spend_usd"] == 0.0 and res.n_items == 100
    assert res.counts["n_expected_cells"] == 8 and res.counts["n_empty_cells"] == 0
