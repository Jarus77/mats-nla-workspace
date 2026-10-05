import json
from pathlib import Path

import pytest
from conftest import tag_of, write_jsonl

from wsbench.evals.role_bound_association import judge, prompts
from wsbench.evals.role_bound_association.judge import build_block
from wsbench.mc import CANNOT
from wsbench.mcjudge import load_bank
from wsbench.readouts import load_readouts

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/golden/role_bound_association_options.json").read_text())
EXAMPLE = ROOT / "examples/readouts/role_bound_association.jsonl"
AB, BA = "oa-police-thief-chase-ab", "oa-police-thief-chase-ba"


def test_bank_loads():
    bank = load_bank("role_bound_association")
    assert len(bank) == 100
    for it in bank:
        assert {
            "id",
            "pair_id",
            "direction",
            "stereotypicality",
            "role_a",
            "role_b",
            "action",
            "names",
        } <= set(it)
    assert {it["stereotypicality"] for it in bank} == {"congruent", "incongruent", "neutral"}


def test_blocks_match_source_golden():
    bank = load_bank("role_bound_association")
    for it in bank:
        block, golds = build_block(it, bank)
        assert {"block": block, "golds": list(golds)} == GOLDEN[it["id"]], it["id"]
        assert block.count(CANNOT) == 3 and "{person A} -> {action} -> {person B}" in block


def test_rendered_prompt_matches_golden():
    bank = load_bank("role_bound_association")
    block, _ = build_block(bank[0], bank)
    user = prompts.render_user("TOY READOUT", block)
    assert user == (ROOT / "tests/golden/role_bound_association_prompt.txt").read_text()
    assert prompts.PROMPT_VERSION == "oa-v1"
    assert prompts.SCHEMA["schema"]["required"] == [
        "q1_choice",
        "q2_choice",
        "q3_choice",
        "evidence",
    ]


def test_every_row_is_a_cell():
    cells, _ = load_readouts(EXAMPLE)
    assert len(cells) == 12  # 2 items x 2 layers x 3 positions


def _scripted(tmp_path):
    rows = []
    for iid in (AB, BA):
        for layer in (20, 36):
            for pos in (1, 2):
                rows.append(
                    {"id": iid, "layer": layer, "pos": pos, "samples": [f"R:{iid}:L{layer}:p{pos}"]}
                )
    rows.append({"id": AB, "layer": 20, "pos": 3, "samples": []})  # empty
    return write_jsonl(tmp_path / "r.jsonl", rows)


def _responder(system, user):
    if user.startswith("Return"):
        return {"a": 1}
    assert system == prompts.SYSTEM
    tag = tag_of(user)
    g = GOLDEN[tag.split(":")[0]]["golds"]
    if tag == f"{AB}:L20:p1":
        return {"q1_choice": g[0], "q2_choice": g[1], "q3_choice": g[2], "evidence": "chased"}
    if tag == f"{AB}:L20:p2":
        return {"q1_choice": g[2], "q2_choice": g[1], "q3_choice": g[0], "evidence": "flip"}
    if tag == f"{AB}:L36:p1":
        return None
    return {"q1_choice": 6, "q2_choice": 6, "q3_choice": 6, "evidence": ""}


def test_run_verdicts_and_numbers(tmp_path, fake_llm, mk_args):
    fake = fake_llm(_responder)
    args = mk_args(_scripted(tmp_path), items=[AB, BA])
    res = judge.run(args)
    assert res.n_items == 2 and res.value == 0.5
    c = res.counts
    assert c["n_expected_cells"] == 9 and c["n_empty_cells"] == 1 and c["n_unjudged_cells"] == 1
    by_key = {r["key"]: r for r in res.rows}
    assert len(by_key) == 7
    r = by_key[f"{AB}__L020__p1"]
    assert r["correct"] == [True, True, True] and r["pass"] is True and r["evidence"] == "chased"
    assert set(r) == {"key", "id", "layer", "pos", "correct", "pass", "evidence"}
    assert by_key[f"{AB}__L020__p2"]["correct"] == [False, True, False]
    assert by_key[f"{BA}__L020__p1"]["pass"] is False
    e = res.extras
    assert e["sites_per_item"] == {"min": 3, "median": 3.5, "max": 4}
    assert e["any_of_grid_floor"] == pytest.approx(
        ((1 - (1 - 1 / 216) ** 3) + (1 - (1 - 1 / 216) ** 4)) / 2, abs=1e-4
    )
    assert e["n_api_failed"] == 1
    assert e["by_stereotypicality"] == {
        "congruent": {"n": 1, "pass": 1},
        "incongruent": {"n": 1, "pass": 0},
        "neutral": {"n": 0, "pass": 0},
    }
    assert res.chance is None and "(1/6)^3" in res.chance_label
    assert len([u for _s, u in fake.calls if not u.startswith("Return")]) == 8

    cannot = {"q1_choice": 6, "q2_choice": 6, "q3_choice": 6, "evidence": ""}
    fake2 = fake_llm(lambda s, u: {"a": 1} if u.startswith("Return") else cannot)
    judge.run(args)
    assert len([u for _s, u in fake2.calls if not u.startswith("Return")]) == 1
    fake3 = fake_llm(lambda s, u: pytest.fail("no call expected"))
    res3 = judge.run(args)
    assert fake3.calls == [] and res3.counts["n_unjudged_cells"] == 0


def test_tokens_kind_routes_through_summarizer(tmp_path, fake_llm, mk_args):
    rows = [{"id": AB, "layer": 20, "pos": 1, "tokens": ["ĠMarcus", "Ġchased"]}]
    seen = []

    def responder(system, user):
        if user.startswith("Return"):
            return {"a": 1}
        if user.startswith("TOKEN READOUTS:"):
            seen.append(user)
            return {"interpretation": "Marcus chased Dylan"}
        assert "READOUT:\nMarcus chased Dylan\n\n" in user
        g = GOLDEN[AB]["golds"]
        return {"q1_choice": g[0], "q2_choice": g[1], "q3_choice": g[2], "evidence": "e"}

    fake_llm(responder)
    res = judge.run(mk_args(write_jsonl(tmp_path / "t.jsonl", rows), items=[AB]))
    assert seen == ["TOKEN READOUTS:\nĠMarcus | Ġchased"] and res.value == 1.0


def test_dry_run_prints_and_makes_no_calls(tmp_path, mk_args, capsys, monkeypatch):
    monkeypatch.setattr("wsbench.llm._make_client", lambda r, k: pytest.fail("no client"))
    res = judge.run(mk_args(EXAMPLE, dry_run=True))
    out = capsys.readouterr().out
    assert prompts.SYSTEM in out and "Q1. Who performs the action (the agent)?" in out
    assert res.value is None and res.rows == [] and res.n_items == 100
