import json
from pathlib import Path

import pytest
from conftest import tag_of, write_jsonl

from wsbench.evals.relational_multihop import judge, prompts
from wsbench.evals.relational_multihop.judge import build_mc, pools, select_cells
from wsbench.mc import CANNOT, listing
from wsbench.mcjudge import load_bank
from wsbench.readouts import load_readouts

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/golden/relational_multihop_options.json").read_text())
EXAMPLE = ROOT / "examples/readouts/relational_multihop.jsonl"
TOKENS = ROOT / "examples/readouts/relational_multihop.tokens.jsonl"
A, B = "rel-s00-a", "rel-s00-b"


def test_bank_loads():
    bank = load_bank("relational_multihop")
    assert len(bank) == 100
    for it in bank:
        assert {"id", "scene", "hop1", "hop2", "stimulus"} <= set(it)
    assert len({it["scene"] for it in bank}) == 50


def test_options_match_source_golden():
    bank = load_bank("relational_multihop")
    prof, kin = pools(bank)
    assert prof == sorted(prof) and kin == sorted(kin) and not set(prof) & set(kin)
    assert set(prompts.KIN_EXTRA) <= set(kin) and set(prompts.PROF_EXTRA) <= set(prof)
    for it in bank:
        shown, x, y = build_mc(it["id"], it["hop1"], it["hop2"], prof, kin)
        assert {"shown": shown, "x_gold_pos": x, "y_gold_pos": y} == GOLDEN[it["id"]], it["id"]
        assert len(shown) == 11 and shown[-1] == CANNOT
        assert shown[x - 1] == it["hop1"] and shown[y - 1] == it["hop2"]


def test_rendered_prompt_matches_golden():
    bank = load_bank("relational_multihop")
    shown, _x, _y = build_mc(bank[0]["id"], bank[0]["hop1"], bank[0]["hop2"], *pools(bank))
    user = prompts.render_question("TOY READOUT", "X", listing(shown))
    assert user == (ROOT / "tests/golden/relational_multihop_prompt.txt").read_text()
    assert "Y (the INNER/second relation word)" in prompts.render_question("r", "Y", "l")
    assert prompts.render_bundle("'s", 21, "a | b") == '[position "\'s"] a | b'
    assert prompts.render_bundle(None, 21, "a") == "[position p21] a"


def test_blank_cell_rule_and_token_check(tmp_path):
    cells, _ = load_readouts(EXAMPLE)
    sel = select_cells(cells)
    assert [(c.id, c.layer, c.pos, c.token) for c in sel] == [
        (A, 20, 21, "'s"),
        (A, 36, 21, "'s"),
        (B, 20, 21, "'s"),
        (B, 36, 21, "'s"),
    ]
    assert judge.bundle_text(sel[0]).startswith('[position "\'s"] Riley is the sibling')
    bad = write_jsonl(
        tmp_path / "bad.jsonl",
        [{"id": A, "layer": 20, "pos": 5, "samples": ["x"], "token": "Riley"}],
    )
    cells, _ = load_readouts(bad)
    with pytest.raises(ValueError, match=A):
        select_cells(cells)
    # no token at all: accepted, rendered with the position
    cells, _ = load_readouts(
        write_jsonl(tmp_path / "nt.jsonl", [{"id": A, "layer": 20, "pos": 5, "samples": ["x"]}])
    )
    assert judge.bundle_text(select_cells(cells)[0]) == "[position p5] x"


def _scripted(tmp_path):
    rows = []
    for iid in (A, B):
        for layer in (20, 36):
            rows.append({"id": iid, "layer": layer, "pos": 1, "samples": ["early"]})
            rows.append(
                {
                    "id": iid,
                    "layer": layer,
                    "pos": 9,
                    "samples": [f"R:{iid}:L{layer}"],
                    "token": "'s",
                }
            )
    rows.append({"id": A, "layer": 44, "pos": 9, "samples": [""], "token": "'s"})  # empty
    return write_jsonl(tmp_path / "r.jsonl", rows)


def _responder(system, user):
    if user.startswith("Return"):
        return {"a": 1}
    assert system == prompts.JUDGE_SYSTEM
    tag = tag_of(user)
    iid = tag.split(":")[0]
    which = "x" if "What is X" in user else "y"
    g = GOLDEN[iid][f"{which}_gold_pos"]
    if tag == f"{A}:L20":
        return {"choice": g, "quote": "q"}
    if tag == f"{A}:L36":
        return {"choice": g if which == "x" else 11, "quote": ""}
    if tag == f"{B}:L20":
        return None if which == "y" else {"choice": g, "quote": ""}
    return {"choice": GOLDEN[iid]["y_gold_pos" if which == "x" else "x_gold_pos"], "quote": ""}


def test_run_verdicts_and_numbers(tmp_path, fake_llm, mk_args):
    fake = fake_llm(_responder)
    args = mk_args(_scripted(tmp_path), items=[A, B])
    res = judge.run(args)
    assert res.n_items == 2 and res.value == 0.5 and res.ci95 is not None
    c = res.counts
    assert c["n_expected_cells"] == 5 and c["n_empty_cells"] == 1 and c["n_unjudged_cells"] == 1
    by_key = {r["key"]: r for r in res.rows}
    assert set(by_key) == {f"{A}__L020__p9", f"{A}__L036__p9", f"{B}__L036__p9"}
    r = by_key[f"{A}__L020__p9"]
    assert r["pass"] is True and r["x_ok"] and r["y_ok"] and r["x_quote"] == "q"
    assert set(r) == {
        "key", "id", "layer", "pos", "x_choice", "x_gold_pos", "x_ok", "x_quote",
        "y_choice", "y_gold_pos", "y_ok", "y_quote", "pass",
    }  # fmt: skip
    assert by_key[f"{A}__L036__p9"]["pass"] is False and by_key[f"{A}__L036__p9"]["x_ok"]
    flip = by_key[f"{B}__L036__p9"]
    assert flip["pass"] is False and not flip["x_ok"] and not flip["y_ok"]  # direction flip
    e = res.extras
    assert e["per_layer_pass"] == {"L20": {"pass": 1, "judged": 1}, "L36": {"pass": 0, "judged": 2}}
    assert e["pair_consistency"] == {"n": 0, "of": 50}
    assert e["n_layers"] == 3 and e["any_of_layers_floor"] == pytest.approx(1 - (120 / 121) ** 3)
    assert e["n_cells_judged"] == 3 and e["n_cells_api_failed"] == 1
    assert e["n_cells_interp_missing"] == 0
    assert "1/121" in res.chance_label and res.chance is None
    assert len([u for _s, u in fake.calls if not u.startswith("Return")]) == 8

    # resume: only the dropped pair's failed call is retried
    fake2 = fake_llm(
        lambda s, u: {"a": 1} if u.startswith("Return") else {"choice": 11, "quote": ""}
    )
    res2 = judge.run(args)
    assert len([u for _s, u in fake2.calls if not u.startswith("Return")]) == 1
    assert res2.counts["n_unjudged_cells"] == 0 and res2.extras["n_cells_judged"] == 4
    fake3 = fake_llm(lambda s, u: pytest.fail("no call expected"))
    judge.run(args)
    assert fake3.calls == []


def test_tokens_kind_routes_through_summarizer(tmp_path, fake_llm, mk_args):
    seen: list[str] = []

    def responder(system, user):
        if user.startswith("Return"):
            return {"a": 1}
        if user.startswith("TOKEN READOUTS:"):
            seen.append(user)
            return {"interpretation": "the landlord's sibling"} if "L020" not in user else None
        assert "the landlord's sibling" in user and "Ġlandlord" not in user
        which = "x" if "What is X" in user else "y"
        return {"choice": GOLDEN[A][f"{which}_gold_pos"], "quote": "landlord"}

    fake_llm(responder)
    res = judge.run(mk_args(TOKENS, items=[A]))
    assert seen[0] == (
        'TOKEN READOUTS:\n[position "\'s"] Ġlandlord (11.20) | Ġsibling (10.40) | '
        "Ġbrother (6.10) | Ġtenant (5.90)"
    )
    assert res.value == 1.0 and res.extras["n_cells_interp_missing"] == 0
    assert len(res.rows) == 2


def test_summary_failure_skips_cell(tmp_path, fake_llm, mk_args):
    def responder(system, user):
        if user.startswith("Return"):
            return {"a": 1}
        if user.startswith("TOKEN READOUTS:"):
            return None
        pytest.fail("no judge call without a summary")

    fake_llm(responder)
    res = judge.run(mk_args(TOKENS, items=[A]))
    assert res.value == 0.0 and res.rows == []
    assert res.extras["n_cells_interp_missing"] == 2 and res.counts["n_unjudged_cells"] == 2


def test_dry_run_prints_and_makes_no_calls(tmp_path, mk_args, capsys, monkeypatch):
    monkeypatch.setattr("wsbench.llm._make_client", lambda r, k: pytest.fail("no client"))
    res = judge.run(mk_args(EXAMPLE, dry_run=True))
    out = capsys.readouterr().out
    assert prompts.JUDGE_SYSTEM in out and "What is X (the OUTER/first relation word)?" in out
    assert res.value is None and res.rows == [] and res.config["dry_run"] is True
    capsys.readouterr()
    res = judge.run(mk_args(TOKENS, dry_run=True))
    out = capsys.readouterr().out
    assert "TOKEN READOUTS:" in out and "What is X" not in out  # summarizer prompt only
