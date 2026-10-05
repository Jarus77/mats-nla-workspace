import json
from pathlib import Path

import pytest
from conftest import tag_of, write_jsonl

from wsbench.evals.conjunctive_association import judge, prompts
from wsbench.evals.conjunctive_association.judge import (
    build_bags,
    build_blob,
    build_options,
    norm_token,
)
from wsbench.mc import CANNOT
from wsbench.mcjudge import load_bank
from wsbench.readouts import load_readouts

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/golden/conjunctive_association_options.json").read_text())
EXAMPLE = ROOT / "examples/readouts/conjunctive_association.jsonl"
QA, QB = "q01_a", "q01_b"


def test_bank_loads():
    bank = load_bank("conjunctive_association")
    assert len(bank) == 100
    for it in bank:
        assert {"id", "gold_label", "mc_options", "contrast_label", "stimulus"} <= set(it)


def test_options_match_source_golden():
    bank = load_bank("conjunctive_association")
    for it in bank:
        shown, gold_pos, contrast_pos = build_options(it)
        assert {"shown": shown, "gold_pos": gold_pos} == GOLDEN[it["id"]], it["id"]
        assert len(shown) == 11 and shown[-1] == CANNOT
        assert shown[contrast_pos - 1] == it["contrast_label"]
    with pytest.raises(ValueError):
        build_options({"id": "x", "gold_label": "g", "mc_options": ["a", "b"]})


def test_rendered_prompt_matches_golden():
    bank = load_bank("conjunctive_association")
    shown, _g, _c = build_options(bank[0])
    user = prompts.render_user("TOY READOUT", prompts.render_question(shown))
    assert user == (ROOT / "tests/golden/conjunctive_association_prompt.txt").read_text()
    assert prompts.PROMPT_VERSION == "comp-v1"


def test_blob_rule(tmp_path):
    assert norm_token("Ġsnake_case▁x") == "snake case x"
    cells, _rep = load_readouts(EXAMPLE)
    bags = build_bags(cells)
    assert set(bags) == {(QA, 20), (QA, 36), (QB, 20), (QB, 36)}
    assert bags[(QA, 20)] == (
        "grief over a child's accidental death | Nora is grieving; her young child died "
        "suddenly in a bicycle accident. | grief over a child's accidental death"
    )
    # file order, not pos order; prose loses underscores; non-printables stripped after the cap
    cells, _ = load_readouts(
        write_jsonl(
            tmp_path / "order.jsonl",
            [
                {"id": QA, "layer": 20, "pos": 5, "samples": ["b_b", ""]},
                {"id": QA, "layer": 20, "pos": 1, "samples": ["a\x00a"]},
            ],
        )
    )
    bags = build_bags(cells)
    assert bags[(QA, 20)] == "b b |  | a\x00a"
    blob = build_blob({20: bags[(QA, 20)], 36: "z"}, [20, 36], char_cap=1000)
    assert blob == "\n[L20] b b |  | aa\n[L36] z"
    assert build_blob({20: "abcdef"}, [20], char_cap=8) == "\n[L20] a"


def _scripted(tmp_path):
    rows = []
    for iid in (QA, QB, "q02_a"):
        for layer in (20, 36):
            if iid == "q02_a" and layer == 36:
                continue  # missing layer -> excluded
            for pos in (1, 2):
                # norm_token turns "_" into " ", so the tag carries the id with "-"
                tag = f"R:{iid.replace('_', '-')}:L{layer}:p{pos}"
                rows.append({"id": iid, "layer": layer, "pos": pos, "samples": [tag]})
    rows.append({"id": "q02_b", "layer": 20, "pos": 1, "samples": [""]})
    rows.append({"id": "q02_b", "layer": 36, "pos": 1, "samples": [""]})  # all-empty item
    return write_jsonl(tmp_path / "r.jsonl", rows)


def _responder(system, user):
    if user.startswith("Return"):
        return {"a": 1}
    assert system == prompts.SYSTEM
    iid = tag_of(user).split(":")[0].replace("-", "_")
    if iid == QA:
        return {"choice": GOLDEN[QA]["gold_pos"], "quote": "R:q01-a:L36:p2"}
    return None


def test_run_verdicts_and_numbers(tmp_path, fake_llm, mk_args):
    fake = fake_llm(_responder)
    args = mk_args(_scripted(tmp_path), items=[QA, QB, "q02_a", "q02_b", "q03_a"])
    res = judge.run(args)
    # q02_a excluded (missing L36); q03_a has no readouts (fail); q02_b all-empty (fail)
    assert res.n_items == 4 and res.value == 0.25 and res.chance == pytest.approx(1 / 11)
    c = res.counts
    assert c["n_expected_cells"] == 6 and c["n_empty_cells"] == 2 and c["n_unjudged_cells"] == 2
    rows = {r["id"]: r for r in res.rows}
    assert rows[QA]["pick"] == "gold" and rows[QA]["correct"] and rows[QA]["quote_ok"] is True
    assert set(rows[QA]) == {
        "id",
        "choice",
        "gold_pos",
        "contrast_pos",
        "pick",
        "correct",
        "quote",
        "quote_ok",
    }
    assert rows[QB]["pick"] == "api_fail" and rows[QB]["correct"] is False
    assert res.extras["breakdown"] == {
        "gold": 1, "contrast": 0, "distractor": 0, "cannot_tell": 0, "invalid": 0, "api_fail": 1
    }  # fmt: skip
    assert res.extras["n_items_excluded"] == 1 and res.extras["char_cap"] == 200000
    assert res.config["char_cap"] == 200000
    assert len([u for _s, u in fake.calls if not u.startswith("Return")]) == 2
    prompt = next(u for _s, u in fake.calls if "R:q01-a" in u)
    assert prompt.startswith(
        "READOUTS:\n\n[L20] R:q01-a:L20:p1 | R:q01-a:L20:p2\n"
        "[L36] R:q01-a:L36:p1 | R:q01-a:L36:p2\n\nWhich"
    )

    # resume: only the failed item is re-judged; the api_fail row disappears
    def contrast(s, u):
        if u.startswith("Return"):
            return {"a": 1}
        _shown, _g, cpos = build_options(
            next(it for it in load_bank("conjunctive_association") if it["id"] == QB)
        )
        return {"choice": cpos, "quote": "nope"}

    fake2 = fake_llm(contrast)
    res2 = judge.run(args)
    assert len([u for _s, u in fake2.calls if not u.startswith("Return")]) == 1
    rows = {r["id"]: r for r in res2.rows}
    assert rows[QB]["pick"] == "contrast" and rows[QB]["quote_ok"] is False
    assert res2.counts["n_unjudged_cells"] == 0 and res2.extras["breakdown"]["contrast"] == 1
    fake3 = fake_llm(lambda s, u: pytest.fail("no call expected"))
    judge.run(args)
    assert fake3.calls == []


def test_char_cap_option_and_pick_labels(tmp_path, fake_llm, mk_args):
    picks = iter([{"choice": 11, "quote": ""}, {"choice": 0, "quote": ""}])

    def responder(s, u):
        if u.startswith("Return"):
            return {"a": 1}
        assert len(u.split("\n\nWhich")[0]) <= len("READOUTS:\n") + 12
        return next(picks)

    fake_llm(responder)
    res = judge.run(mk_args(_scripted(tmp_path), items=[QA, QB], extra={"char_cap": "12"}))
    assert res.extras["char_cap"] == 12 and res.config["char_cap"] == 12
    assert sorted(r["pick"] for r in res.rows) == ["cannot_tell", "invalid"]
    assert res.value == 0.0


def test_tokens_kind_routes_through_summarizer(tmp_path, fake_llm, mk_args):
    rows = [
        {"id": QA, "layer": 20, "pos": 1, "tokens": ["Ġgrief", "Ġchild_"]},
        {"id": QA, "layer": 20, "pos": 2, "tokens": ["Ġaccident"]},
        {"id": QA, "layer": 36, "pos": 1, "tokens": ["Ġmourning"]},
    ]
    seen = []

    def responder(system, user):
        if user.startswith("Return"):
            return {"a": 1}
        if user.startswith("TOKEN READOUTS:"):
            seen.append(user)
            return {
                "interpretation": "grief over her child's death" if "grief" in user else "mourning"
            }
        assert "READOUTS:\n\n[L20] grief over her child's death\n[L36] mourning\n\n" in user
        return {"choice": GOLDEN[QA]["gold_pos"], "quote": "grief"}

    fake_llm(responder)
    res = judge.run(mk_args(write_jsonl(tmp_path / "t.jsonl", rows), items=[QA]))
    assert seen == ["TOKEN READOUTS:\ngrief | child | accident", "TOKEN READOUTS:\nmourning"]
    assert res.value == 1.0


def test_dry_run_prints_and_makes_no_calls(tmp_path, mk_args, capsys, monkeypatch):
    monkeypatch.setattr("wsbench.llm._make_client", lambda r, k: pytest.fail("no client"))
    res = judge.run(mk_args(EXAMPLE, dry_run=True))
    out = capsys.readouterr().out
    assert prompts.SYSTEM in out and "Which of the following does the readout state?" in out
    assert res.value is None and res.rows == [] and res.n_items == 100


def test_char_cap_validation(tmp_path):
    import pytest

    from wsbench import registry
    from wsbench.evals.conjunctive_association import judge as cj
    from wsbench.judge_config import JudgeConfig, resolve
    from wsbench.registry import JudgeArgs

    def args(v: str) -> JudgeArgs:
        return JudgeArgs(
            readouts=registry.REPO_ROOT / "examples/readouts/conjunctive_association.jsonl",
            out=tmp_path,
            judge=resolve(JudgeConfig()),
            layers=None,
            items=None,
            limit=0,
            allow_missing=False,
            concurrency=1,
            rpm=1e9,
            dry_run=True,
            extra={"char_cap": v},
        )

    for bad in ("abc", "0", "-5"):
        with pytest.raises(SystemExit):
            cj.run(args(bad))
