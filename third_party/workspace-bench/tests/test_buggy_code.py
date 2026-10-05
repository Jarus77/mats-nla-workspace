"""buggy_code: the bank, the blind inference, the informed 1-10 grade, the mean-score headline
with clean twins inside it, a scripted two-stage run, dry run."""

import json
from pathlib import Path

import pytest

from wsbench import registry
from wsbench.banks import load_bank
from wsbench.cli import main
from wsbench.evals.buggy_code import judge
from wsbench.evals.buggy_code.prompts import (
    GRADE_SYSTEM,
    INFER_SYSTEM,
    PROMPT_VERSION,
    SILENT_SCORE,
    render_grade,
    render_infer,
)
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.registry import JudgeArgs
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "evals/buggy_code/items.json"
EXAMPLE = REPO / "examples/readouts/buggy_code.jsonl"


@pytest.fixture(autouse=True)
def _registered():
    from wsbench.evals.buggy_code import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)


def _args(tmp_path, ids, **over):
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
    return JudgeArgs(**{**base, **over})


def _example_ids():
    _h, items = load_bank(BANK)
    by_id = {it["id"]: it for it in items}
    ids = sorted({json.loads(x)["id"] for x in EXAMPLE.read_text().splitlines() if x.strip()})
    buggy = [i for i in ids if by_id[i]["src"] == "buggy"]
    clean = [i for i in ids if by_id[i]["src"] == "clean"]
    return by_id, ids, buggy, clean


def test_bank():
    header, items = load_bank(BANK)
    assert header["family"] == "buggy_code" and len(items) == 49
    assert header["read_cells"] == {
        "python": {"anchor": "eof", "layer": 60},
        "nonpython": {"anchor": "eof", "layer": 56},
    }
    srcs = [it["src"] for it in items]
    assert srcs.count("buggy") == 25 and srcs.count("clean") == 24
    for it in items:
        assert it["lang_group"] in ("python", "nonpython") and it["code"] and it["verified"]
        if it["src"] == "buggy":
            assert it["cause"] and it.get("consequence_class") in (
                "value",
                "categorical",
                "crash",
                None,
            )


def test_stage_a_is_blind_and_stage_b_holds_the_truth():
    _h, items = load_bank(BANK)
    it = next(i for i in items if i["src"] == "buggy")
    blind = render_infer(["# Expected: -53.52", "assert total < 0"])
    assert it["code"] not in blind and it["verified"] not in blind and "src=" not in blind
    assert "readout samples (k=2):\n[0] # Expected: -53.52\n[1] assert total < 0" in blind
    informed = render_grade(it, "bug", "the total goes negative")
    assert it["code"] in informed and it["verified"] in informed
    assert "cause (the bug):" in informed and f"bug line: {it['bug_line']}" in informed
    assert "(claim=bug): the total goes negative" in informed
    clean = next(i for i in items if i["src"] == "clean" and not i.get("why_correct"))
    g = render_grade(clean, "none", "")
    assert "cause (" not in g and "bug line" not in g and "(nothing claimed" in g
    told = next(i for i in items if i["src"] == "clean" and i.get("why_correct"))
    assert f"cause (the twin context): {told['why_correct']}" in render_grade(told, "bug", "x")


def test_inference_and_grade_clamp():
    ok = judge.inference({"claim": "bug", "inferred": "negative", "quote": "(50 - 70) = -20"})
    assert ok["claim"] == "bug" and ok["inferred"] == "negative" and ok["judged"]
    # the quote is evidence, not a gate: an unmatched quote leaves the claim standing
    kept = judge.inference({"claim": "bug", "inferred": "negative", "quote": "never said"})
    assert kept["claim"] == "bug" and kept["inferred"] == "negative"
    corr = judge.inference({"claim": "correct", "inferred": "runs", "quote": "looks fine"})
    assert corr["claim"] == "correct"
    assert judge.inference({"claim": "weird", "inferred": "", "quote": ""})["claim"] == "none"
    assert not judge.inference(None)["judged"]
    assert judge.grade({"score": 7, "why": ""}) == 7
    assert judge.grade({"score": 14, "why": ""}) == 10 and judge.grade({"score": 0, "why": ""}) == 1
    assert judge.grade({"score": "x", "why": ""}) is None and judge.grade(None) is None


def test_scripted_two_stage_run(tmp_path, monkeypatch):
    _by, ids, buggy, clean = _example_ids()
    assert len(buggy) == 2 and len(clean) == 1

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            i, stage = c.meta["item"], c.meta["stage"]
            if stage == "infer":
                readout = c.user.split("[0] ", 1)[1].split("\n")[0]
                if i == buggy[0]:  # a verified bug claim
                    out[c.key] = {
                        "claim": "bug",
                        "inferred": "total goes negative",
                        "quote": readout[:20],
                    }
                elif i == buggy[1]:  # the readouts say nothing about behaviour
                    out[c.key] = {"claim": "none", "inferred": "", "quote": ""}
                else:  # the clean twin: the readout says it works
                    out[c.key] = {
                        "claim": "correct",
                        "inferred": "prints the charge",
                        "quote": readout[:12],
                    }
            else:
                assert "code:" in c.user and "verified executed truth" in c.user
                out[c.key] = {"score": 8 if c.meta["src"] == "buggy" else 10, "why": "close"}
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    r = judge.run(_args(tmp_path, ids))
    rows = {row["id"]: row for row in r.rows}
    assert rows[buggy[0]]["claim"] == "bug" and rows[buggy[0]]["score"] == 8
    assert rows[buggy[0]]["layer"] == 60
    assert (
        rows[buggy[1]]["claim"] == "none" and rows[buggy[1]]["score"] == SILENT_SCORE["buggy"] == 1
    )
    assert rows[clean[0]]["claim"] == "correct" and rows[clean[0]]["score"] == 10
    assert r.metric == "score"
    assert r.value == pytest.approx((8 + 1 + 10) / 3)
    assert r.extras["n_calls"] == 3 + 2  # three blind calls, two graded (the silent one skips)
    assert r.extras["score_buggy"] == pytest.approx(4.5)
    assert r.extras["score_clean"] == 10.0 and r.extras["false_alarm_rate_clean"] == 0.0
    assert r.extras["claims"]["buggy"] == {"bug": 0.5, "correct": 0.0, "none": 0.5}
    assert r.chance == pytest.approx((1 + 1 + 5) / 3) == pytest.approx(r.extras["silent_floor"])
    assert r.counts["n_expected_cells"] == 3 and r.counts["n_missing_cells"] == 0
    assert r.ci95 is not None and r.extras["score_hist"]["buggy"]["8"] == 1


def test_silent_and_empty_cells_take_the_fixed_scores(tmp_path, monkeypatch):
    _by, ids, buggy, clean = _example_ids()
    rows = [json.loads(x) for x in EXAMPLE.read_text().splitlines() if x.strip()]
    for r in rows:
        if r["id"] == buggy[1]:
            r["samples"] = ["", "  "]  # an empty cell
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    calls_seen = []

    def fake_run_calls(calls, **kw):
        calls_seen.extend(calls)
        return {c.key: {"claim": "none", "inferred": "", "quote": ""} for c in calls}

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    r = judge.run(_args(tmp_path, ids, readouts=path))
    got = {row["id"]: row for row in r.rows}
    assert got[buggy[1]]["empty"] and got[buggy[1]]["score"] == 1
    assert got[buggy[0]]["score"] == 1 and got[clean[0]]["score"] == 5
    assert all(c.meta["stage"] == "infer" for c in calls_seen)  # nothing to grade
    assert r.value == pytest.approx(r.chance)  # a silent lens sits on the silent floor
    assert r.counts["n_empty_cells"] == 1


def test_unjudged_stage_leaves_the_item_out(tmp_path, monkeypatch):
    _by, ids, buggy, _clean = _example_ids()

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            if c.meta["stage"] == "infer":
                out[c.key] = {
                    "claim": "bug",
                    "inferred": "x",
                    "quote": c.user.split("[0] ", 1)[1][:8],
                }
            else:
                out[c.key] = None if c.meta["item"] == buggy[0] else {"score": 3, "why": ""}
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    r = judge.run(_args(tmp_path, ids))
    got = {row["id"]: row for row in r.rows}
    assert not got[buggy[0]]["judged"] and got[buggy[0]]["score"] is None
    assert r.counts["n_unjudged_cells"] == 1 and r.n_items == 3
    assert r.value == pytest.approx(3.0)
    assert not r.complete


def test_missing_read_layer_is_fatal_and_two_layers_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "run_calls", lambda calls, **kw: {})
    _by, ids, _b, _c = _example_ids()
    with pytest.raises(SystemExit) as e:  # the toy has no layer-56 rows
        judge.run(_args(tmp_path, ids, layers=[56]))
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        judge.run(_args(tmp_path, ids, layers=[56, 60]))
    r = judge.run(_args(tmp_path, ids, layers=[56], allow_missing=True))
    assert r.counts["n_missing_cells"] == 3 and r.value is None


def test_registered_readme_and_dry_run(tmp_path, capsys, monkeypatch):
    from wsbench.evals.buggy_code import SPEC

    assert SPEC.group == "computational" and SPEC.metric == "score"
    readme = (REPO / "evals/buggy_code/README.md").read_text(encoding="utf-8")
    assert INFER_SYSTEM in readme and GRADE_SYSTEM in readme
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    argv = ["judge", f"family={SPEC.name}", f"readouts={EXAMPLE}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] == 46
    assert "readout samples (k=" in capsys.readouterr().out
