"""The capability runner: questions come from the banks (pinned by a golden), the greedy leg and
the sampled rate both gate an item, and a failed call is not a free pass."""

import json
from pathlib import Path

from wsbench.capable import questions as capq
from wsbench.capable import run as caprun
from wsbench.cli import main
from wsbench.judge_config import JudgeConfig, resolve

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "capable_questions.json"


def built() -> dict[str, dict]:
    out = {}
    for family in sorted(capq.BUILDERS):
        qs = capq.build(family)
        out[family] = {
            "n": len(qs),
            "first_id": qs[0].id,
            "first_ask": qs[0].ask,
            "first_golds": qs[0].golds,
        }
    return out


def test_every_family_builds_answerable_questions():
    for family in sorted(capq.BUILDERS):
        qs = capq.build(family)
        assert qs, family
        assert len({q.id for q in qs}) == len(qs), family
        for q in qs:
            assert q.ask.strip() and all(g.strip() for g in q.golds), (family, q.id)


def test_questions_match_the_golden():
    """The question text IS the instrument here: a silent edit changes every number."""
    assert built() == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_task_asked_is_the_task_the_bank_gates():
    # a typo item's prompt ENDS in the misspelling: continuing it never yields the correction
    typo = capq.build("typo")[0]
    assert typo.ask.startswith(capq.CORRECT_SPELLING) and capq.CONTINUE not in typo.ask
    # an association item's prompt is a finished sentence that never names the concept
    assert capq.build("association")[0].ask.startswith(capq.NAME_REFERENT)
    # the mt bridge families ask the bank's own bridge question beside the surface answer
    bank = {it["id"]: it for it in capq.bank_items("multihop_mt")}
    bridges = [q for q in capq.build("multihop_mt") if q.meta.get("leg") == "bridge"]
    assert bridges
    for q in bridges:
        item = bank[q.id.split(":")[0]]
        assert q.ask in [b["question"] for b in item["bridges"]]
        assert q.golds and set(q.golds) <= {str(a) for b in item["bridges"] for a in b["answers"]}
    # role-bound and conjunctive send the bank's own asked_question, not a hand-written one
    for family in ("role_bound_association", "conjunctive_association"):
        items = {it["id"]: it for it in capq.bank_items(family)}
        for q in capq.build(family):
            assert items[q.id]["asked_question"] in q.ask


def test_items_with_no_model_independent_answer_are_left_out():
    bank = capq.bank_items("basic_readout")
    implicit = {it["id"] for it in bank if it["subfamily"] == "implicit"}
    asked = {q.id for q in capq.build("basic_readout")}
    assert implicit and not (implicit & asked)
    assert len(asked) == len(bank) - len(implicit)
    assert "implicit" in capq.PARTIAL["basic_readout"]


def test_reference_rates_come_from_the_bank():
    typo = {it["id"]: it for it in capq.bank_items("typo")}
    for q in capq.build("typo"):
        assert q.reference == typo[q.id]["consistency"]
    chain = {it["id"]: it for it in capq.bank_items("chain_intermediates")}
    for q in capq.build("chain_intermediates"):
        gate = chain[q.id]["gate"]["last"]
        assert q.reference == gate["correct"] / gate["n"]


def test_the_two_banks_gated_at_ten_of_ten_keep_that_threshold():
    assert capq.threshold_for("chain_intermediates", 0.8) == 1.0
    assert capq.threshold_for("brew_intermediates", 0.8) == 1.0
    assert capq.threshold_for("poetry", 0.8) == 0.8


def test_buggy_gold_is_readable_beside_the_harness_string():
    assert capq.readable_outcome("exec: exit 0; stdout = '-53.5'") == "the program prints: -53.5"
    assert "exits 1" in capq.readable_outcome("exec: exit 1; ZeroDivisionError")
    assert capq.readable_outcome("plain text") == "plain text"
    q = capq.build("buggy_code")[0]
    assert q.golds[0].startswith("the program") and q.golds[1].startswith("exec:")


def test_no_question_families_say_what_their_gate_is():
    from wsbench import registry

    registry.load_all()
    assert set(registry.FAMILIES) <= set(capq.BUILDERS) | set(capq.NO_QUESTION)
    for family, why in capq.NO_QUESTION.items():
        assert why.strip()
        try:
            capq.build(family)
        except KeyError as e:
            assert why in str(e)
        else:
            raise AssertionError(f"{family} should have no capability question")


def _judge():
    return resolve(JudgeConfig(prompt_version=caprun.PROMPT_VERSION))


def _answers(fn):
    """A fake ``run_calls`` that answers with ``fn(item, draw)`` and grades "right" as correct."""

    def fake(calls, **kw):
        out = {}
        for c in calls:
            if c.system == caprun.ANSWER_SYSTEM:
                out[c.key] = fn(c.meta["item"], c.meta["draw"])
            else:
                out[c.key] = {"correct": c.meta["answer"] == "right", "why": ""}
        return out

    return fake


def test_scripted_run_needs_the_greedy_answer_and_the_sampled_rate(tmp_path, monkeypatch):
    def answer(item, draw):
        if item == "greedy-miss" and draw == caprun.GREEDY:
            return {"answer": "wrong"}
        if item == "sampled-miss" and draw == 0:
            return {"answer": "wrong"}
        return {"answer": "right"}

    monkeypatch.setattr(caprun, "run_calls", _answers(answer))
    monkeypatch.setattr(
        caprun,
        "build",
        lambda family: [
            capq.Question("clean", "ask", ["gold"], 1.0),
            capq.Question("greedy-miss", "ask", ["gold"], 0.9),
            capq.Question("sampled-miss", "ask", ["gold"], None),
        ],
    )
    r = caprun.run_family("poetry", model="m", judge=_judge(), out=tmp_path, draws=4)
    rows = {row["id"]: row for row in r["rows"]}
    assert rows["clean"]["rate"] == 1.0 and rows["clean"]["passes_gate"]
    # greedy wrong, every sample right: the bank's gate required both
    assert rows["greedy-miss"]["rate"] == 1.0 and not rows["greedy-miss"]["passes_gate"]
    assert rows["greedy-miss"]["greedy_correct"] is False
    # 3 of 4 sampled right is below the 0.8 threshold
    assert rows["sampled-miss"]["rate"] == 0.75 and not rows["sampled-miss"]["passes_gate"]
    a = r["aggregate"]
    assert a["gate_rate"] == 1 / 3 and a["greedy_rate"] == 2 / 3
    assert a["reference_rate"] == 0.95  # mean of the two items whose bank recorded a rate
    assert "bank=0.950" in caprun.report_line(r)


def test_greedy_draw_is_a_separate_temperature_batch(tmp_path, monkeypatch):
    temps = []

    def fake(calls, **kw):
        if calls and calls[0].system == caprun.ANSWER_SYSTEM:
            temps.append((kw["temperature"], sorted(c.meta["draw"] for c in calls)))
        return _answers(lambda i, d: {"answer": "right"})(calls, **kw)

    monkeypatch.setattr(caprun, "run_calls", fake)
    monkeypatch.setattr(caprun, "build", lambda f: [capq.Question("i", "ask", ["g"])])
    caprun.run_family("poetry", model="m", judge=_judge(), out=tmp_path, draws=2)
    assert temps == [(0.0, [caprun.GREEDY]), (0.7, [0, 1])]


def test_an_empty_answer_is_wrong_and_a_failed_call_is_not(tmp_path, monkeypatch):
    monkeypatch.setattr(
        caprun,
        "run_calls",
        _answers(lambda item, draw: {"answer": "   "} if item == "empty" else None),
    )
    monkeypatch.setattr(
        caprun,
        "build",
        lambda f: [capq.Question("empty", "ask", ["g"]), capq.Question("failed", "ask", ["g"])],
    )
    r = caprun.run_family("poetry", model="m", judge=_judge(), out=tmp_path, draws=4)
    rows = {row["id"]: row for row in r["rows"]}
    assert rows["empty"]["rate"] == 0.0 and rows["empty"]["passes_gate"] is False
    assert rows["failed"]["rate"] is None and rows["failed"]["passes_gate"] is None
    a = r["aggregate"]
    assert a["n_answers_empty"] == 5 and a["n_answers_failed"] == 5  # 4 draws + the greedy one
    assert a["n_items_decided"] == 1 and a["gate_rate"] == 0.0


def test_mostly_failed_item_is_undecided_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(
        caprun,
        "run_calls",
        _answers(lambda item, draw: {"answer": "right"} if draw in (0, caprun.GREEDY) else None),
    )
    monkeypatch.setattr(caprun, "build", lambda f: [capq.Question("thin", "ask", ["g"])])
    r = caprun.run_family("poetry", model="m", judge=_judge(), out=tmp_path, draws=10)
    row = r["rows"][0]
    assert caprun.min_decided(10, 0.8) == 8
    assert row["n_decided"] == 1 and row["rate"] == 1.0 and row["passes_gate"] is None
    assert r["aggregate"]["gate_rate"] is None


def test_identical_answers_are_graded_once_per_item(tmp_path, monkeypatch):
    grade_calls = []

    def fake(calls, **kw):
        if calls and calls[0].system == caprun.GRADE_SYSTEM:
            grade_calls.extend(calls)
        return _answers(lambda i, d: {"answer": "right"})(calls, **kw)

    monkeypatch.setattr(caprun, "run_calls", fake)
    monkeypatch.setattr(
        caprun,
        "build",
        lambda f: [capq.Question("a", "ask", ["g"]), capq.Question("b", "ask", ["g"])],
    )
    caprun.run_family("poetry", model="m", judge=_judge(), out=tmp_path, draws=5)
    # one grade per (item, distinct answer): two items answering the same text still get two
    assert len(grade_calls) == 2
    assert {c.meta["item"] for c in grade_calls} == {"a", "b"}


def test_cli_dry_run_and_unknown_family(capsys):
    assert main(["capable", "model=m", "families=poetry", "limit=2", "dry_run=True"]) == 0
    out = capsys.readouterr().out
    assert "2 questions" in out and "grade prompt" in out
    assert main(["capable", "model=m", "families=hallucination"]) == 2
    assert "no capability question" in capsys.readouterr().err


def test_cli_writes_one_file_per_family(tmp_path, monkeypatch):
    monkeypatch.setattr(caprun, "run_calls", _answers(lambda i, d: {"answer": "right"}))
    argv = ["capable", "model=m", "families=poetry", "limit=2", "draws=1", f"out={tmp_path}"]
    assert main(argv) == 0
    written = json.loads((tmp_path / "poetry" / "capable.json").read_text())
    assert written["model"] == "m" and written["aggregate"]["gate_rate"] == 1.0
    assert written["threshold"] == 0.8 and written["greedy"] is True
