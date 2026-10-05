from pathlib import Path

from wsbench.cache import Cache
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.llm import Spend
from wsbench.summarizer import (
    INTERP_SYSTEM,
    PROMPTS,
    SUMMARIZER_PROMPT_VERSION,
    aux_judge,
    render_bag,
    summarize,
)

JUDGE = resolve(JudgeConfig(), env={})


def test_render_bag():
    assert render_bag(["a", "b"], None) == "a | b"
    assert render_bag(["a", "b"], [1.0, 2.456]) == "b (2.46) | a (1.00)"
    assert render_bag([], None) == ""


def test_prompt_constants():
    assert SUMMARIZER_PROMPT_VERSION == "interp-v1"
    assert INTERP_SYSTEM.startswith("You are shown the top-10 token readouts")
    assert PROMPTS["INTERP_USER"] == "TOKEN READOUTS:\n{txt}"


def test_aux_judge():
    assert aux_judge(JUDGE, {}, "summarizer") is JUDGE
    other = aux_judge(JUDGE, {"summarizer": "x/y"}, "summarizer")
    assert other.model == "x/y" and other.reasoning == JUDGE.reasoning


def test_summarize_caches_and_marks_failures(tmp_path: Path, fake_llm):
    def responder(system, user):
        assert system == INTERP_SYSTEM
        if "bad" in user:
            return None
        return {"interpretation": f"about {user.splitlines()[-1]}"}

    fake = fake_llm(responder)
    spend = Spend()
    pre: list[int] = []
    with Cache(tmp_path / "c.jsonl") as cache:
        got = summarize(
            {"k1": "tok1 | tok2", "k2": "bad"},
            judge=JUDGE,
            cache=cache,
            spend=spend,
            concurrency=4,
            rpm=1e9,
            preflight=lambda: pre.append(1),
        )
    assert got == {"k1": "about tok1 | tok2", "k2": None}
    assert pre == [1]
    assert fake.calls[0][1] == "TOKEN READOUTS:\ntok1 | tok2"
    assert spend.calls == 1 and spend.errors == 1
    # second run: the success is cached, the failure is retried
    fake2 = fake_llm(lambda s, u: {"interpretation": "now ok"})
    with Cache(tmp_path / "c.jsonl") as cache:
        got = summarize(
            {"k1": "tok1 | tok2", "k2": "bad"},
            judge=JUDGE,
            cache=cache,
            spend=Spend(),
            concurrency=4,
            rpm=1e9,
        )
    assert got == {"k1": "about tok1 | tok2", "k2": "now ok"}
    assert len(fake2.calls) == 1
    # an empty interpretation is a failure, never an empty readout
    fake_llm(lambda s, u: {"interpretation": "  "})
    with Cache(tmp_path / "c2.jsonl") as cache:
        got = summarize({"k": "z"}, judge=JUDGE, cache=cache, spend=Spend(), concurrency=1, rpm=1e9)
    assert got == {"k": None}


def test_summarize_dry_run_prints_and_makes_no_calls(tmp_path: Path, capsys, monkeypatch):
    def boom(route, key):
        raise AssertionError("no client under dry run")

    monkeypatch.setattr("wsbench.llm._make_client", boom)
    with Cache(tmp_path / "c.jsonl") as cache:
        got = summarize(
            {"k": "tok"},
            judge=JUDGE,
            cache=cache,
            spend=Spend(),
            concurrency=1,
            rpm=1e9,
            dry_run=True,
        )
    assert got == {}
    out = capsys.readouterr().out
    assert INTERP_SYSTEM in out and "TOKEN READOUTS:\ntok" in out
