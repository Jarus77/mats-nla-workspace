"""Offline tests for the structured-JSON client: fakes are injected via ``_make_client``."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from wsbench import llm
from wsbench.llm import JudgeConfigError, Spend, schema_block, stream_json, stream_json_async

SCHEMA = schema_block("t", {"a": {"type": "integer"}}, ["a"])
GEMINI = "google/gemini-3.8-flash"
CLAUDE = "claude-sonnet-5"


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


class APITimeoutError(Exception):
    status_code = None


def _or_response(content: str, cost: float = 0.01) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(cost=cost),
    )


class FakeOpenRouter:
    """``responses`` is a list per prompt index... simplified: one shared queue of outcomes."""

    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    async def close(self) -> None:
        self.closed = True


class FakeAnthropic:
    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    async def close(self) -> None:
        self.closed = True


def _claude_response(text: str, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(llm, "_backoff", lambda attempt, status: 0.0)

    async def no_pace(rpm: float) -> None:
        return None

    monkeypatch.setattr(llm, "_pace", no_pace)


def _install(monkeypatch, fake: Any) -> Any:
    monkeypatch.setattr(llm, "_make_client", lambda route, key: fake)
    return fake


def _collect(prompts, model, fake, monkeypatch, **kw):
    _install(monkeypatch, fake)
    got: dict[int, dict | None] = {}
    spend = stream_json(
        prompts, schema=SCHEMA, model=model, on_result=lambda i, r: got.__setitem__(i, r), **kw
    )
    return got, spend


# ---------------------------------------------------------------- routing / keys / schema


def test_route():
    assert llm.route(CLAUDE) == "anthropic"
    assert llm.route(GEMINI) == "openrouter"


def test_api_key_missing_and_malformed(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(JudgeConfigError):
        llm.api_key(GEMINI)
    with pytest.raises(JudgeConfigError):
        llm.api_key(CLAUDE)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-proj-not-openrouter")
    with pytest.raises(JudgeConfigError):
        llm.api_key(GEMINI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-ok")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ok")
    assert llm.api_key(GEMINI) == "sk-or-ok"
    assert llm.api_key(CLAUDE) == "sk-ant-ok"


def test_schema_block_shape():
    assert SCHEMA == {
        "name": "t",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["a"],
            "properties": {"a": {"type": "integer"}},
        },
    }


def test_spend_report():
    s = Spend(usd=1.5, calls=3, retries=1, errors=0, refusals=0)
    assert s.report() == "calls=3 retries=1 errors=0 refusals=0 spend=$1.50"
    s.input_tokens = 10
    s.output_tokens = 2
    assert s.report().endswith(" in_tok=10 out_tok=2")


# ---------------------------------------------------------------- OpenRouter route


def test_openrouter_success(keys, fast, monkeypatch):
    fake = FakeOpenRouter([_or_response('{"a": 1}', cost=0.25)])
    got, spend = _collect([("sys", "usr")], GEMINI, fake, monkeypatch, reasoning={"effort": "low"})
    assert got == {0: {"a": 1}}
    assert spend.calls == 1 and spend.errors == 0 and spend.usd == pytest.approx(0.25)
    kw = fake.calls[0]
    assert kw["model"] == GEMINI
    assert kw["max_tokens"] == 8000
    assert kw["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert kw["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
    assert kw["extra_body"]["reasoning"] == {"effort": "low"}
    assert kw["extra_body"]["provider"] == {"require_parameters": True}
    assert fake.closed


def test_openrouter_reasoning_omitted_when_none(keys, fast, monkeypatch):
    fake = FakeOpenRouter([_or_response('{"a": 1}')])
    _collect([("s", "u")], GEMINI, fake, monkeypatch)
    assert "reasoning" not in fake.calls[0]["extra_body"]


def test_openrouter_fenced_json(keys, fast, monkeypatch):
    fake = FakeOpenRouter([_or_response('```json\n{"a": 2}\n```')])
    got, _ = _collect([("s", "u")], GEMINI, fake, monkeypatch)
    assert got == {0: {"a": 2}}


def test_openrouter_transient_then_success(keys, fast, monkeypatch):
    fake = FakeOpenRouter([RateLimitError("slow down"), _or_response('{"a": 1}')])
    got, spend = _collect([("s", "u")], GEMINI, fake, monkeypatch)
    assert got == {0: {"a": 1}}
    assert spend.retries == 1 and spend.errors == 0 and spend.calls == 1


def test_openrouter_timeout_is_transient(keys, fast, monkeypatch):
    fake = FakeOpenRouter([APITimeoutError("t"), _or_response('{"a": 1}')])
    got, spend = _collect([("s", "u")], GEMINI, fake, monkeypatch)
    assert got == {0: {"a": 1}} and spend.retries == 1


def test_openrouter_fatal_raises(keys, fast, monkeypatch):
    fake = FakeOpenRouter([AuthenticationError("bad key")])
    _install(monkeypatch, fake)
    with pytest.raises(JudgeConfigError):
        stream_json([("s", "u")], schema=SCHEMA, model=GEMINI, on_result=lambda i, r: None)
    assert fake.closed


def test_openrouter_non_object_after_retries(keys, fast, monkeypatch):
    fake = FakeOpenRouter([_or_response("[1, 2]")] * 12)
    got, spend = _collect([("s", "u")], GEMINI, fake, monkeypatch)
    assert got == {0: None}
    assert spend.errors == 1 and spend.retries == 11
    assert fake.outcomes == []


def test_on_result_order_independent(keys, fast, monkeypatch):
    order: list[int] = []

    class Slow(FakeOpenRouter):
        async def _create(self, **kw: Any) -> Any:
            self.calls.append(kw)
            n = len(self.calls)
            await asyncio.sleep(0.02 if n == 1 else 0.0)
            return _or_response(json.dumps({"a": n}))

    fake = Slow([])
    _install(monkeypatch, fake)
    got: dict[int, Any] = {}

    def on_result(i: int, r: dict | None) -> None:
        order.append(i)
        got[i] = r

    stream_json([("s", "u"), ("s", "u2")], schema=SCHEMA, model=GEMINI, on_result=on_result)
    assert set(got) == {0, 1}
    assert order == [1, 0]


def test_lone_surrogate_sanitised(keys, fast, monkeypatch):
    fake = FakeOpenRouter([_or_response('{"a": 1}')])
    _collect([("s\ud800", "u\udfff x")], GEMINI, fake, monkeypatch)
    msgs = fake.calls[0]["messages"]
    for m in msgs:
        m["content"].encode("utf-8")  # must not raise
    assert msgs[1]["content"] == "u? x"


def test_stream_json_inside_loop_raises_but_async_works(keys, fast, monkeypatch):
    fake = FakeOpenRouter([_or_response('{"a": 1}')])
    _install(monkeypatch, fake)

    async def inner():
        with pytest.raises(JudgeConfigError):
            stream_json([("s", "u")], schema=SCHEMA, model=GEMINI, on_result=lambda i, r: None)
        got: dict[int, Any] = {}
        await stream_json_async(
            [("s", "u")], schema=SCHEMA, model=GEMINI, on_result=lambda i, r: got.__setitem__(i, r)
        )
        return got

    assert asyncio.run(inner()) == {0: {"a": 1}}


def test_pacer_shared_across_spends(keys, monkeypatch):
    monkeypatch.setattr(llm, "_backoff", lambda attempt, status: 0.0)
    monkeypatch.setattr(llm, "_now", lambda: 100.0)
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(llm, "_sleep", fake_sleep)
    fake = FakeOpenRouter([_or_response('{"a": 1}'), _or_response('{"a": 1}')])
    _install(monkeypatch, fake)
    s1, s2 = Spend(), Spend()
    stream_json(
        [("s", "u")], schema=SCHEMA, model=GEMINI, on_result=lambda i, r: None, rpm=60, spend=s1
    )
    stream_json(
        [("s", "u")], schema=SCHEMA, model=GEMINI, on_result=lambda i, r: None, rpm=60, spend=s2
    )
    # first call takes slot t=100 (no sleep); the second, in a new loop with a new Spend,
    # sees the shared pacer's next slot t=101 and must wait 1s.
    assert sleeps == [1.0]
    assert s1.calls == 1 and s2.calls == 1


def test_pace_reserves_slots():
    monkeypatch_now = [100.0]
    orig = llm._now
    llm._now = lambda: monkeypatch_now[0]
    try:
        llm._PACER.reset()
        assert llm._PACER.reserve(60.0) == 0.0
        assert llm._PACER.reserve(60.0) == pytest.approx(1.0)
        assert llm._PACER.reserve(60.0) == pytest.approx(2.0)
        monkeypatch_now[0] = 200.0
        assert llm._PACER.reserve(60.0) == 0.0
    finally:
        llm._now = orig


def test_backoff_bounds(monkeypatch):
    monkeypatch.setattr(llm.random, "random", lambda: 0.5)
    assert llm._backoff(0, None) == pytest.approx(2.0)
    assert llm._backoff(10, None) == pytest.approx(30.0)
    assert llm._backoff(10, 429) == pytest.approx(90.0)


# ---------------------------------------------------------------- Anthropic route


def test_anthropic_success(keys, fast, monkeypatch):
    fake = FakeAnthropic([_claude_response('{"a": 1}')])
    got, spend = _collect([("sys", "usr")], CLAUDE, fake, monkeypatch, reasoning={"effort": "x"})
    assert got == {0: {"a": 1}}
    assert spend.calls == 1 and spend.input_tokens == 7 and spend.output_tokens == 3
    assert spend.usd == 0.0
    kw = fake.calls[0]
    assert kw["model"] == CLAUDE and kw["max_tokens"] == 16000
    assert kw["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert kw["messages"] == [{"role": "user", "content": "usr"}]
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA["schema"]}}
    assert "reasoning" not in kw and "extra_body" not in kw
    assert fake.closed


def test_anthropic_refusal(keys, fast, monkeypatch):
    fake = FakeAnthropic([_claude_response("", stop_reason="refusal")])
    got, spend = _collect([("s", "u")], CLAUDE, fake, monkeypatch)
    assert got == {0: None}
    assert spend.refusals == 1 and spend.errors == 0 and len(fake.calls) == 1


def test_anthropic_max_tokens(keys, fast, monkeypatch):
    fake = FakeAnthropic([_claude_response("{", stop_reason="max_tokens")])
    got, spend = _collect([("s", "u")], CLAUDE, fake, monkeypatch)
    assert got == {0: None}
    assert spend.errors == 1 and spend.refusals == 0 and len(fake.calls) == 1


# ---------------------------------------------------------------- preflight / edge cases


def test_preflight_pass_and_fail(keys, fast, monkeypatch):
    _install(monkeypatch, FakeOpenRouter([_or_response('{"a": 1}')]))
    llm.preflight(GEMINI, None)
    fake = _install(monkeypatch, FakeOpenRouter([_or_response('{"a": 2}')]))
    with pytest.raises(JudgeConfigError):
        llm.preflight(GEMINI, None)
    assert fake.calls[0]["messages"][1]["content"] == 'Return {"a":1}'


def test_empty_prompts_build_no_client(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def boom(route, key):
        raise AssertionError("client must not be built")

    monkeypatch.setattr(llm, "_make_client", boom)
    spend = stream_json([], schema=SCHEMA, model=GEMINI, on_result=lambda i, r: None)
    assert spend.calls == 0


def test_bad_concurrency_and_rpm(keys, monkeypatch):
    _install(monkeypatch, FakeOpenRouter([]))
    with pytest.raises(JudgeConfigError):
        stream_json(
            [("s", "u")], schema=SCHEMA, model=GEMINI, on_result=lambda i, r: None, concurrency=0
        )
    with pytest.raises(JudgeConfigError):
        stream_json([("s", "u")], schema=SCHEMA, model=GEMINI, on_result=lambda i, r: None, rpm=0)


def test_openrouter_fatal_cancels_inflight_task(keys, fast, monkeypatch):
    """A 401 on one call must cancel a hung sibling call and close the client."""
    cancelled: list[bool] = []

    class Hanging(FakeOpenRouter):
        async def _create(self, **kw: Any) -> Any:
            self.calls.append(kw)
            if kw["messages"][1]["content"] == "hang":
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise
            raise AuthenticationError("bad key")

    fake = Hanging([])
    _install(monkeypatch, fake)
    with pytest.raises(JudgeConfigError):
        stream_json(
            [("s", "hang"), ("s", "die")],
            schema=SCHEMA,
            model=GEMINI,
            on_result=lambda i, r: None,
        )
    assert cancelled == [True]
    assert fake.closed


def test_anthropic_context_window_stop_is_error_not_retry(keys, fast, monkeypatch):
    fake = FakeAnthropic([_claude_response("{", stop_reason="model_context_window_exceeded")])
    got, spend = _collect([("s", "u")], CLAUDE, fake, monkeypatch)
    assert got == {0: None}
    assert spend.errors == 1 and spend.retries == 0 and len(fake.calls) == 1


def test_real_clients_are_built_with_expected_kwargs(monkeypatch):
    """Guards the SDK call shapes without any network: client construction + create() signatures."""
    import inspect

    from anthropic.resources.messages import AsyncMessages
    from openai.resources.chat.completions import AsyncCompletions

    orc = llm._make_client("openrouter", "sk-or-x")
    assert str(orc.base_url).rstrip("/") == llm.OPENROUTER and orc.max_retries == 0
    anc = llm._make_client("anthropic", "k")
    assert anc.max_retries == 0
    a = inspect.signature(AsyncMessages.create).parameters
    assert {"output_config", "system", "timeout", "max_tokens"} <= set(a)
    o = inspect.signature(AsyncCompletions.create).parameters
    assert {"extra_body", "response_format", "max_tokens", "timeout"} <= set(o)


# ---------------------------------------------------------------- temperature (phase 4)


def test_openrouter_temperature_passed_or_omitted(keys, fast, monkeypatch):
    fake = FakeOpenRouter([_or_response('{"a": 1}'), _or_response('{"a": 1}')])
    _collect([("s", "u")], GEMINI, fake, monkeypatch, temperature=0.0)
    assert fake.calls[0]["temperature"] == 0.0
    _collect([("s", "u")], GEMINI, fake, monkeypatch)
    assert "temperature" not in fake.calls[1]


def test_openrouter_max_tokens_override(keys, fast, monkeypatch):
    fake = FakeOpenRouter([_or_response('{"a": 1}')])
    _collect([("s", "u")], GEMINI, fake, monkeypatch, max_tokens=16000)
    assert fake.calls[0]["max_tokens"] == 16000


def test_anthropic_temperature_dropped_with_one_warning(keys, fast, monkeypatch, capsys):
    fake = FakeAnthropic([_claude_response('{"a": 1}'), _claude_response('{"a": 1}')])
    got, _ = _collect([("s", "u"), ("s", "u2")], CLAUDE, fake, monkeypatch, temperature=0.0)
    assert got == {0: {"a": 1}, 1: {"a": 1}}
    assert all("temperature" not in kw for kw in fake.calls)
    out = capsys.readouterr().out
    assert out.count("temperature") == 1 and CLAUDE in out
    fake2 = FakeAnthropic([_claude_response('{"a": 1}')])
    _collect([("s", "u")], CLAUDE, fake2, monkeypatch)
    assert "temperature" not in capsys.readouterr().out
