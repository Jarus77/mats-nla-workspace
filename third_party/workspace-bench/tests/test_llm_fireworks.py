"""Offline tests for the Fireworks route (OpenAI-compatible; added in the mats fork)."""

from types import SimpleNamespace
from typing import Any

import pytest

from wsbench import llm
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.llm import JudgeConfigError, schema_block, stream_json

SCHEMA = schema_block("t", {"a": {"type": "integer"}}, ["a"])
FW = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"


class FakeOpenAI:
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


def _resp(content: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
    )


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(llm, "_backoff", lambda attempt, status: 0.0)

    async def no_pace(rpm: float) -> None:
        return None

    monkeypatch.setattr(llm, "_pace", no_pace)


def test_route_fireworks():
    assert llm.route(FW) == "fireworks"
    assert llm.route("google/gemini-3.8-flash") == "openrouter"


def test_fireworks_key_from_either_env_name(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_API", raising=False)
    with pytest.raises(JudgeConfigError):
        llm.api_key(FW)
    monkeypatch.setenv("FIREWORKS_API", "fw_legacy")
    assert llm.api_key(FW) == "fw_legacy"
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_primary")
    assert llm.api_key(FW) == "fw_primary"


def test_fireworks_success_sends_no_openrouter_fields(fast, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test")
    fake = FakeOpenAI([_resp('{"a": 1}')])
    seen: dict[str, str] = {}

    def make(route: str, key: str) -> Any:
        seen.update(route=route, key=key)
        return fake

    monkeypatch.setattr(llm, "_make_client", make)
    got: dict[int, Any] = {}
    spend = stream_json(
        [("sys", "usr")], schema=SCHEMA, model=FW, on_result=lambda i, r: got.__setitem__(i, r)
    )
    assert got == {0: {"a": 1}}
    assert seen == {"route": "fireworks", "key": "fw_test"}
    kw = fake.calls[0]
    assert kw["model"] == FW
    assert kw["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
    assert "extra_body" not in kw
    assert spend.calls == 1 and spend.input_tokens == 11 and spend.output_tokens == 4
    assert fake.closed


def test_fireworks_client_base_url():
    c = llm._make_client("fireworks", "fw_x")
    assert str(c.base_url).rstrip("/") == llm.FIREWORKS and c.max_retries == 0


def test_fireworks_resolve_has_no_openrouter_reasoning():
    r = resolve(JudgeConfig(), flag=FW)
    assert r.model == FW and r.reasoning is None and not r.pinned
