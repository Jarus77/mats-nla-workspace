"""Shared fixtures: isolate the family registry and the RPM pacer between tests."""

import pytest

from wsbench import llm, mcjudge, registry


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(registry.FAMILIES)
    yield
    registry.FAMILIES.clear()
    registry.FAMILIES.update(saved)


@pytest.fixture(autouse=True)
def _reset_preflighted():
    mcjudge._PREFLIGHTED.clear()  # process-global "already preflighted" set (phase 6)
    yield
    mcjudge._PREFLIGHTED.clear()


@pytest.fixture(autouse=True)
def _reset_pacer():
    llm._PACER.reset()
    yield
    llm._PACER.reset()


collect_ignore = ["golden"]  # data files, never tests


class BadRequestError(Exception):
    status_code = 400  # neither transient nor fatal: one failed call, no retry


class FakeClient:
    """An OpenRouter-shaped fake: ``responder(system, user)`` returns the JSON object the model
    "wrote" (``None`` -> a non-transient error, i.e. one failed call, ``result is None``)."""

    def __init__(self, responder):
        from types import SimpleNamespace

        self.responder = responder
        self.calls: list[tuple[str, str]] = []
        self.kwargs: list[dict] = []  # every create() kwargs (model, temperature, ...)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        import json
        from types import SimpleNamespace

        system, user = kw["messages"][0]["content"], kw["messages"][1]["content"]
        self.calls.append((system, user))
        self.kwargs.append(kw)
        r = self.responder(system, user)
        if r is None:
            raise BadRequestError("fake failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(r)))],
            usage=SimpleNamespace(cost=0.001),
        )

    async def close(self):
        return None


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a FakeClient behind ``llm._make_client``; returns ``install(responder) -> fake``."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(llm, "_backoff", lambda attempt, status: 0.0)

    async def no_pace(rpm: float) -> None:
        return None

    monkeypatch.setattr(llm, "_pace", no_pace)

    def install(responder):
        fake = FakeClient(responder)
        monkeypatch.setattr(llm, "_make_client", lambda route, key: fake)
        return fake

    return install


@pytest.fixture
def mk_args(tmp_path):
    """``mk_args(readouts_path, **overrides) -> JudgeArgs`` with the family default judge."""
    from pathlib import Path

    from wsbench.judge_config import JudgeConfig, resolve
    from wsbench.registry import JudgeArgs

    def make(readouts, **kw):
        base: dict = {
            "readouts": Path(readouts),
            "out": tmp_path / "out",
            "judge": resolve(JudgeConfig(), env={}),
            "layers": None,
            "items": None,
            "limit": 0,
            "allow_missing": False,
            "concurrency": 4,
            "rpm": 1e9,
            "dry_run": False,
        }
        base.update(kw)
        return JudgeArgs(**base)

    return make


def write_jsonl(path, rows):
    import json

    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def tag_of(user: str) -> str:
    """The ``R:<tag>`` marker a scripted readout carries (tests script verdicts per cell)."""
    import re

    m = re.search(r"R:(\S+)", user)
    return m.group(1) if m else ""
