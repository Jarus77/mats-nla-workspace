"""One structured-JSON judge client with two routes.

* ``claude-*`` model ids go to the Anthropic SDK (structured outputs via ``output_config``);
  everything else goes to OpenRouter through the OpenAI SDK (``response_format`` json_schema).
* Requests are paced under a process-wide requests-per-minute cap shared by every event loop
  and thread (``_PACER``), so several families judged concurrently share one budget.
* Transient failures (408/409/429/5xx, timeouts, connection errors, garbled bodies) are
  retried with jittered backoff; any other per-call failure yields ``None`` (an unjudged cell,
  never a clean one). Problems that would fail every call raise :class:`JudgeConfigError`.
* ``on_result(i, result)`` fires as each call lands so callers can cache as they go.
"""

import asyncio
import json
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

OPENROUTER = "https://openrouter.ai/api/v1"
FIREWORKS = "https://api.fireworks.ai/inference/v1"
FIREWORKS_PREFIX = "accounts/fireworks/"
_TRANSIENT = (408, 409, 429, 500, 502, 503, 504, 529)
_FATAL = ("AuthenticationError", "PermissionDeniedError", "NotFoundError")
_ATTEMPTS = 12
_DEFAULT_MAX_TOKENS = {"openrouter": 8000, "anthropic": 16000, "fireworks": 8000}

Route = Literal["anthropic", "openrouter", "fireworks"]


class JudgeConfigError(RuntimeError):
    """A configuration problem that would fail every judge call (raised, never degraded)."""


@dataclass
class Spend:
    """Per-run tally. ``calls`` counts HTTP 200 responses (including garbled bodies that were
    then retried); ``errors``/``refusals`` count cells left unjudged."""

    usd: float = 0.0
    calls: int = 0
    retries: int = 0
    errors: int = 0
    refusals: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def report(self) -> str:
        s = (
            f"calls={self.calls} retries={self.retries} errors={self.errors} "
            f"refusals={self.refusals} spend=${self.usd:.2f}"
        )
        if self.input_tokens or self.output_tokens:
            s += f" in_tok={self.input_tokens} out_tok={self.output_tokens}"
        return s


def route(model: str) -> Route:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(FIREWORKS_PREFIX):
        return "fireworks"
    return "openrouter"


def api_key(model: str) -> str:
    if route(model) == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise JudgeConfigError(f"ANTHROPIC_API_KEY is missing (needed for {model})")
        return key
    if route(model) == "fireworks":
        key = os.environ.get("FIREWORKS_API_KEY", "") or os.environ.get("FIREWORKS_API", "")
        if not key:
            raise JudgeConfigError(f"FIREWORKS_API_KEY is missing (needed for {model})")
        return key
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key.startswith("sk-or-"):
        raise JudgeConfigError("OPENROUTER_API_KEY is missing or not an OpenRouter key (sk-or-…)")
    return key


def schema_block(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": list(required),
            "properties": properties,
        },
    }


# ---------------------------------------------------------------- pacing / backoff seams


def _now() -> float:
    return time.monotonic()


class _Pacer:
    """Process-wide next-slot time behind a lock: shared by every loop, thread and Spend."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0

    def reset(self) -> None:
        with self._lock:
            self._next = 0.0

    def reserve(self, rpm: float) -> float:
        """Reserve the next slot; return how long the caller must wait for it (seconds)."""
        with self._lock:
            now = _now()
            slot = max(now, self._next)
            self._next = slot + 60.0 / rpm
            return slot - now


_PACER = _Pacer()


_sleep = asyncio.sleep  # seam: tests patch this instead of the global asyncio.sleep


async def _pace(rpm: float) -> None:
    delay = _PACER.reserve(rpm)
    if delay > 0:
        await _sleep(delay)


def _backoff(attempt: int, status: int | None) -> float:
    cap = 90.0 if status == 429 else 30.0
    return min(cap, 2.0 * 2**attempt) * (0.5 + random.random())


def _make_client(route: str, key: str) -> Any:
    """The ONLY place SDK clients are built (tests monkeypatch this to inject fakes)."""
    if route == "anthropic":
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=key, max_retries=0)
    from openai import AsyncOpenAI

    base_url = FIREWORKS if route == "fireworks" else OPENROUTER
    return AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0)


# ---------------------------------------------------------------- one call


def _parse_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    data = json.loads(content)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("not a JSON object", content, 0)
    return data


async def _openrouter_once(
    client: Any,
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    reasoning: dict[str, Any] | None,
    temperature: float | None,
    max_tokens: int,
    timeout: float,
    spend: Spend,
) -> dict[str, Any] | None:
    extra_body: dict[str, Any] = {
        "usage": {"include": True},
        "provider": {"require_parameters": True},
    }
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = await client.chat.completions.create(
        timeout=timeout,
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": schema},
        extra_body=extra_body,
        **kwargs,
    )
    spend.calls += 1
    if not getattr(resp, "choices", None):  # an error delivered in a 200 body
        raise ValueError("response has no choices")
    usage = getattr(resp, "usage", None)
    cost = getattr(usage, "cost", None) if usage is not None else None
    if cost is None and usage is not None:
        cost = (getattr(usage, "model_extra", None) or {}).get("cost")
    spend.usd += float(cost or 0.0)
    return _parse_object(resp.choices[0].message.content or "")


async def _fireworks_once(
    client: Any,
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    temperature: float | None,
    max_tokens: int,
    timeout: float,
    spend: Spend,
) -> dict[str, Any] | None:
    """OpenAI-compatible call without OpenRouter's extra_body fields. Fireworks reports token
    usage, not cost, so ``spend.usd`` is not incremented on this route."""
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = await client.chat.completions.create(
        timeout=timeout,
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": schema},
        **kwargs,
    )
    spend.calls += 1
    if not getattr(resp, "choices", None):
        raise ValueError("response has no choices")
    usage = getattr(resp, "usage", None)
    if usage is not None:
        spend.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        spend.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
    return _parse_object(resp.choices[0].message.content or "")


async def _anthropic_once(
    client: Any,
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    max_tokens: int,
    timeout: float,
    spend: Spend,
) -> dict[str, Any] | None:
    resp = await client.messages.create(
        timeout=timeout,
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema["schema"]}},
    )
    spend.calls += 1
    usage = getattr(resp, "usage", None)
    if usage is not None:
        spend.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        spend.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
    stop = getattr(resp, "stop_reason", None)
    if stop == "refusal":
        spend.refusals += 1
        print(f"  llm refusal: {model}")
        return None
    if stop in ("max_tokens", "model_context_window_exceeded"):
        spend.errors += 1
        print(f"  llm error: {model} stop_reason={stop}")
        return None
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if text is None:
        raise ValueError("response has no text block")
    return _parse_object(text)


async def _one(
    route_: Route,
    client: Any,
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    reasoning: dict[str, Any] | None,
    temperature: float | None,
    max_tokens: int,
    timeout: float,
    rpm: float,
    spend: Spend,
) -> dict[str, Any] | None:
    # detokenised lens readouts can contain lone surrogates, which both APIs reject every time
    system = system.encode("utf-8", "replace").decode("utf-8")
    user = user.encode("utf-8", "replace").decode("utf-8")
    for attempt in range(_ATTEMPTS):
        await _pace(rpm)
        try:
            if route_ == "anthropic":
                return await _anthropic_once(
                    client,
                    system,
                    user,
                    schema=schema,
                    model=model,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    spend=spend,
                )
            if route_ == "fireworks":
                return await _fireworks_once(
                    client,
                    system,
                    user,
                    schema=schema,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    spend=spend,
                )
            return await _openrouter_once(
                client,
                system,
                user,
                schema=schema,
                model=model,
                reasoning=reasoning,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                spend=spend,
            )
        except Exception as e:
            name = type(e).__name__
            status = getattr(e, "status_code", None)
            if name in _FATAL or status in (401, 402, 403):  # bad key, no credits, no access
                raise JudgeConfigError(f"{name}: {str(e)[:300]}") from e
            transient = (
                status in _TRANSIENT
                or "Timeout" in name
                or "Connection" in name
                or "Deadline" in name
                or isinstance(e, json.JSONDecodeError | ValueError)
            )
            if transient and attempt < _ATTEMPTS - 1:
                spend.retries += 1
                await _sleep(_backoff(attempt, status))
                continue
            spend.errors += 1
            print(f"  llm error: {name}: {str(e)[:200]}")
            return None
    return None


# ---------------------------------------------------------------- the primitive


async def stream_json_async(
    prompts: list[tuple[str, str]],
    *,
    schema: dict[str, Any],
    model: str,
    on_result: Callable[[int, dict[str, Any] | None], None],
    reasoning: dict[str, Any] | None = None,
    temperature: float | None = None,
    concurrency: int = 64,
    rpm: float = 240.0,
    max_tokens: int | None = None,
    timeout: float = 180.0,
    spend: Spend | None = None,
) -> Spend:
    """Run ``(system, user)`` prompt pairs concurrently, handing each parsed result (or
    ``None``) to ``on_result(index, result)`` as it lands. Raises :class:`JudgeConfigError` on
    a config problem. Builds no client when there is nothing to call. ``temperature`` is sent on
    the OpenRouter route only: newer Claude models reject the field (400), so the Anthropic route
    drops it with one printed warning per batch."""
    spend = spend or Spend()
    if not prompts:
        return spend
    if concurrency < 1 or rpm <= 0:
        raise JudgeConfigError(f"concurrency must be >= 1 and rpm > 0 (got {concurrency}, {rpm})")
    route_ = route(model)
    key = api_key(model)
    if temperature is not None and route_ == "anthropic":
        print(f"  llm: dropping temperature={temperature} for {model} (Anthropic route rejects it)")
        temperature = None
    if max_tokens is None:
        max_tokens = _DEFAULT_MAX_TOKENS[route_]
    client = _make_client(route_, key)
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int, system: str, user: str) -> tuple[int, dict[str, Any] | None]:
        async with sem:
            res = await _one(
                route_,
                client,
                system,
                user,
                schema=schema,
                model=model,
                reasoning=reasoning,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                rpm=rpm,
                spend=spend,
            )
        return i, res

    tasks = [asyncio.ensure_future(one(i, s, u)) for i, (s, u) in enumerate(prompts)]
    try:
        for fut in asyncio.as_completed(tasks):
            i, res = await fut
            on_result(i, res)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
    return spend


def _in_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def stream_json(
    prompts: list[tuple[str, str]],
    *,
    schema: dict[str, Any],
    model: str,
    on_result: Callable[[int, dict[str, Any] | None], None],
    reasoning: dict[str, Any] | None = None,
    temperature: float | None = None,
    concurrency: int = 64,
    rpm: float = 240.0,
    max_tokens: int | None = None,
    timeout: float = 180.0,
    spend: Spend | None = None,
) -> Spend:
    """Synchronous wrapper around :func:`stream_json_async` for the CLI and tests."""
    if _in_running_loop():
        raise JudgeConfigError(
            "stream_json called inside a running event loop; await stream_json_async"
        )
    return asyncio.run(
        stream_json_async(
            prompts,
            schema=schema,
            model=model,
            on_result=on_result,
            reasoning=reasoning,
            temperature=temperature,
            concurrency=concurrency,
            rpm=rpm,
            max_tokens=max_tokens,
            timeout=timeout,
            spend=spend,
        )
    )


async def preflight_async(model: str, reasoning: dict[str, Any] | None) -> None:
    """One tiny structured call, so a bad key/model/route fails before a long run starts."""
    out: list[dict[str, Any] | None] = []
    schema = schema_block("t", {"a": {"type": "integer"}}, ["a"])
    await stream_json_async(
        [("Reply JSON.", 'Return {"a":1}')],
        schema=schema,
        model=model,
        reasoning=reasoning,
        concurrency=1,
        on_result=lambda _i, r: out.append(r),
    )
    if out != [{"a": 1}]:
        raise JudgeConfigError(f"preflight failed for {model}: {out}")


def preflight(model: str, reasoning: dict[str, Any] | None) -> None:
    if _in_running_loop():
        raise JudgeConfigError(
            "preflight called inside a running event loop; await preflight_async"
        )
    asyncio.run(preflight_async(model, reasoning))


# ---------------------------------------------------------------- free-text primitive
# Used by agentic_misalignment (three free-text stages, no schema). Anthropic route only:
# streaming (the SDK refuses non-streaming requests whose max_tokens implies a >10 min
# operation), per-call thinking on/off, and a budget-doubling retry when the text is empty and
# stop_reason == "max_tokens" (thinking ate the budget), up to _TEXT_BUDGET_CEILING.

_TEXT_BUDGET_CEILING = 64_000


async def _anthropic_text_once(
    client: Any,
    user: str,
    *,
    model: str,
    thinking: bool,
    max_tokens: int,
    timeout: float,
    spend: Spend,
) -> str | None:
    budget = max_tokens
    while True:
        async with client.messages.stream(
            timeout=timeout,
            model=model,
            max_tokens=budget,
            thinking={"type": "adaptive" if thinking else "disabled"},
            messages=[{"role": "user", "content": user}],
        ) as stream:
            resp = await stream.get_final_message()
        spend.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            spend.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            spend.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            spend.refusals += 1
            print(f"  llm refusal: {model}")
            return None
        txt = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        if not txt and stop == "max_tokens" and budget < _TEXT_BUDGET_CEILING:
            budget *= 2
            continue
        return txt


async def _one_text(
    client: Any,
    user: str,
    *,
    model: str,
    thinking: bool,
    max_tokens: int,
    timeout: float,
    rpm: float,
    spend: Spend,
) -> str | None:
    """Same backoff / fatal classification as :func:`_one`, for one free-text call."""
    user = user.encode("utf-8", "replace").decode("utf-8")
    for attempt in range(_ATTEMPTS):
        await _pace(rpm)
        try:
            return await _anthropic_text_once(
                client,
                user,
                model=model,
                thinking=thinking,
                max_tokens=max_tokens,
                timeout=timeout,
                spend=spend,
            )
        except Exception as e:
            name = type(e).__name__
            status = getattr(e, "status_code", None)
            if name in _FATAL or status in (401, 402, 403):
                raise JudgeConfigError(f"{name}: {str(e)[:300]}") from e
            transient = (
                status in _TRANSIENT
                or "Timeout" in name
                or "Connection" in name
                or "Deadline" in name
                or isinstance(e, json.JSONDecodeError | ValueError)
            )
            if transient and attempt < _ATTEMPTS - 1:
                spend.retries += 1
                await _sleep(_backoff(attempt, status))
                continue
            spend.errors += 1
            print(f"  llm error: {name}: {str(e)[:200]}")
            return None
    return None


async def stream_text_async(
    prompts: list[str],
    *,
    model: str,
    on_result: Callable[[int, str | None], None],
    thinking: bool,
    max_tokens: int,
    concurrency: int = 64,
    rpm: float = 240.0,
    timeout: float = 600.0,
    spend: Spend | None = None,
) -> Spend:
    """Run user-only free-text prompts concurrently on the Anthropic route (no system block,
    no schema), handing each stripped text (or ``None`` = exhausted retries / non-transient
    error / refusal) to ``on_result(index, text)`` as it lands. ``""`` is a valid result (the
    source returns it after the 64k budget ceiling). Any non-``claude-*`` model raises
    :class:`JudgeConfigError`. Builds no client when there is nothing to call."""
    spend = spend or Spend()
    if not prompts:
        return spend
    if concurrency < 1 or rpm <= 0:
        raise JudgeConfigError(f"concurrency must be >= 1 and rpm > 0 (got {concurrency}, {rpm})")
    if route(model) != "anthropic":
        raise JudgeConfigError(f"free-text stage requires a claude-* model (got {model})")
    client = _make_client("anthropic", api_key(model))
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int, user: str) -> tuple[int, str | None]:
        async with sem:
            res = await _one_text(
                client,
                user,
                model=model,
                thinking=thinking,
                max_tokens=max_tokens,
                timeout=timeout,
                rpm=rpm,
                spend=spend,
            )
        return i, res

    tasks = [asyncio.ensure_future(one(i, u)) for i, u in enumerate(prompts)]
    try:
        for fut in asyncio.as_completed(tasks):
            i, res = await fut
            on_result(i, res)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
    return spend


def stream_text(
    prompts: list[str],
    *,
    model: str,
    on_result: Callable[[int, str | None], None],
    thinking: bool,
    max_tokens: int,
    concurrency: int = 64,
    rpm: float = 240.0,
    timeout: float = 600.0,
    spend: Spend | None = None,
) -> Spend:
    """Synchronous wrapper around :func:`stream_text_async` for the CLI and tests."""
    if _in_running_loop():
        raise JudgeConfigError(
            "stream_text called inside a running event loop; await stream_text_async"
        )
    return asyncio.run(
        stream_text_async(
            prompts,
            model=model,
            on_result=on_result,
            thinking=thinking,
            max_tokens=max_tokens,
            concurrency=concurrency,
            rpm=rpm,
            timeout=timeout,
            spend=spend,
        )
    )
