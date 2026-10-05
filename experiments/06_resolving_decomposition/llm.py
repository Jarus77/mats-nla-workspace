"""Minimal chat-completion client for OpenRouter (Gemini) and Fireworks (kimi-k3).

Keys come from the repo .env (OPENROUTER_API_KEY; FIREWORKS_API_KEY or FIREWORKS_API) and are
never printed. Spend: OpenRouter reports usage.cost; Fireworks reports tokens only, so its cost
is estimated from tokens at FIREWORKS_PRICE (USD per 1M tokens, in/out).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ENV = Path(__file__).resolve().parents[2] / ".env"
GEMINI = "google/gemini-3.8-flash"
KIMI = "accounts/fireworks/models/kimi-k3"
FIREWORKS_PRICE = (1.0, 4.0)  # conservative upper estimate, USD / 1M tokens (in, out)


def _key(*names: str) -> str:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    for line in ENV.read_text(encoding="utf-8").splitlines():
        for n in names:
            if line.startswith(n + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"none of {names} found")


@dataclass
class Client:
    model: str = GEMINI
    temperature: float = 1.0
    max_tokens: int = 400
    reasoning: str | None = "minimal"
    timeout: float = 120.0
    retries: int = 5
    spend_usd: float = field(default=0.0, init=False)
    n_calls: int = field(default=0, init=False)
    n_failed: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def fireworks(self) -> bool:
        return self.model.startswith("accounts/fireworks/")

    def chat(self, user: str, json_schema: dict | None = None, system: str | None = None) -> str | None:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        body: dict = {"model": self.model, "messages": msgs, "temperature": self.temperature,
                      "max_tokens": self.max_tokens}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "out", "strict": True, "schema": json_schema}}
        if self.fireworks:
            url = "https://api.fireworks.ai/inference/v1/chat/completions"
            key = _key("FIREWORKS_API_KEY", "FIREWORKS_API")
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            key = _key("OPENROUTER_API_KEY")
            if self.reasoning:
                body["reasoning"] = {"effort": self.reasoning}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(self.retries):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=self.timeout))
                u = r.get("usage", {}) or {}
                with self._lock:
                    self.n_calls += 1
                    if self.fireworks:
                        self.spend_usd += (u.get("prompt_tokens", 0) * FIREWORKS_PRICE[0]
                                           + u.get("completion_tokens", 0) * FIREWORKS_PRICE[1]) / 1e6
                    else:
                        self.spend_usd += float(u.get("cost") or 0.0)
                return r["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                msg = e.read()[:300].decode(errors="replace")
                if e.code in (400, 401, 403, 404):
                    print("HTTP", e.code, msg, flush=True)
                    break
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
                time.sleep(2 ** attempt)
        with self._lock:
            self.n_failed += 1
        return None
