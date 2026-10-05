"""TextNLA: text-only baselines with the paired-candidate scoring analogue (prereg §5, §6.3).

Two inputs (prereg §5):
  (a) mode="prompt": the true text prefix -> "what is recoverable from the text alone";
  (b) mode="explanation": the NLA's own explanation -> the re-solving ceiling.

Two backends:
- :class:`TextNLALocal`: base Qwen3.6-27B (no activation). Teacher-forced LO = log p(a) - log p(b)
  of the same NLA-register candidate strings, after the warm-start instruction prompt
  (``av_base/nla_meta.yaml`` api_summaries.instruction_prompt, the prompt Sonnet used to write
  the NLA's warm-start explanations) and the ``<analysis>\\n`` opening. Pass a PeftModel only
  inside ``ol.base()`` (adapter off).
- :class:`TextNLAGemini`: Gemini 3.8 Flash via OpenRouter. OpenRouter returns no logprobs for
  this model (checked 2026-10-05: ``choices[0].logprobs`` is null), so the analogue is an
  ELICITED probability: the model returns P(option A) as JSON; LO = logit(p), clipped to
  [eps, 1-eps]; both option orders are asked and averaged to cancel position bias.
  The key comes from OPENROUTER_API_KEY (env or repo .env) and is never printed or logged.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

FALLBACK_INSTRUCTION = (
    "A language model needs to predict what text comes next after a snippet which will be "
    "presented to you shortly. Identify the 2-3 most important features it would use for this "
    "prediction.\n\nFormat:\n<analysis>\n[features]\n</analysis>\n\nText to analyze:\n\n"
    "<begin_text>{text}<end_text>"
)
EXPLANATION_INSTRUCTION = (
    "Below is a description of what a language model was representing internally at some "
    "point in a text. Restate the most important features it describes.\n\n"
    "<description>{text}</description>\n\nFormat:\n<analysis>\n[features]\n</analysis>"
)
ANALYSIS_PREFIX = "<analysis>\n"


def nla_instruction(hf_home: str = "/weights/hf") -> str:
    """The NLA warm-start instruction prompt from av_base/nla_meta.yaml (fallback if absent)."""
    try:
        import yaml

        from .readers.nla import snapshot_dir

        meta = yaml.safe_load((snapshot_dir(hf_home=hf_home) / "av_base" / "nla_meta.yaml")
                              .read_text(encoding="utf-8"))
        return meta["api_summaries"]["instruction_prompt"]
    except (FileNotFoundError, KeyError, ImportError):
        return FALLBACK_INSTRUCTION


# ----------------------------------------------------------------------------- local (Qwen)


@dataclass
class TextNLALocal:
    """Teacher-forced paired LO from base Qwen given only text."""

    model: Any
    tok: Any
    instruction: str = FALLBACK_INSTRUCTION
    explanation_instruction: str = EXPLANATION_INSTRUCTION
    prefix: str = ANALYSIS_PREFIX
    device: str = "cuda"

    def _head(self, text: str, mode: str) -> list[int]:
        tmpl = self.instruction if mode == "prompt" else self.explanation_instruction
        if mode not in ("prompt", "explanation"):
            raise ValueError(f"mode must be prompt|explanation, got {mode!r}")
        chat = self.tok.apply_chat_template(
            [{"role": "user", "content": tmpl.replace("{text}", text)}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(chat + self.prefix, add_special_tokens=False)

    @torch.no_grad()
    def logprob_of_strings(
        self, texts: Sequence[str], candidates: Sequence[str], mode: str = "prompt",
        batch: int = 16,
    ) -> torch.Tensor:
        """[N, C] summed log p(candidate | instruction(text), prefix). Candidates of one text
        share a batch."""
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        cands = [self.tok.encode(c, add_special_tokens=False) for c in candidates]
        C = len(cands)
        per = max(1, batch // C)
        pad = self.tok.pad_token_id or 0
        out = torch.empty(len(texts), C)
        for s in range(0, len(texts), per):
            heads = [self._head(t, mode) for t in texts[s : s + per]]
            rows = [(i, c) for i in range(len(heads)) for c in range(C)]
            seqs = [heads[i] + cands[c] for i, c in rows]
            n = max(map(len, seqs))
            ids = torch.full((len(rows), n), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, q in enumerate(seqs):
                ids[r, : len(q)] = torch.tensor(q)
                mask[r, : len(q)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            h = base.model(input_ids=ids, attention_mask=mask).last_hidden_state
            for r, (i, c) in enumerate(rows):
                L, k = len(heads[i]), len(cands[c])
                lg = base.lm_head(h[r, L - 1 : L - 1 + k]).float()
                lp = torch.log_softmax(lg, -1).gather(-1, ids[r, L : L + k, None]).sum()
                out[s + i, c] = float(lp)
        return out

    def paired_logodds(self, texts: Sequence[str], cand_a: str, cand_b: str,
                       mode: str = "prompt") -> torch.Tensor:
        lp = self.logprob_of_strings(texts, [cand_a, cand_b], mode)
        return lp[:, 0] - lp[:, 1]

    def paired_logodds_templates(self, texts: Sequence[str], pairs: Sequence[tuple[str, str]],
                                 mode: str = "prompt") -> torch.Tensor:
        return torch.stack([self.paired_logodds(texts, a, b, mode) for a, b in pairs])


# ----------------------------------------------------------------------------- Gemini


def _api_key(env_file: str | Path | None = None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        path = Path(env_file) if env_file else Path(__file__).resolve().parents[2] / ".env"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key.startswith("sk-or-"):
        raise RuntimeError("OPENROUTER_API_KEY missing or not an OpenRouter key")
    return key


GEMINI_PROMPT = (
    "{context_label}:\n<text>{text}</text>\n\n{question}\n"
    "Option A: {a}\nOption B: {b}\n\n"
    'Answer with JSON only: {{"p_A": <your probability, between 0 and 1, that option A is '
    'correct>}}.'
)


@dataclass
class TextNLAGemini:
    """Elicited-probability paired LO from Gemini 3.8 Flash (OpenRouter)."""

    model: str = "google/gemini-3.8-flash"
    eps: float = 0.01
    max_workers: int = 8
    timeout: float = 90.0
    retries: int = 4
    env_file: str | None = None
    spend_usd: float = field(default=0.0, init=False)
    n_calls: int = field(default=0, init=False)
    n_failed: int = field(default=0, init=False)
    n_unparsed: int = field(default=0, init=False)

    def _call(self, prompt: str) -> float | None:
        body = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 600, "reasoning": {"effort": "minimal"},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "prob", "strict": True,
                "schema": {"type": "object", "properties": {"p_A": {"type": "number"}},
                           "required": ["p_A"], "additionalProperties": False}}},
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {_api_key(self.env_file)}",
                     "Content-Type": "application/json"})
        for attempt in range(self.retries):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=self.timeout))
                self.n_calls += 1
                self.spend_usd += float(r.get("usage", {}).get("cost") or 0.0)
                txt = r["choices"][0]["message"]["content"] or ""
                m = re.search(r'"p_A"\s*:\s*([0-9.eE+-]+)', txt)
                if m is None:  # e.g. hidden reasoning ate the token budget
                    self.n_unparsed += 1
                    return None
                return min(1.0, max(0.0, float(m.group(1))))
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
                time.sleep(2 ** attempt)
        self.n_failed += 1
        return None

    def _logit(self, p: float | None) -> float:
        if p is None:
            return float("nan")
        q = min(1 - self.eps, max(self.eps, p))
        return math.log(q / (1 - q))

    def paired_logodds(
        self, texts: Sequence[str], cand_a: str, cand_b: str, question: str,
        context_label: str = "Text",
    ) -> torch.Tensor:
        """[N] LO = mean(logit P(a) with a first, -logit P(b) with b first). NaN if both fail."""
        jobs = []
        for t in texts:
            for first, second in ((cand_a, cand_b), (cand_b, cand_a)):
                jobs.append(GEMINI_PROMPT.format(context_label=context_label, text=t,
                                                 question=question, a=first, b=second))
        with ThreadPoolExecutor(self.max_workers) as ex:
            ps = list(ex.map(self._call, jobs))
        out = []
        for i in range(len(texts)):
            ab, ba = self._logit(ps[2 * i]), self._logit(ps[2 * i + 1])
            vals = [v for v in (ab, -ba) if math.isfinite(v)]
            out.append(sum(vals) / len(vals) if vals else float("nan"))
        return torch.tensor(out)
