# mats fork of workspace-bench

Upstream: github.com/camilablank/workspace-bench at the commit in `UPSTREAM_COMMIT`.

## Changes
- **Fireworks judge route** (`src/wsbench/llm.py`): model ids starting with `accounts/fireworks/`
  go to `https://api.fireworks.ai/inference/v1` (OpenAI-compatible, json_schema response_format,
  no OpenRouter `extra_body`). Key from `FIREWORKS_API_KEY`, falling back to `FIREWORKS_API`.
  Fireworks reports tokens, not cost, so `spend.usd` stays 0 on this route. Tests:
  `tests/test_llm_fireworks.py`. Live preflight passed 2026-10-05 for deepseek-v4p1-flash,
  kimi-k3, glm-5p3-flash, gpt-oss-120b.

## Usage
    set -a; source ../../.env; set +a
    uv run wsbench judge family=F readouts=R out=O judge_model=accounts/fireworks/models/kimi-k3
    # or: export WSBENCH_JUDGE_MODEL=accounts/fireworks/models/kimi-k3

## Caveat
Published WorkspaceBench numbers use Gemini 3.8 Flash (OpenRouter). A swapped judge is NOT
comparable without an agreement check (see `plans/0007-judge-swap.md`). Avoid Qwen-family
judges: the model under study is Qwen3.6-27B.
