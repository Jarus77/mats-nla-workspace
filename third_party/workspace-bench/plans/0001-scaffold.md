# 0001 — Phase 1: scaffold + shared infrastructure

Design: `plans/0000-bench-v2-design.md` §1. This plan is the exact spec for the first PR.
No eval family is ported here; the deliverable is the package every family will plug into.
Reference code to port from (read-only, outside this repo):

- `/workspace/camila/hallucination-bench/src/hallucination_bench/{llm.py,cli.py}` — the
  OpenRouter client (pacer, backoff, preflight, spend) and the append-only JSONL cache.
- `/workspace/camila/global-workspace-clean/src/global_workspace/judges/llm_client.py`
  `_one_claude` — the Anthropic structured-output call (system block with `cache_control`,
  `output_config.format.json_schema`, `stop_reason in (refusal, max_tokens)` → `None`).
- `/workspace/camila/global-workspace-clean/scripts/oracle_lens_evals/olens_sglang/common.py`
  `load_unit_rows` — the in-house gen-dir layout the converter reads.

## Deliverable tree

```
workspace-bench/
├── pyproject.toml               # package `wsbench`, script `wsbench`, uv-managed
├── uv.lock
├── .gitignore                   # .venv/ outputs/ .env __pycache__/ .pytest_cache/ .ruff_cache/ *.egg-info/
├── .github/workflows/ci.yml     # uv sync --extra dev; ruff check; ruff format --check; pytest -q
├── CLAUDE.md                    # agent runbook (contracts + invariants + how to add a family)
├── README.md                    # keep the stub; phase 6 writes the real one
├── examples/readouts/           # empty dir with .gitkeep (families add toy files)
├── src/wsbench/
│   ├── __init__.py              # __version__ = "0.1.0"
│   ├── llm.py
│   ├── judge_config.py
│   ├── cache.py
│   ├── readouts.py
│   ├── results.py
│   ├── registry.py
│   ├── cli.py
│   └── evals/__init__.py        # empty package; families land here in later phases
└── tests/
    ├── conftest.py
    ├── test_llm.py
    ├── test_judge_config.py
    ├── test_cache.py
    ├── test_readouts.py
    ├── test_results.py
    ├── test_registry.py
    └── test_cli.py
```

## pyproject.toml

```toml
[project]
name = "workspace-bench"
version = "0.1.0"
description = "workspace-bench: nine evals of whether an activation-reading lens surfaces what a model computes but never writes"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = ["openai>=1.40", "anthropic>=1.0"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
wsbench = "wsbench.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/wsbench"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "C4", "RUF"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Both SDKs are imported lazily inside the functions that need them, so `import wsbench` and
the offline tests never touch them.

## `llm.py` — one structured-JSON client, two routes

Public surface (everything else private):

```python
OPENROUTER = "https://openrouter.ai/api/v1"
class JudgeConfigError(RuntimeError): ...
@dataclass
class Spend: usd: float = 0.0; calls: int = 0; retries: int = 0; errors: int = 0; refusals: int = 0; input_tokens: int = 0; output_tokens: int = 0
    def report(self) -> str      # "calls=… retries=… errors=… refusals=… spend=$…" plus " in_tok=… out_tok=…" only when nonzero
def route(model: str) -> Literal["anthropic", "openrouter"]     # "anthropic" iff model.startswith("claude-")
def api_key(model: str) -> str                                   # ANTHROPIC_API_KEY for anthropic; OPENROUTER_API_KEY (must start "sk-or-") otherwise; JudgeConfigError if missing/malformed
def schema_block(name: str, properties: dict, required: list[str]) -> dict   # {"name", "strict": True, "schema": {type object, additionalProperties False, required, properties}}
async def stream_json_async(prompts: list[tuple[str, str]], *, schema: dict, model: str, on_result: Callable[[int, dict | None], None],
                reasoning: dict | None = None, concurrency: int = 64, rpm: float = 240.0,
                max_tokens: int | None = None, timeout: float = 180.0, spend: Spend | None = None) -> Spend   # the primitive; max_tokens None → 8000 on OpenRouter, 16000 on Anthropic (thinking counts against it)
def stream_json(...same signature...) -> Spend                   # = asyncio.run(stream_json_async(...)); for CLI/tests
def preflight(model: str, reasoning: dict | None) -> None
# test seams (module-level, monkeypatchable):
def _make_client(route: str, key: str) -> Any                    # AsyncOpenAI / AsyncAnthropic construction lives ONLY here
def _backoff(attempt: int, status: int | None) -> float          # min(90 if status == 429 else 30, 2 * 2**attempt) * (0.5 + random())
async def _pace(rpm: float) -> None                              # see pacer below
_PACER: _Pacer                                                   # module singleton; _Pacer.reset() clears the next-slot time (tests)
async def preflight_async(model: str, reasoning: dict | None) -> None   # preflight = asyncio.run(preflight_async(...))
```

Behaviour, both routes:
- `(system, user)` strings are sanitised with `.encode("utf-8", "replace").decode("utf-8")`
  before every attempt (lone surrogates from detokenised readouts).
- Process-wide RPM pacer: a module-level `_Pacer` holding the monotonic next-slot time behind a
  `threading.Lock` (so it is shared by every `Spend`, every event loop and every thread, unlike
  hallucination-bench's per-`Spend` slot). `_pace(rpm)` reserves the next slot under the lock,
  then `await asyncio.sleep` outside it. `stream_json_async` is the primitive so phase 6 can run
  the nine families as coroutines in ONE loop under one pacer; `stream_json` only wraps it in
  `asyncio.run` and raises `JudgeConfigError` if called from inside a running loop.
- Retry loop: 12 attempts; transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or
  exception name containing "Timeout" / "Connection" / "Deadline" or a JSON decode failure or `ValueError`
  (empty choices / non-object body). Backoff `min(cap, 2 * 2**attempt) * (0.5 + random())`,
  cap 90 s on 429 else 30 s (`_backoff`). Fatal = exception name in ("AuthenticationError",
  "PermissionDeniedError", "NotFoundError") or status in (401, 402, 403) → raise
  `JudgeConfigError`. Any other failure after retries → `spend.errors += 1`, return `None`.
- `on_result(i, parsed_or_None)` fires as each call lands (as_completed); tasks cancelled and
  the client closed on any exception.
- `concurrency < 1` or `rpm <= 0` → `JudgeConfigError`. Empty `prompts` → return without
  building a client.

OpenRouter route (`route == "openrouter"`): `AsyncOpenAI(api_key, base_url=OPENROUTER,
max_retries=0)`; `chat.completions.create(timeout=timeout, model, max_tokens, messages=[system, user],
response_format={"type": "json_schema", "json_schema": schema}, extra_body={"reasoning":
reasoning, "usage": {"include": True}, "provider": {"require_parameters": True}})`. `reasoning`
key is omitted from `extra_body` when `None`. Strip a ```` ```json ```` fence before
`json.loads`. Tally `usage.cost` (or `usage.model_extra["cost"]`) into `spend.usd`.

Anthropic route (`route == "anthropic"`): `AsyncAnthropic(api_key, max_retries=0)`;
`messages.create(timeout=timeout, model, max_tokens, system=[{"type": "text", "text": system, "cache_control":
{"type": "ephemeral"}}], messages=[{"role": "user", "content": user}],
output_config={"format": {"type": "json_schema", "schema": schema["schema"]}})`. `reasoning` is
ignored on this route. `stop_reason in ("refusal", "max_tokens")` → `spend.refusals += 1`
(refusal) or `spend.errors += 1` (max_tokens), return `None`, no retry. Parse the first
`text` content block. `usage.input_tokens` / `output_tokens` are added to `spend.input_tokens`
/ `spend.output_tokens`; `spend.usd` is unchanged on this route (no price table).

`preflight`: one call with schema `{"a": integer}` and prompts `("Reply JSON.", 'Return {"a":1}')`
at concurrency 1; anything but `{"a": 1}` → `JudgeConfigError`.

## `judge_config.py`

```python
DEFAULT_JUDGE = "google/gemini-3.8-flash"
DEFAULT_REASONING = {"effort": "minimal"}      # Gemini cannot turn reasoning off
ENV_OVERRIDE = "WSBENCH_JUDGE_MODEL"

@dataclass(frozen=True)
class JudgeConfig:
    model: str = DEFAULT_JUDGE
    prompt_version: str = "v1"
    reasoning: dict | None = None            # None → DEFAULT_REASONING if route is openrouter, else None
    aux_models: Mapping[str, str] = field(default_factory=dict)   # e.g. {"summarizer": DEFAULT_JUDGE}

@dataclass(frozen=True)
class ResolvedJudge:
    model: str
    reasoning: dict | None
    pinned: bool            # True iff model == config.model (no override took effect)
    source: Literal["flag", "env", "family"]

def resolve(config: JudgeConfig, *, flag: str | None = None, env: Mapping[str, str] | None = None) -> ResolvedJudge
```

Precedence: `flag` > `env[ENV_OVERRIDE]` (non-empty) > `config.model`. `reasoning` is
`config.reasoning` if set, else `DEFAULT_REASONING` when the resolved model routes to
OpenRouter, else `None`. `pinned` is True only when the resolved model equals `config.model`
(an override that names the same model still counts as pinned).

## `cache.py` — append-only JSONL verdict cache

```python
def fingerprint(*parts: Any) -> str                # sha256 of json.dumps(parts, sort_keys=True, ensure_ascii=False)[:16]
class Cache:
    def __init__(self, path: Path): ...            # reads existing rows; torn last line tolerated (hallucination-bench read_cache/open_append semantics)
    def get(self, key: str, fp: str, *, include_failed: bool = False) -> dict | None
        # newest row with this key AND this fp whose payload["result"] is not None; with include_failed, the newest such row regardless
    def put(self, key: str, fp: str, payload: dict) -> None   # appends {"key", "fp", "ts", **payload}; flushes; raises ValueError if payload uses the reserved keys key/fp/ts
    def __len__(self) -> int                       # rows loaded at open + rows appended this session
    def close(self) -> None
```

Rows are never rewritten; a changed fingerprint simply appends a new row, and older rows stay
so switching back reuses them. A row is *failed* iff `row.get("result") is None` (a payload without a `result` key counts as failed);
`get` skips it unless `include_failed=True`, so failed cells are retried on the next run.

## `readouts.py` — the one input contract + converter

```python
@dataclass(frozen=True)
class Cell:
    id: str; layer: int; pos: int
    samples: tuple[str, ...] | None      # prose lens
    tokens: tuple[str, ...] | None       # top-k token lens (vocab strings, best first)
    scores: tuple[float, ...] | None
    @property
    def kind(self) -> Literal["prose", "tokens"]
    @property
    def key(self) -> str                 # f"{id}__L{layer:03d}__p{pos}"

@dataclass
class LoadReport:
    kind: Literal["prose", "tokens"] | None
    n_rows: int
    skipped: dict[str, int]              # malformed, duplicate, mixed_kind, unknown_id, layer_not_selected, pos_not_selected
    n_empty: int                         # kept cells whose samples/tokens list is empty (or all-whitespace samples)
    layers: list[int]                    # sorted layers present after filtering

def load_readouts(path: Path, *, ids: Collection[str] | None = None, layers: Collection[int] | None = None,
                  positions: Mapping[str, Collection[int]] | None = None) -> tuple[list[Cell], LoadReport]
def expected_cells(positions: Mapping[str, Collection[int]], layers: Collection[int]) -> set[tuple[str, int, int]]
def missing_cells(cells: Iterable[Cell], expected: set[tuple[str, int, int]]) -> list[tuple[str, int, int]]
def convert_gen_dir(gen_dir: Path, out: Path, *, kind: Literal["prose", "tokens"], layers: Collection[int] | None = None) -> LoadReport
```

Loader rules: a row must be a JSON object with string `id`, int `layer`, int `pos`, and exactly
one of `samples` (list of str) / `tokens` (list of str, optional parallel `scores` list of
numbers of the same length); otherwise `malformed`. A file must be all-prose or all-tokens; a row of the other kind
is `mixed_kind`. `(id, layer, pos)` duplicates keep the first, count `duplicate`. Filters count
`unknown_id` / `layer_not_selected` / `pos_not_selected`. Malformed JSON lines are `malformed`,
never fatal. Empty samples are kept and counted in `n_empty` (an empty readout is a result,
not a missing cell; `missing_cells` treats them as present). `unknown_id` fires when `ids` is
given and the id is not in it, else when `positions` is given and the id is not a key of it.

Converter: reads `<gen_dir>/<label>/L<layer:03d>.jsonl` (rows `{"label", "layer", "pos",
"samples", ...}`, optional `"scores"`), writes one contract row per source row with
`id` = the directory name (not the row's `label`, which older dirs may lack); `kind="tokens"`
maps `samples`→`tokens` and `scores`→`scores`. Layer files are discovered by glob
`*/L*.jsonl`; the layer comes from the filename (a row's own `layer` field is ignored); rows
missing `pos` or `samples` are skipped and counted in the returned report's
`skipped["malformed"]`; `layers` restricts. Samples are written raw: no `extract_phrase` stripping
(fresh gen dirs are already stripped at write time). Returns the LoadReport of the written file.

## `results.py` — one results schema

```python
SCHEMA_VERSION = 1
@dataclass
class FamilyResult:
    family: str
    metric: str                          # e.g. "pass_rate", "hallucination_rate", "precision"
    value: float | None                  # None when nothing was judged
    ci95: tuple[float, float] | None
    n_items: int
    higher_is_better: bool
    chance: float | None
    chance_label: str | None
    complete: bool
    pinned_instrument: bool
    config: dict                         # judge_model, prompt_version, reasoning, layers, ...
    counts: dict                         # n_expected_cells, n_missing_cells, n_unjudged_cells, n_empty_cells, skipped_rows, spend_usd
    extras: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    def to_json(self) -> dict
        # exactly: {"schema_version", "family", "complete", "pinned_instrument", "config", "n_items",
        #   "counts": {"n_expected_cells", "n_missing_cells", "n_unjudged_cells", "n_empty_cells", "skipped_rows", "spend_usd"},
        #   "numbers": {"metric", "value", "ci95", "chance", "chance_label", "higher_is_better", "extras"},
        #   "rows": [...]}
    @classmethod
    def from_json(cls, d: dict) -> "FamilyResult"   # raises ValueError on schema_version != SCHEMA_VERSION; ci95 list → tuple

def bootstrap_ci(values: Sequence[float], *, n_resamples: int = 1000, seed: int = 0) -> tuple[float, float]   # percentile 2.5/97.5 of resampled means; (v, v) for a single value; raises on empty
def write_results(out_dir: Path, result: FamilyResult) -> Path        # <out_dir>/results.json, indent=1, ensure_ascii=False
def read_results(out_dir: Path) -> FamilyResult
def completeness(*, pinned: bool, subset: bool, n_expected: int, n_missing: int, n_unjudged: int, n_empty: int) -> bool
    # pinned and not subset and n_missing == 0 and n_unjudged == 0 and n_empty <= 0.05 * n_expected
def macro(results: Sequence[FamilyResult]) -> dict
    # {"value": mean of value over results with complete, metric == "pass_rate" and value is not None; "families": [...], "excluded": [{family, reason}]} — reasons: "incomplete", "metric", "no value"
def markdown_table(results: Sequence[FamilyResult], macro_row: dict | None) -> str
    # columns: family | metric | value | 95% CI | n | chance | judge | pinned | complete
```

## `registry.py`

```python
@dataclass(frozen=True)
class EvalSpec:
    name: str                            # family key, e.g. "moral_rationale"
    title: str                           # README name, e.g. "Moral rationale"
    group: str                           # "safety" | "association" | "bag_of_words" | "precision" | "logic"
    bank: Path                           # evals/<family>/items.json, relative to REPO_ROOT
    judge: JudgeConfig
    metric: str
    higher_is_better: bool
    run: Callable[[JudgeArgs], FamilyResult]   # the family's judge entrypoint

@dataclass
class JudgeArgs:                          # what cli.judge passes to every family
    readouts: Path; out: Path; judge: ResolvedJudge; layers: list[int] | None; items: list[str] | None
    limit: int; allow_missing: bool; concurrency: int; rpm: float; dry_run: bool    # limit == 0 → no limit

REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES: dict[str, EvalSpec] = {}       # families register themselves by importing wsbench.evals.<family>
def register(spec: EvalSpec) -> EvalSpec  # raises on duplicate name
def get(name: str) -> EvalSpec            # KeyError with the sorted list of known names in the message
def load_all() -> None                    # imports every wsbench.evals.<pkg> so FAMILIES is populated (pkgutil.iter_modules); idempotent (a second call is a no-op because the modules are already imported)
```

## `cli.py`

`wsbench` with subcommands (argparse, `main(argv=None) -> int`):

`main` calls `registry.load_all()` once before dispatching.

- `list` — table: family, group, n items, judge model, prompt version. n items =
  `len(d)` if the bank JSON's top level is a list, else `len(d["items"])`; `?` if the file is
  absent or any other shape. Empty registry prints "no families registered yet".
- `judge <family> --readouts F --out DIR [--judge-model M] [--layers 20,36] [--items a,b]
  [--limit N] [--allow-missing] [--concurrency 64] [--rpm 240] [--dry-run]` — resolves the judge
  (`judge_config.resolve` with flag + `os.environ`), builds `JudgeArgs`, calls `spec.run`,
  prints `Spend.report()`-style summary and the `value`. Unknown family → message listing known
  families, exit 2. `JudgeConfigError` → message, exit 3. `--out` default
  `outputs/<readouts stem>/<family>/` (so several individually judged families share a
  parent that `report` can read). **`spec.run` returns a `FamilyResult` and writes nothing;
  the CLI calls `write_results(out, result)` for both `judge` and `run`.** Families set
  `pinned_instrument=args.judge.pinned` and `complete=completeness(pinned=args.judge.pinned,
  subset=args.items is not None or args.limit > 0, ...)`; the test stub does the same.
  `judge` prints `result.counts["spend_usd"]` and `result.value`; no `Spend` object crosses
  the `spec.run` boundary.
- `run (--all | --families a,b) --readouts-root DIR --out DIR [same judge flags]` — for each
  selected family expects `DIR/<family>.jsonl`; missing file → recorded as skipped, not fatal;
  runs families sequentially in phase 1 (concurrent scheduling is phase 6); writes
  `DIR_out/<family>/results.json` each and finally `DIR_out/summary.md` via `markdown_table`.
- `report DIR` — reads every `DIR/*/results.json`, prints `markdown_table` + macro, writes
  `DIR/summary.md`.
- `convert-gen-dir GEN_DIR --out F.jsonl --kind prose|tokens [--layers ...]`.

`--dry-run` is passed through to the family; phase 1 only guarantees the flag exists.

## `CLAUDE.md` (repo runbook, ≤ 80 lines)

Sections: what the repo is (one paragraph); the readout contract (the two JSON row shapes);
the judge layer (default, override precedence, pinned semantics, routes, env vars); the
results contract (`results.json` keys, `complete` rule); invariants (prompts live in
`prompts.py` and verbatim in the family README with a test asserting equality; bump
`prompt_version` on any prompt edit; failures never score; frozen banks never edited); how to
add a family (create `src/wsbench/evals/<family>/{prompts,judge,score}.py`, register an
`EvalSpec`, add `evals/<family>/{items.json,README.md}`, `examples/readouts/<family>.jsonl`,
`tests/test_<family>.py`); workflow (`uv sync --extra dev`, `uv run pytest -q`, `uv run ruff
check`, keys `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY`, never committed).

## Tests (write first; all offline)

`conftest.py`: an autouse fixture that snapshots `registry.FAMILIES` before each test and
restores it after, so a stub family registered by one test never leaks into another (tests
must not clear-and-reload; `load_all` is a no-op after first import). Tests that call
`stream_json` pass `rpm=1e9` or monkeypatch `wsbench.llm._pace`, and monkeypatch
`wsbench.llm._backoff` to return 0 and `wsbench.llm._make_client` to return a fake. No async
test plugin: tests drive coroutines with `asyncio.run(...)` inside sync test functions. A second
autouse fixture calls `wsbench.llm._PACER.reset()` before each test. The "pacer shared across
two `Spend`s" test asserts on reserved slot times with `time.monotonic` / `asyncio.sleep`
monkeypatched, never on wall-clock.

- `test_llm.py`: `route` for `claude-sonnet-5` / `google/gemini-3.8-flash`; `api_key` errors on
  missing and malformed keys; `schema_block` shape; `stream_json` with a fake OpenRouter client
  (via the `_make_client` seam): success, fenced JSON, transient 429 then success (retries
  counted, `_backoff` patched to 0), fatal 401 raises, `stream_json` from inside a running loop
  raises `JudgeConfigError` while `stream_json_async` works, pacer shared across two `Spend`s, non-object body
  after retries → `None` + errors, `on_result` order-independent, lone-surrogate prompt
  sanitised; Anthropic fake: success, `stop_reason="refusal"` → `None` + `refusals == 1`,
  `reasoning` not sent; `preflight` pass/fail; empty prompts build no client; bad concurrency.
- `test_judge_config.py`: precedence flag > env > family; `pinned` semantics incl. override
  equal to pin; reasoning defaulting per route.
- `test_cache.py`: put/get round trip; fingerprint mismatch → None; failed row skipped unless
  `include_failed`; torn last line tolerated; foreign lines skipped; resume across instances.
- `test_readouts.py`: prose + tokens files load; every `skipped` category triggers; mixed kind;
  duplicates; `expected_cells` / `missing_cells`; converter round-trips a temp gen dir for both
  kinds and restricts layers.
- `test_results.py`: `to_json`/`from_json` round trip; `bootstrap_ci` deterministic under seed,
  single value, empty raises; `completeness` each clause; `macro` excludes incomplete and
  non-pass-rate metrics with reasons; `markdown_table` has one row per result + macro row.
- `test_registry.py`: register/duplicate/get error message; `load_all` on the empty package.
- `test_cli.py`: `list` with empty registry; `judge unknown` exit 2; a stub family registered
  in the test whose `run` writes a `FamilyResult` → `judge` writes `results.json` and
  `--judge-model` flips `pinned_instrument`; `run --families stub` with a missing readouts
  file is skipped, present file is processed, `summary.md` written; `report` on that dir;
  `convert-gen-dir` end to end.

## CI

`.github/workflows/ci.yml`: on push + pull_request; ubuntu-latest; `astral-sh/setup-uv@v5`;
`uv python install 3.12`; `uv sync --extra dev`; `uv run ruff check .`; `uv run ruff format
--check .`; `uv run pytest -q`.

## Acceptance

`uv sync --extra dev && uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
green locally; `uv run wsbench list` prints the empty-registry line; `uv run wsbench judge
nope --readouts x --out y` exits 2 with the known-family list. No network access in tests.
