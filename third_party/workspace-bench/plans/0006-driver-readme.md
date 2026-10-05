# 0006 — Phase 6: `run --all`, `report`, top-level README, credits files

Design: `plans/0000-bench-v2-design.md` §1.4, §2, §5. Infra: 0001–0005 merged (nine families
registered). This PR makes the repo usable end to end and writes the public-facing docs.

## 1. `wsbench run --all` (cli.py, new `runner.py`)

Today `cmd_run` judges families sequentially by calling `spec.run` in a loop. Replace the loop
body with `runner.run_families(specs, args) -> list[FamilyResult]`:

- **Concurrency across families, one pacer.** Each family's `run` is synchronous and calls
  `llm.stream_json` (which uses `asyncio.run`). Run families in a `ThreadPoolExecutor` with
  `max_workers = min(len(specs), --family-workers (default 3))`; the process-wide `_PACER` is
  thread-safe (phase 1) so the RPM budget is shared, and each thread's `asyncio.run` has its
  own loop. Per-family `--concurrency` still bounds in-flight calls inside a family. Under
  `--dry-run` `max_workers` is forced to 1 so the printed prompts do not interleave.
  (`stream_json_async` in one loop was the design's stated route; threads reach the same
  outcome with zero changes to the families and are the simplest correct thing.)
- **Fail-soft per family.** `--opt` is parsed and validated once before any thread starts (a
  bad option is one error, exit 2, not nine). A family whose `run` raises `JudgeConfigError`
  or `SystemExit` (missing cells) is recorded as `{"family", "error"}` — for `SystemExit` the
  error text is `e.code` if it is a string else `f"exit {e.code}"` — in `summary.md`'s
  "skipped / failed" list and does not stop the others; exit code is 3 if any family failed
  with `JudgeConfigError`, 2 if any failed with `SystemExit`, else 0. `KeyboardInterrupt` in
  the main thread calls `executor.shutdown(cancel_futures=True)`; in-flight families finish
  their current batch (documented in CLAUDE.md).
- **Resume.** Unchanged: each family's `cells.jsonl` cache under `<out>/<family>/`.
- **Manifest.** `run` writes `<out>/run.json`: `{started, finished, families: {name: {status,
  results_path, value, complete, pinned_instrument, spend_usd, n_calls?}}, judge_overrides:
  {flag, env}, readouts_root}`.
- **Preflight once per model.** `mcjudge.Preflighter` currently keeps a per-instance `_done`
  set; move it to a module-level, lock-guarded `mcjudge._PREFLIGHTED: set[str]` keyed exactly
  as today (`json.dumps([model, reasoning])`) so every family's `Preflighter` shares it.
  Before starting threads (and not under `--dry-run`) the runner preflights each distinct
  resolved judge model (`resolve(spec.judge, …)`, NOT aux models — those are preflighted
  lazily, once, inside the family) and seeds that set; a failing preflight — or a
  `JudgeConfigError` from `resolve` on a bad override — aborts the whole run with exit 3 before
  any thread starts (nothing was spent). Both the runner and `Preflighter.ensure` call
  `llm.preflight` through the module attribute so one monkeypatch counts both.
  `KeyboardInterrupt` → exit 130 after writing `run.json` / `summary.md` for the families that
  finished.
- `--families a,b` and `--all` as today; `--readouts-root DIR` expects `DIR/<family>.jsonl`;
  missing file → skipped (recorded).

## 2. `wsbench report DIR` (results.py `markdown_table`, cli.py)

- Table columns: family | metric | value | 95% CI | n | chance | judge | pinned | complete.
  Values formatted to 3 decimals; `hallucination_rate` rows carry "(lower is better)" in the
  metric cell; the macro row keeps its in-row `excluded: …` text (as today), and a footnote
  list below the table repeats each exclusion with its reason (tests assert both).
- Macro row: mean of complete `pass_rate` families (six of the nine: user_modeling,
  jailbreak_recognition, conjunctive_association, role_bound_association, relational_multihop,
  moral_rationale — jailbreak is judged by Sonnet 5, the others by Gemini). `hallucination` (rate), `jlens_concept_pr` (precision) and
  `agentic_misalignment` (design_score) are always listed as "not in macro: metric".
- Add `--json`: prints `{"families": [r.to_json() …], "macro": macro(results)}` to stdout
  instead of the table; `summary.md` is still written. Add `n_items_without_readouts` to every family's
  `extras` (phase-2 review finding: `complete` cannot distinguish a full run from a truncated
  readouts file for families with no position list). Shared helper
  `mcjudge.items_without_readouts(scope: list[dict], cells: list[Cell]) -> int` = in-scope ids
  with zero rows in the readouts file; called in each family's `judge.py` (where `cells` is
  available) and passed into `score()` as an extra. Grid families (hallucination, jailbreak,
  jlens) already surface an absent item as `n_missing_cells`; they compute the same extra
  anyway so `report` renders one `n` cell uniformly: `n (k no readouts)` when k > 0, read from
  `extras` in `markdown_table`.

## 3. `wsbench list` — add columns `metric`, `calls/arm (approx)`, `credit`

`EvalSpec` gains `calls_per_arm: str = ""` (a human string, e.g. "≈ 7k") and `sources: str = ""`
(a short credit substring that must appear verbatim in README §Credits, e.g. "Choi et al.
2025") — defaults so the concurrently landing phase 3–5 registrations keep working; this PR
sets values for all nine families (it lands after 3–5 merge).

## 4. Top-level README.md (replaces the stub)

Sections, in order:

1. Title + one paragraph: what workspace-bench measures (a lens surfacing what Qwen3.6-27B
   computes but never writes), the two lens kinds (prose "O-lens" samples, J-lens top-10 token
   bags), nine evals in five groups, one judge layer, Gemini 3.8 Flash default.
2. **Quickstart** (copy-paste): clone, `uv sync --extra dev`, `uv run pytest -q`, a dry run
   on an example file with no key, a real `judge` on one family, `run --all`, `report`.
   Readout contract shown as the two JSON rows + the gen-dir converter command.
3. **The nine evals** — for each, in the exact format Camila asked for: one bold name line,
   then three labelled lines: *What it is* (one sentence/phrase), *Example* (question →
   target, taken verbatim from the design doc §2 drafts, checked against the vendored bank),
   *Judged by* (one sentence/phrase). Grouped under: Safety (agentic_misalignment,
   jailbreak_recognition) · Association (user_modeling, conjunctive_association,
   role_bound_association) · Bag of words (relational_multihop) · Precision (hallucination,
   jlens_concept_pr) · Logical processing (moral_rationale). Each entry links to
   `evals/<family>/README.md`.
4. **Judges** — a table: family | judge model | prompt version | why (default / pin reason).
   Three pins: agentic_misalignment → claude-sonnet-5 (judge of record, no Gemini agreement
   data); jailbreak_recognition → claude-sonnet-5 (Gemini 3.8 Flash refuses a share of
   jailbreak cells); jlens_concept_pr Stage A → deepseek-v4-flash (frozen concept lists).
   Override precedence and the `pinned_instrument` rule in three lines. A note that five
   families moved from claude-opus-5 to Gemini in this repo and that phase 7 records the
   agreement.
5. **Results contract** — `results.json` keys, `complete`, the macro rule; cost line
   (≈ 120k calls per arm over the nine, ~8 h at 240 rpm; per-family from `wsbench list`).
6. **Credits** — one bullet per external source, from the §5 table of the design doc plus
   DeepSeek V4 Flash (the jlens Stage A model; amend §5 accordingly), no bullet for in-house
   families. Licence names come from the family README Sources blocks landed in phases 3–4;
   Qwen falls back to "see the model card": Qwen3.6-27B; Gemini 3.8 Flash via OpenRouter, Claude
   Sonnet 5, DeepSeek V4 Flash; Lynch et al. 2025 + `anthropic-experimental/agentic-misalignment`
   (MIT); WildChat (Zhao et al. 2024, ODC-BY) + CHIVE (Karvonen et al.); Transluce (Choi et al.
   2025); J-lens paper (arXiv:2607.15495) + `anthropics/jacobian-lens` (Apache-2.0) +
   `neuronpedia/jacobian-lens`; LMSYS-Chat-1M (license agreement), DailyDialog (CC BY-NC-SA
   4.0), pile-10k, fineweb-edu (ODC-BY); CC-CEDICT only if any vendored code uses it (check
   `token_types.py` / `concept_pr.py`; if not, omit).
7. **License** — MIT for code and in-house items; third-party terms in `NOTICE.md`.

## 5. `CITATION.cff`, `NOTICE.md`

- `CITATION.cff`: cff-version 1.2.0, title "workspace-bench", authors: Camila Blank, year
  2026, repository-code URL, license MIT, `preferred-citation` left as the repo itself (no
  paper yet).
- `NOTICE.md`: third-party data and code terms, one section per source with the exact
  license name, URL, what is included, and (for LMSYS / DailyDialog) the non-commercial /
  agreement clauses restated; Apache-2.0 notice text for any vendored jacobian-lens code.

## 6. CLAUDE.md

Trim to ≤ 100 lines: fold the phase-2 "MC families" section into a generic "Family
conventions" section; add `run --all` threading + preflight note, `run.json`, the README
entry format rule (what / example / judged-by), and "credits live in README §Credits and
NOTICE.md; in-house families get no credit line".

## Tests

`conftest.py`: the autouse fixture also clears `mcjudge._PREFLIGHTED` (process-global state).
`test_runner.py`: three stub families (one raises `JudgeConfigError`, one `SystemExit`, one
succeeds) → correct per-family statuses, exit code, `run.json`, `summary.md`; preflight
called once per distinct model (monkeypatch `wsbench.llm.preflight`); families run
concurrently: stubs meet at a `threading.Barrier(3, timeout=5)` with `--family-workers 3`
pinned, `BrokenBarrierError` being the failure signal.
`test_report.py`: table formatting incl. lower-is-better and not-in-macro footnotes; `--json`.
`test_readme.py`: the top-level README lists all nine registered families by title with the
three labelled lines each, and every `evals/<family>/README.md` is linked; every family's non-empty
`sources` substring appears verbatim in README §Credits; `CITATION.cff` parses (yaml-free: check required
keys by regex); every non-empty `spec.sources` substring also appears in `NOTICE.md`.
`test_list.py`: the new columns render for all nine.

## Acceptance

ruff + pytest green; `wsbench run --all --readouts-root examples/readouts --out /tmp/x
--dry-run` completes for all nine (examples are named `<family>.jsonl`) and writes
`summary.md` + `run.json`; README renders the nine entries; CI green.
