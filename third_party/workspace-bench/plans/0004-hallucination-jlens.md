# 0004 — Phase 4: hallucination (chat) and jlens_concept_pr

Design: `plans/0000-bench-v2-design.md`. Infra: `plans/0001-scaffold.md`, shared pieces from
`plans/0002-mc-families.md` (`mcjudge.run_calls`, `summarizer`, cache payload conventions,
dry-run conventions, `JudgeArgs.extra` / `aux_models`). Both families are already judged by
Gemini 3.8 Flash in their source form; this PR re-homes them onto the shared client and cache
without changing their instruments. Sources (read-only):

- `H` = `/workspace/camila/hallucination-bench` (public repo, MIT): `src/hallucination_bench/{judge,cli,llm,sites}.py`,
  `data/{items.json,capture_rows.json,judge_prompt_v5c_chat.json}`, `examples/`, `tests/`, `README.md`.
- `W` = `/workspace/camila/global-workspace-clean/.claude/worktrees/jlens-pr`:
  `scripts/oracle_lens_evals/jlens_pr/{judge_prompts,judge_openrouter,judge_pr,score_pr,items}.py`,
  `src/global_workspace/judges/concept_pr.py`, `src/global_workspace/token_types.py`,
  `evals/workspace-bench/jlens_pr/`, and `scripts/oracle_lens_evals/jlens_pr/CLAUDE.md` (spec of record).

## hallucination (`evals/hallucination/`)

Bank = `H/data/items.json` (149 items) **verbatim** plus `H/data/judge_prompt_v5c_chat.json`
(system, user template, schema). `H/data/capture_rows.json` is the exact-token capture record;
the judge never reads it (verified: `H/cli.py` reads `items.json` only) — ship it under
`evals/hallucination/capture_rows.json` for reproducibility, documented as capture-only. Family README = `H/README.md` re-cut to the repo format (what it is, example, how judged,
Sources/Data terms block copied verbatim, Judge prompts section quoting the three prompt fields).

Port rules:
- `src/wsbench/evals/hallucination/prompts.py`: `JUDGE_SYSTEM`, `JUDGE_USER` (template),
  `JUDGE_SCHEMA` loaded from the JSON file at import and ALSO pinned as literals with a test
  asserting equality (the source has this test; keep it). **This family renders `JUDGE_USER`
  with `.format(...)` exactly as the source** (`H/judge.py` L127-177): the template carries
  `{{text, type, why}}` escapes, and the README-equality test compares the escaped literal.
  The 0002 `str.replace` rule does not apply here. `PROMPT_VERSION = "v5c-chat"` (the
  source `PROMPT_VERSION`). `JudgeConfig(model=DEFAULT_JUDGE, prompt_version="v5c-chat",
  reasoning={"effort": "minimal"})` — identical to the source pin, so `pinned_instrument` is
  true under the default.
- `judge.py`: port `H/judge.py` label derivation unchanged (span verification `verify_quote` /
  `normalize_ws`, classes hallucinated / off_topic / consistent / generic / unjudged, unknown
  span type = wrong, revocation across the k readouts of one cell, `HAL_K = 3` first non-empty
  readouts judged). Cells = every row of the readouts file that is on a bank read site
  (`items[].sites[].pos`); rows off-site are `pos_not_selected` (use `load_readouts(positions=…)`).
  Expected grid = every site of every in-scope item × the selected layers (`--layers`, else
  all layers in the file); `n_missing_cells` computed with `missing_cells`; a missing cell is
  fatal (exit 2 via `SystemExit`) unless `--allow-missing` — this family DOES use
  `--allow-missing`, unlike the MC families. One call per cell with the kept samples
  (`[s for s in samples if s.strip()][:HAL_K]`, ≤ 3; `k = 1` for the tokens kind). Tokens kind
  → `decode_token` (source `H/judge.py`, byte-level BPE → text; copy it) runs on every token
  BEFORE `render_bag`, then the shared summarizer per cell (replaces the source's own
  summariser prompt; record the source summariser prompt in the README as legacy), then judged
  as one readout. Cache payload per the 0002 convention, using the new `run_calls(validate=…)`
  hook (see Shared change): **a verdict in which any readout parses to `unjudged` fails
  validation** and is stored as a failure so the next run re-queues the whole cell (source
  `usable()`, `H/cli.py` L761-769; port `test_partly_judged_cell_is_unjudged_and_retried`,
  asserting a second run re-issues exactly those keys).
- `score.py`: `metric = "hallucination_rate"`, `higher_is_better = False`, `value =
  n_hallucinated / n_specific` (revocation NOT applied; `hallucination_rate_revoked` is the
  revoked variant in extras), `n_items` = items in scope (149) with the README noting the value
  is a readout-level ratio, `ci95` = the source `bootstrap_rate_ci` (ratio of per-item sums,
  N_BOOT = 2000, index `int(0.025*(n-1))` — port it verbatim into `score.py`; do NOT use
  `results.bootstrap_ci`), `chance = None`,
  `chance_label = "precision instrument: no chance line; excluded from the macro"`,
  `extras = {hallucination_rate_revoked, assert_share, off_topic_rate, n_specific, n_readouts,
  by_layer, by_site_kind, n_unverified_spans}` (the source `numbers` block, same names).
  `complete` additionally requires `n_empty_cells <= 5% of expected` as the source does
  (already the shared rule). Deviation from the shared rule: **`--layers` does not count as a
  subset for this family** (a single-layer lens judged at its one layer is complete, as in the
  source); `--items`/`--limit` still do.
- `run --all` macro excludes it automatically (metric ≠ pass_rate).
- Not ported: the standalone `hallucination-bench-judge` CLI and its `--layers`-driven grid
  flags beyond what `JudgeArgs` carries; `examples/topk_toy.jsonl` becomes
  `examples/readouts/hallucination.tokens.jsonl`; `examples/readouts_toy.jsonl` becomes
  `examples/readouts/hallucination.jsonl`.
- Tests: port `H/tests/test_judge.py` (label derivation, span verification, revocation) and the
  relevant parts of `test_cli.py` / `test_sites.py` / `test_data.py` onto the wsbench entrypoint;
  prompt-literal equality; grid validation incl. `--allow-missing`; resume; tokens kind; dry-run.

## jlens_concept_pr (`evals/jlens_concept_pr/`)

Bank dir contents (from `W/evals/workspace-bench/jlens_pr/`): `gen-jlens-pr-jlens/` (the
reference J-lens top-10, 299 labels × 12 layer files, 2.1 MB — required), the acts manifest
`acts-jlens-pr/manifest.json` (labels, families, `eval_positions`, tokens — required; ship it
as `evals/jlens_concept_pr/manifest.json`, and `acts-jlens-pr-L42/manifest.json` as
`manifest_L42.json`), `README.md`, `items_manifest.json`, the three gold files (audit
provenance, small). NOT shipped: `items.json` (792 KB capture rows with input_ids) and
`seeds.json` (capture-only) — the README says where they live (source repo) and that they are
needed only to recapture. Total ≈ 4 MB.

Instrument (unchanged from the judge of record `judge_openrouter.py`):
- prompts.py — copy verbatim from `judge_prompts.py`: `STAGE_A_SYSTEM` (L17-30),
  `STAGE_B_SYSTEM` (L32-47), `STAGE_A_SCHEMA` / `STAGE_B_SCHEMA` (L51-73), `GRADE_VALUE`
  (L75), `STAGE_P_SYSTEM` (L182-209 — **the worked-example guide is calibrated against the
  hand-label audit; copy byte-for-byte, never reflow**), `STAGE_P_SCHEMA` (L211-228),
  `STAGE_P_CHUNK = 60` (L229-232); user templates from `stage_a_params` (L110-120:
  `"Text:\n{text}"`), `stage_b_params` (L123-130: `"Token: {token_repr}\n\nConcepts:\n{listing}"`),
  `stage_p_params` (L246-251: `"Tokens: {tok_list}\n\nConcepts:\n{listing}"`, `tok_list` =
  `", ".join(repr(t))`); `concat_samples` (L105-107, `"\n---\n"` join of non-empty stripped
  samples); parsers `parse_stage_a` / `parse_stage_b` / `parse_stage_p` (L137-171, L254-272;
  a `ValueError` = reject, never a partial score). `PROMPT_VERSION = "jlens-pr-v1"`.
- Models: `JudgeConfig(model=DEFAULT_JUDGE, prompt_version="jlens-pr-v1",
  reasoning={"effort": "minimal"}, aux_models={"extract": "deepseek/deepseek-v4-flash"})`.
  Stage A uses `aux_models["extract"]` with reasoning `{"enabled": False}` (build a
  `ResolvedJudge(model=aux, reasoning={"enabled": False}, pinned=judge.pinned, source=judge.source)`
  for that batch) and default temperature; Stage B default temperature; **Stage P temperature
  0.0**; `max_tokens = 16000` on every stage (see Shared change).
- Inputs: an AO arm's readouts file in the contract (prose, k samples per cell; the cell is
  (label, layer, eval position) — exactly one eval position per label per the manifest);
  `text = concat_samples(samples)`. Expected grid = 299 labels × selected layers at
  `pos == eval_positions[0]` (from the manifest); an ABSENT row is `n_missing_cells` (fatal
  without `--allow-missing`, unlike the source which scored it silently); a PRESENT-but-empty
  row is scored as zero-concept ("lens silent"), counted in `n_empty_cells` (so the shared
  5%-empty completeness rule applies, which the source did not have — say so in the README),
  with a `has_text` flag per cell row. The reference tokens for
  the same (label, layer) come from `gen-jlens-pr-jlens/<label>/L{layer:03d}.jsonl` rows
  (`samples` = 10 display strings, `scores` unused); each token is decoded with
  `bpe_display_to_text` (`items.py` L91-122, copy verbatim) and content-filtered with
  `is_content_token` (`concept_pr.py` L213-219, which needs `token_types.classify_token` /
  `CONTENT_TYPES` — copy `token_types.py` and `concept_pr.py` whole into
  `src/wsbench/evals/jlens_concept_pr/{token_types,concept_pr}.py`; `token_types.py` imports
  `global_workspace.glossary.is_cjk` — inline `is_cjk` (L87-90) + `_CJK_RANGES` (L28) from `glossary.py`; `concept_pr.py` uses numpy — **add `numpy>=1.26` to `pyproject.toml` dependencies**
  rather than re-implementing the percentile numerics). Reference tokens are loaded
  unstripped (`strip=False`, as `jlens_tokens` does). Load `manifest.json` for labels /
  families / `eval_positions`; a readouts layer is valid iff
  `gen-jlens-pr-jlens/<label>/L{layer:03d}.jsonl` exists (covers L42); `manifest_L42.json`
  is provenance only.
- Stages and call units (one `run_calls` batch per stage). Keys: the source keys carry an
  `arm` prefix (`arm__label__Lxxx__A`); wsbench has no arm, so keys are `f"{cell.key}:A"`,
  `f"{cell.key}:B{ti:02d}"`, `:F{ti:02d}` (foil), `:P{ci:03d}`, `:Q{ci:03d}` (pfoil) with
  `cell.key` the contract key; the ported key tests are rewritten to these:
  A = one call per cell with text; B = one call per content reference token per cell with
  concepts; P = one call per 60-concept chunk with the full token set. `foil` (Stage B against
  a deranged partner's tokens, `foil_pairing` L125-136 with seed 0 per family group) runs by
  default as in `wsbench_stage.sh`; `pfoil` only with `--opt stage_p_foil=1`.
- Scoring (`concept_pr.py`, unchanged): precision = mean Stage-P support over all concepts;
  recall@10 = mean over content tokens of `expected_max_grade(m=10)` (hypergeometric,
  L222-242); raw_recall; statuses `ok / missing_a / incomplete_b / missing_p` (a `missing_p`
  cell NaNs precision only); bootstrap 1000 resamples seed 0. Layer selection: default = all
  layers in the readouts file; any selected layer without a reference file
  `gen-jlens-pr-jlens/<label>/L{layer:03d}.jsonl` is fatal (`SystemExit(2)` listing the
  offending layers, before any call). Headline layer rule (`adapters/jlens_pr.py` L103-104):
  if exactly one judged layer, that one; else L44; if several layers are judged and L44 is
  not among them, `value=None, ci95=None, n_items=0, extras.headline_layer=None,
  complete=False` with a printed warning.
- FamilyResult: `metric = "precision"`, `value` = mean item precision at the headline layer,
  `ci95` = bootstrap, `n_items` = cells at the headline layer with non-NaN precision (status
  `ok`; a zero-content-token cell is `ok` with NaN and is dropped from the mean, as
  `score_pr._block` / `_mean` do), `chance = None`, `chance_label = "shuffled-partner foil precision (measured, see extras.foil)"`,
  `extras = {headline_layer, recall_at_10, recall_at_10_ci, raw_recall, foil: {recall_at_10,
  precision (None unless pfoil)}, by_layer: {L: block}, mean_n_concepts, punct_frac,
  reject_rate: {stage: {n_requests, n_missing, rate}}, statuses: {...}}`; per-item `passed`
  (precision ≥ 0.5) is display-only in `rows`. `n_unjudged_cells` = cells with status
  `missing_a` / `incomplete_b` / `missing_p` at any judged layer; for this family `complete`
  replaces the shared `n_unjudged == 0` clause with "every stage's reject rate ≤ 5%"
  (`JLENS_PR_MAX_REJECT_RATE` semantics): call `completeness(..., n_unjudged=0, ...)` and AND
  the reject-rate clause, while `counts["n_unjudged_cells"]` still reports the true number.
- Not ported: `judge_pr.py` (legacy GPT route), `judge_modal.py`, `judge_bakeoff.py`,
  `validate_precision_judge.py`, `wsbench_stage.sh` (the `run --all` driver replaces it), the
  bundle adapter.
- README: Sources block = the J-lens paper (*Verbalizable Representations Form a Global
  Workspace in Language Models*, Anthropic / Transformer Circuits, 2026, arXiv:2607.15495),
  `anthropics/jacobian-lens` (Apache-2.0) and the `neuronpedia/jacobian-lens` n1000 wikitext
  artifact the reference tokens come from, plus the four seed corpora (`lmsys/lmsys-chat-1m`,
  `ConvLab/dailydialog` with the same data terms as hallucination, `NeelNanda/pile-10k`,
  `HuggingFaceFW/fineweb-edu` sample-10BT). Judge prompts section: all three system prompts,
  three user templates, three schemas, the chunk rule, the grade values. Cost line: ≈ 35k
  calls per 11-layer arm.
- Tests: port `W/tests/test_jlens_pr_judge_prompts.py`, `test_concept_pr.py`, the
  `bpe_display_to_text` / `foil_pairing` tests from `test_jlens_pr_items.py`, and from
  `test_jlens_pr_score.py` only the `assemble_grids` / `full_grid` / `score_one_cell` /
  `assemble_support` tests (the CLI-module tests do not port; a `_reject_block` test is NEW);
  key construction; stage sequencing with a fake client (A → B/foil → P); reject handling;
  headline-layer rule; resume; dry-run (prints the Stage A prompt only).

## Shared change

`mcjudge.run_calls` gains `validate: Callable[[dict], bool] | None = None`. A landed result
that fails `validate` is stored as `{"result": None, "meta": {**call.meta, "raw": r}}` (so
`Cache.get` skips it and the next run re-queues the key) and returned as `None`. Hallucination
passes "no readout parses to `unjudged`"; jlens passes each stage's parser (a `ValueError` =
reject = invalid), matching the source's resume-by-`done` behaviour (`judge_openrouter.py`
L211). Tests: hallucination partial-verdict retry; jlens "reject handling / resume" asserting
that a second run re-issues exactly the rejected keys.

`llm.stream_json_async` / `stream_json`: add `temperature: float | None = None`. OpenRouter
route: passed as the `temperature=` kwarg. **Anthropic route: dropped with one printed warning
per batch** (newer Claude models return 400 on `temperature`; the source refuses to send it,
`judge_pr.py` L358-363) — test both. `max_tokens` may be overridden per batch (already a
kwarg). `mcjudge.run_calls` gains `temperature` and `max_tokens` pass-through kwargs and
accepts a per-batch `judge: ResolvedJudge` (so a stage can run on an aux model with its own
reasoning, e.g. Stage A = DeepSeek with `{"enabled": False}`); the fingerprint becomes
`fingerprint(prompt_version, judge.model, judge.reasoning, temperature, system, user)` — this
also changes the 0002 fingerprint (temperature `None` for the MC families), which is fine
because no cache predates this PR.

## Acceptance

ruff + pytest green; `wsbench list` shows `hallucination` (149, hallucination_rate, lower is
better) and `jlens_concept_pr` (299, precision); dry-run prints a prompt for each example file;
ported source tests pass unchanged in semantics.
