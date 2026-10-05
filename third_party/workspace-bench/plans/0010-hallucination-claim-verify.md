# 0010 — hallucination: count unverifiable claims (second-stage verify pass) (2026-09-17)

Camila: "for hallucination bench, can we make a tiny modification to the judge where we keep
track of the number of unfalsifiable claims as well, like we did with the second round of
judging? just to keep this as a metric".

The "second round" is the source repo's `chat_verify.py` (`v5c-chat-verify-v1`, 2026-09-16,
`docs/project/experiments/oracle_lens/hallucination_claim_level.md`): the judge is GIVEN the
v5c wrong spans as established-false and enumerates every other specific claim as `true` /
`unverifiable` / `disputed` / `off_topic`. It was designed as a separate pass on purpose — a
first attempt that let one enumerate-and-classify prompt also decide falsity agreed with v5c
at κ ≈ 0.44 (the prediction rule drifted). So the "tiny modification" is a second stage, not
an edit to the v5c prompt.

## Invariants restated (the headline is untouched)

- `JUDGE_SYSTEM` / `JUDGE_USER` / `JUDGE_SCHEMA`, `FROZEN`, `PROMPT_VERSION = "v5c-chat"`,
  `parse_verdict`, `_readout_class`, revocation, `hallucination_rate`, its bootstrap, and the
  `complete` rule are retained exactly. The verify stage reads the span judge's verified
  `wrong` spans; it never changes them.
- `counts` keeps its fixed key set; every new number lives in `extras` (pooled, `by_layer`,
  `by_site_kind`).

## 1. `prompts.py`

`VERIFY_PROMPT_VERSION = "v5c-chat-verify-v1"`, `VERIFY_STATUSES = ("true", "unverifiable",
"disputed", "off_topic")`, `VERIFY_SYSTEM`, `VERIFY_USER`, `VERIFY_SCHEMA`
(`schema_block("hallucination_chat_verify", …)`) copied verbatim from `chat_verify.py`;
`verify_samples_block(samples, wrong)` (readout blocks each followed by "Established FALSE
spans of readout i:" + `json.dumps` lines or "(none)") and `verify_prompt(item, site, samples,
wrong) -> (system, user)` rendered with `str.format` like `JUDGE_USER` (same `{{quote, status,
why}}` escapes). `PROMPTS` gains `VERIFY_SYSTEM` and `VERIFY_USER` (README-verbatim test).

## 2. `judge.py`

- `tally_claims(raw, samples, wrong) -> list[dict[str, int]] | None`: port of
  `chat_verify.tally`. Per readout `{"false": len(wrong[i]), "true", "unverifiable",
  "disputed", "off_topic", "n_unverified"}`; a quote that is not verbatim in its own readout
  (`verify_quote`) → `n_unverified`, dropped; a verified quote overlapping an established
  false span (`normalize_ws` substring either way) → skipped (already false); `disputed` →
  `disputed` AND `unverifiable`; any other / unknown status → `unverifiable` (never inflates
  `true`); a missing `idx` → `None` (cell unjudged, re-queued).
- `ReadoutVerdict.claims: dict[str, int] | None = None`.
- In `run`, after the span batch and inside the same `Cache`: for every cell whose verdict
  parsed (`verdicts is not None`) build `Call(f"{c.key}:V", *verify_prompt(item, site,
  samples_of[c.key], [v.wrong for v in verdicts]), meta={..., "wrong": wrong})`; one
  `run_calls(schema=VERIFY_SCHEMA, prompt_version=VERIFY_PROMPT_VERSION, judge=args.judge,
  validate=lambda call, r: tally_claims(r, call.meta["samples"], call.meta["wrong"]) is not
  None)`; attach the tallies to the verdicts. Skipped entirely under `opts=verify=0`
  (`args.extra.get("verify") == "0"`); a dry run makes no verify call (no verdict exists).
  The fingerprint `(VERIFY_PROMPT_VERSION, model, reasoning, temperature, system, user)`
  embeds the wrong spans, so a re-judged cell is re-verified.
- Rows: each `verdict[i]` gains `"claims": {...} | null`. `config` gains `verify` (bool) and
  `verify_prompt_version` (`None` when off).

## 3. `score.py`

- `_counts` adds `n_readouts_claims_judged` (readouts with a tally), `n_claims_false`
  (= Σ len(wrong) over those readouts), `n_claims_true`, `n_claims_unverifiable`,
  `n_claims_disputed`, `n_claims_off_topic`, `n_unverified_claim_quotes`, and
  `n_cells_claims_unjudged` (judged cells whose verify failed or was skipped).
- `_rates` adds `verifiable_share = (true + false) / (true + false + unverifiable)`,
  `false_share_of_verifiable = false / (true + false)`, `unverifiable_claims_per_readout =
  unverifiable / n_readouts_claims_judged` (all `None` on a zero denominator, like the rest).
  These flow into pooled / `by_layer` / `by_site_kind` through `_block` unchanged.
- `complete` unchanged (headline only); `n_cells_claims_unjudged` is the coverage signal.

## 4. Docs and tests

- `evals/hallucination/README.md`: a "Claim verification (second stage)" bullet + section
  (what, the four statuses, the folding rules, the extras, `opts=verify=0`, cost 2×), the two
  prompts verbatim under "Judge prompts", `VERIFY_PROMPT_VERSION` named. `README.md`
  hallucination *Judged by:* line gains "; a second call per cell counts the readout's other
  claims as true / unverifiable against the transcript (extras)". `EvalSpec.calls_per_arm` →
  `"≈ 11k (5.6k span + 5.6k claim)"`. CLAUDE.md: no change needed (family READMEs are the
  source of truth).
- `tests/test_hallucination.py`: `tally_claims` units (unverified quote counted+dropped,
  overlap with a false span skipped, disputed folded, unknown status → unverifiable, missing
  idx → None, empty claims → zeros); the `_responder` answers `VERIFY_SYSTEM` with claims
  (one true, one unverifiable, one disputed, one off-topic, one paraphrase) so the end-to-end
  test asserts the extras arithmetic, the `:V` cache keys and 8 calls (4 span + 4 verify);
  `opts=verify=0` → 4 calls, `config.verify_prompt_version is None`, claim extras all zero and
  `n_cells_claims_unjudged == n_cells`; a failed verify call → `n_cells_claims_unjudged == 1`
  with the headline unchanged; resume reuses the cached `:V` rows; dry run prints only the
  span prompt. `test_partly_judged_cell_is_unjudged_and_retried` etc. get the extra calls
  accounted for.

## Cost

A full multilayer arm goes from 5,615 to 11,230 judge calls (the verify prompt is longer:
it carries the readouts twice — text and false spans). The source run of this pass was ≈ $60
reported / ≈ $150 actual on OpenRouter per arm. No live run in this PR.

## Verification

Full `pytest -q`, `ruff check`, `ruff format --check`; `wsbench judge family=hallucination
readouts=examples/readouts/hallucination.jsonl out=/tmp/h dry_run=True` still prints the v5c
prompt only.

## Amendments after critique round 1

- **`tests/test_hallucination.py::_judge_calls` filters on the user prefix**, which the verify
  prompt shares. It becomes a filter on `system == hp.JUDGE_SYSTEM`; a sibling
  `_verify_calls` filters on `hp.VERIFY_SYSTEM`. Call-count assertions to re-check: L299 (4
  span calls), L339, L364 (resume: the sorted user lists), L372 (`len(users) == 2` → the
  summary + span calls only), L460 (1123 span calls). `_responder` dispatches on `system` for the
  verify prompt (an `assert` there would be swallowed by the client's exception handling and
  surface only as `n_cells_claims_unjudged`, so the new tests assert that count is 0).
- **`tally_claims` re-queues a wrong-shaped answer:** every sample must carry a `claims` list
  (`isinstance(s.get("claims"), list)`), else `None` — a span-shaped answer never scores as
  "zero claims".
- **Toy arithmetic:** the responder flags readout 1's whole text as wrong, so any claim quoted
  from readout 1 overlaps a false span and is skipped; the end-to-end expectations quote claims
  from readouts 0 and 2 only, and one claim from readout 1 to assert the overlap skip.
- **Dry run / empty batch:** the verify `run_calls` is guarded by `if vcalls` (like the span
  batch), so a dry run prints only the span prompt.
- **Definitions:** `n_cells_claims_unjudged` = span-judged cells without a claim tally
  (verify failed OR `opts=verify=0`); `n_claims_false` is span-level (Σ `len(wrong)` over
  tallied readouts) and ignores revocation, unlike the readout-level headline — the README says
  so. Token arms: quotes are verified against the summarizer's interpretation, not the raw
  bag (same caveat as the span stage). Verify runs on `args.judge`, so a `judge_model=` override
  re-verifies too (fingerprint embeds the model).
- Cache keys: `<key>:V` is disjoint from the summarizer's `summ:<key>`; `Call.meta` uses
  `samples` / `wrong` / `id` / `layer` / `pos` / `site_kind` (never `key` / `fp` / `ts`).
