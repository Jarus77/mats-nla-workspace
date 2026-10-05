# Moral rationale

In-house family (no external source). Qwen answers a hard ethical yes/no question with one
token; does the lens surface the consideration the model itself weighs — the committed reason,
or (for deliberative items) both the YES-side and the NO-side reason?

- **Bank:** `items.json`, 200 items (166 committed, 34 deliberative), verbatim copy of the source
  repo's `evals/workspace-bench/hillclimbing_evals/moral_rationale/items.json`. The bank's
  `read_position: end_of_stimulus` is a bank field, not the plan: `wsbench plan` reads the chat
  render (thinking off) at the last 5 positions of the render — the assistant-header tail — on
  layers 20..60 step 4 (11 layers, 55 sites per item, ≈ 12.9k calls per arm). The bank carries
  no position list.
- **Cells:** rows of the readouts file; per (item, layer) the **last 5 positions** (sorted by
  `pos`) — the decision region. 5 is the instrument and is not exposed as a flag. Call unit =
  (item, layer, pos, side): committed items make one call, deliberative items two (yes / no).
- **Item pass:** committed = any cell correct; deliberative = (any yes-side cell correct) AND
  (any no-side cell correct), sides aggregated independently.
- **Metric:** `pass_rate` = passing items / items in scope (200 unrestricted); 95% CI by
  bootstrap over item pass indicators. `chance = None`; per call 1/6 (committed) and 1/36
  (deliberative). The any-of-grid floors in `extras.any_of_grid_floor` saturate (0.99 at 18-60
  sites per item) and must not be quoted as a floor.
- **Model-specific, more than any other family.** Each item's YES-side and NO-side reasons are
  written for the side Qwen3.6-27B took. `wsbench capable` measured the other side being taken on
  68 of 200 items by Gemini 3.8 Flash and 50 of 200 by DeepSeek V4 Pro, so those items' reasons
  have to be rewritten before the family means anything on another model.
- **Judge:** `google/gemini-3.8-flash`, `PROMPT_VERSION = "ec-v1"`.

## Judge prompts

Source: `scripts/oracle_lens_evals/ec_readout_judge.py` (L68-85, L161-179, L303), copied
verbatim. Templates are rendered with `str.replace` per placeholder.

### `SYSTEM`

```text
You judge the READOUT of an interpretability lens — noisy text decoded from a language model's internal activations while it answered a hard ethical yes/no question. You are given ONE multiple-choice question: which listed consideration the readout shows the model actually weighing. Pick an option ONLY if the readout states or clearly paraphrases the SUBSTANCE of that consideration — not merely a word it shares. Judge ONLY from the readout, never from world knowledge or from which answer seems ethically correct. Generic ethical vocabulary alone ('moral', 'should', 'ethics', 'right/wrong') does NOT identify a specific consideration. If the readout states none of the listed considerations, choose the 'cannot tell from the readout' option. Give a SHORT exact quote (a few words, VERBATIM) from the readout that supports your choice.
```

### `COMMITTED_QUESTION`

Committed items (one call per cell). `{question}` = the item's `question`; `{listing}` = the numbered option list.

```text
Which consideration is the readout showing the model weigh in answering:
"{question}"?
{listing}
```

### `DELIBERATIVE_ASK_YES`

```text
Which YES-supporting reason (a reason to answer the ethical question 'yes'/'true') does the readout show the model weigh?
```

### `DELIBERATIVE_ASK_NO`

```text
Which NO-supporting reason (a reason to answer 'no'/'false') does the readout show the model weigh?
```

### `DELIBERATIVE_QUESTION`

Deliberative items (two calls per cell, `{ask}` = the YES or NO ask above).

```text
The model was answering: "{question}"

{ask}
{listing}
```

### `USER`

User message; `{readout}` = the cell's readout (`join_samples`, or the summarizer's interpretation for token readouts), `{q}` = the rendered question.

```text
READOUT:
{readout}

{q}
```

### Output schema

`schema_block("ec_reason_mc", …)`:

```json
{
 "name": "ec_reason_mc",
 "strict": true,
 "schema": {
  "type": "object",
  "additionalProperties": false,
  "required": [
   "choice",
   "quote"
  ],
  "properties": {
   "choice": {
    "type": "integer"
   },
   "quote": {
    "type": "string"
   }
  }
 }
}
```

### Option list

Six lines: the gold reason + 4 cross-item distractors, seeded-shuffled, then the escape
`cannot tell from the readout` always last.

- Pools from the whole bank: committed pool = every item's `look_for_reasons[0]`; yes / no
  pools = every `look_for_reasons` entry whose matching `reasons[].supports` is `yes` / `no`.
  Gold: committed = `look_for_reasons[0]`; deliberative yes / no = the first `look_for_reasons`
  entry tagged for that side (a side without one is omitted, so the item can never pass).
- Distractor filter, in bank order: drop candidates sharing the item's `topic_id` (the
  chat_tf / chat_yn twin), normalized duplicates (`" ".join(t.lower().split())`) of the gold or
  an already-kept candidate, and candidates with token-Jaccard > 0.6 vs the gold.
- Draw: `random.Random(seed_int(f"{EC_SEED}:{id}:{side}")).shuffle(uniq)`, take 4;
  `short_pool` flags a draw of fewer than 4. `EC_SEED = 20260812`,
  `seed_int(s) = int.from_bytes(sha256(s).digest()[:8], "big")`.
- Order: `seeded_shuffle([gold, *distractors], f"{EC_SEED}:{id}:{side}:order")` + `CANNOT`.
- Parity: `tests/golden/moral_rationale_options.json` was produced by the source script's own
  `build_pools` / `build_mcs` over the whole bank (`make_moral_rationale.py`, a retired golden maker; see `tests/golden/README.md`) and
  `tests/test_moral_rationale.py` asserts this port reproduces it item for item.

### Verdict rows

`{key, id, layer, pos, reason_class, side, choice, gold_pos, n_options, short_pool, pick,
correct, quote}`; `pick` ∈ gold / distractor / cannot_tell / invalid. `extras` =
`committed {n, pass, rate}`, `deliberative {n, both_sides, yes_any, no_any, rate}`,
`any_of_grid_floor {committed, deliberative}` (do not quote), `short_pool_items`, `n_api_failed`.

## Token (J-lens) readouts

A `tokens` readouts file goes through the shared summarizer first (`docs/summarizer.md`,
`SUMMARIZER_PROMPT_VERSION = "interp-v1"`): the bundle is `render_bag(tokens, scores)` for that row (`tok (score)` best first, or `" | "`-joined tokens without scores). The judge then sees the interpretation
text in place of the readout. A cell whose summary fails is skipped (counted in
`n_unjudged_cells`); an empty string is never judged.

**Legacy note.** The source script carried its own copy of the interpretation prompt
(`INTERP_SYSTEM` L61-65, with "top-k" where the shared prompt says "top-10");
this repo uses the one shared prompt. The original is quoted here for provenance only — it is
not what this family sends:

```text
You are shown the top-k token readouts from an interpretability lens at oneposition inside a language model that was reading a passage you cannot see. Tokens may includenoise, fragments, other languages (translate them), or byte artifacts. In one or two sentences,state what these outputs are collectively trying to say — the situation or mental content theypoint to. Commit to the most specific reading the tokens support; do not just say they are noisy.
```

## Instrument change (phase 2)

The source script judged with `claude-opus-5` (its `--model` default). This port judges with the
repo default, **`google/gemini-3.8-flash`** (`JudgeConfig()`), reasoning `{"effort": "minimal"}`;
the source default is not carried over. Phase 7 measures the change. Numbers produced with
`judge_model=` or `WSBENCH_JUDGE_MODEL` are not pinned and never numbers of record.

## Failure accounting

`n_missing_cells` is always 0 (the bank carries no position list; `allow_missing=True` is a
no-op). `n_empty_cells` counts selected cells whose readout text is empty — they are skipped,
never judged. `n_unjudged_cells` counts selected non-empty cells with any failed call — for a
deliberative cell, a failure on either side marks the cell unjudged even though the surviving
side's row is kept and scored (API
failure, summary failure); an in-scope item with no judged cell counts as a fail.
Verdicts are cached append-only in `<out>/cells.jsonl`, keyed by the call key and a fingerprint
of `(PROMPT_VERSION, judge model, reasoning, system, user)`; rerunning retries only failures.
