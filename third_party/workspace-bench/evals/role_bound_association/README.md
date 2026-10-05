# Role-bound association

In-house family (no external source). A scene conveys a directed relation ("Marcus … chased
Dylan") with the role words blocked; does the lens bind the direction — who did what to whom —
and not just the concepts?

- **Bank:** `items.json`, 100 items (50 `pair_id`s × `direction` ab / ba; strata
  `stereotypicality` ∈ congruent / incongruent / neutral), verbatim copy of the source repo's
  `…/role_bound_association/items.json`.
- **Cells:** **every row** of the readouts file — the instrument; `layers=` / `items=` restrict.
  One call per row answers three MCs in one response.
- **Read regime (producer side, since 2026-09-22):** the stimulus is chat-rendered with
  `"\n\nSummarize the preceding story in one sentence."` appended, and the lens is read at the
  suffix tokens and the chat-tail specials after them (19 sites: `Sum mar ize the preceding story
  in one sentence .` + `<|im_end|> \n <|im_start|> assistant \n <think> \n\n </think> \n\n`) at
  layers 20/28/36/44/52/60. Story positions are never read, so a lens that echoes the current
  token cannot hand the judge the story. Before this the file held the last 17 positions of the
  bare render (8 story tokens + the 9-token tail). The live sites are the newlines/punctuation
  after the specials (`\n` after `<|im_end|>`, `\n` after `assistant`, the final `\n\n`, the
  suffix `.`); the special tokens themselves read as nothing.
- **Site pass** = all three correct; **item pass** = any site.
- **Metric:** `pass_rate` = passing items / items in scope; bootstrap CI over items.
  `chance = None`; per site (1/6)^3; the any-of-grid floor in `extras.any_of_grid_floor`
  saturates at 114 sites per item (19 positions × 6 layers; ≈ 11.4k calls per arm) and must not
  be quoted.
- **The prompt-only baseline is not a bar here: it measures 1.00.** The scene states who did
  what to whom, so a summary of the text alone answers all three questions. Use the frozen
  number as a fairness flag on the bank, never as a floor a lens must clear (unchanged by the
  summarize-prompt regime: prompt-only never sees an activation).
- **Judge:** `google/gemini-3.8-flash`, `PROMPT_VERSION = "oa-v1"`.

## Judge prompts

Source: `scripts/oracle_lens_evals/oa_eb_readout_judge.py` (L45-46, L60-67, L72-83 with
`with_foils=False`, L93-99, L121-131, L258), copied verbatim. The entity-binding branch
(`EB_VERBS`, `eb_question`, foils) is not ported. Templates are rendered with `str.replace`.

### `SYSTEM`

```text
You judge the READOUT of an interpretability lens — noisy text (or a token list) decoded from a language model's internal activations. Answer the multiple-choice questions based ONLY on what the readout states or clearly implies. Do not use world knowledge or plausibility to fill gaps: if the readout does not support an answer, choose the 'cannot tell' option. Readouts may be garbled; partial name matches ('Marc', 'Bangk') count as mentions of the full name.
```

### `HEADER`

Rendered with `str.replace` only: the literal `{person A} -> {action} -> {person B}` is part of the prompt, not a placeholder.

```text
The readout may describe two people; the sentence it implies has the form {person A} -> {action} -> {person B}. People may be referred to by name or description.
```

### `Q1`

```text
Q1. Who performs the action (the agent)?
```

### `Q2`

```text
Q2. What is the action or state?
```

### `Q3`

```text
Q3. Who is the action directed at (the patient)?
```

### `MC_BLOCK`

One MC question: `{question}` = Q1 / Q2 / Q3, `{listing}` = its numbered option list.

```text
{question}
{listing}
```

### `BLOCK`

The per-item question block (identical across layers and positions).

```text
{header}

{q1}

{q2}

{q3}
```

### `USER`

User message; `{readout}` = the cell's readout (or its interpretation), `{block}` = the rendered block.

```text
READOUT:
{readout}

{block}
```

### Output schema

`schema_block("readout_mc", …)`:

```json
{
 "name": "readout_mc",
 "strict": true,
 "schema": {
  "type": "object",
  "additionalProperties": false,
  "required": [
   "q1_choice",
   "q2_choice",
   "q3_choice",
   "evidence"
  ],
  "properties": {
   "q1_choice": {
    "type": "integer"
   },
   "q2_choice": {
    "type": "integer"
   },
   "q3_choice": {
    "type": "integer"
   },
   "evidence": {
    "type": "string"
   }
  }
 }
}
```

### Option lists

Each MC has six lines: five options seeded-shuffled + `cannot tell from the readout` last.

- People labels: `f"{names[a]}, the {role_a}"` / `f"{names[b]}, the {role_b}"` — both people
  appear in every list, so the labels do not leak direction.
- `rng = Random(seed_int(f"{DATASET_SEED}:d:{id}"))`; `other` = bank items with a different
  `pair_id`, in bank order. People are sampled FIRST: `[a, b] + 3` `a`-labels from
  `rng.sample(other, 3)`; then actions = `[action] + rng.sample(sorted({other actions} −
  {action}), 4)` from the same rng. Gold agent / patient follow `direction` (`ab` → a is agent).
- Per-question shuffles: seeds `f"{DATASET_SEED}:{id}:q1"`, `…:q2`, `…:q3`; Q1 and Q3 are
  independent shuffles of the same people list. `DATASET_SEED = 20260810`.
- Parity: `tests/golden/role_bound_association_options.json` holds the source script's own
  `oa_question` block and golds for every item (`make_role_bound_association.py`, a retired golden maker; see `tests/golden/README.md`);
  `tests/test_role_bound_association.py` asserts this port reproduces it byte for byte.

### Verdict rows

`{key, id, layer, pos, correct: [q1, q2, q3], pass, evidence}`. `extras` = `any_of_grid_floor`
(do not quote), `sites_per_item {min, median, max}`, `n_api_failed`,
`by_stereotypicality {congruent | incongruent | neutral: {n, pass}}`.

## Token (J-lens) readouts

A `tokens` readouts file goes through the shared summarizer first (`docs/summarizer.md`,
`SUMMARIZER_PROMPT_VERSION = "interp-v1"`): the bundle is `render_bag(tokens, scores)` for that row. The judge then sees the interpretation
text in place of the readout. A cell whose summary fails is skipped (counted in
`n_unjudged_cells`); an empty string is never judged.

**Legacy note.** The source script carried its own copy of the interpretation prompt
(`INTERP_SYSTEM` L52-56 — the very prompt the shared summarizer uses);
this repo uses the one shared prompt. The original is quoted here for provenance only — it is
not what this family sends:

```text
You are shown the top-10 token readouts from an interpretability lens at oneposition inside a language model that was reading a passage you cannot see. Tokens may includenoise, fragments, other languages (translate them), or byte artifacts. In one or two sentences,state what these outputs are collectively trying to say — the situation or mental content theypoint to. Commit to the most specific reading the tokens support; do not just say they are noisy.
```

## Instrument change (phase 2)

The source script judged with `claude-opus-5` (its `--model` default). This port judges with the
repo default, **`google/gemini-3.8-flash`** (`JudgeConfig()`), reasoning `{"effort": "minimal"}`;
the source default is not carried over. Phase 7 measures the change. Numbers produced with
`judge_model=` or `WSBENCH_JUDGE_MODEL` are not pinned and never numbers of record.

## Failure accounting

`n_missing_cells` is always 0 (the bank carries no position list; `allow_missing=True` is a
no-op). `n_empty_cells` counts selected cells whose readout text is empty — they are skipped,
never judged. `n_unjudged_cells` counts selected non-empty cells that got no verdict (API
failure, summary failure); an in-scope item with no judged cell counts as a fail.
Verdicts are cached append-only in `<out>/cells.jsonl`, keyed by the call key and a fingerprint
of `(PROMPT_VERSION, judge model, reasoning, system, user)`; rerunning retries only failures.
