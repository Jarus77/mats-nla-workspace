# Relational multihop

In-house family (no external source). A two-hop possessive cloze — one professional hop, one
kinship hop ("Avery's landlord is Sam. Sam's sibling is Riley. In other words, Riley is
Avery's ___") — read at its blank; does the lens hold the composed relation in the right order?

- **Bank:** `items.json`, 100 items (50 scenes × a/b direction pair), verbatim copy of the source
  repo's `…/relational_multihop/cloze_items_final.json`. Fields used: `id`, `scene`, `hop1`
  (outer), `hop2` (inner).
- **Cells:** per (item, layer) the **max-pos row** is the blank. If that row carries a `token`
  it must be the possessive `'s` (`BLANK_TOKEN`); any other token raises with the item id. A
  row without a `token` is accepted and rendered as `[position p<pos>]`. `wsbench plan`: bare
  render, the final (blank) token, layers 20..60 step 4 (11 layers → 2,200 calls per arm).
- **Calls:** two per cell over the same readout and the same option list — X (outer) and Y
  (inner), keyed `<cell>:X` / `<cell>:Y`. Cell pass = both correct (a direction flip picks the
  other hop and fails). Item pass = any layer.
- **Metric:** `pass_rate` = passing items / items in scope; bootstrap CI over items.
  `chance = None`; per cell 1/121; the any-of-layers floor `1-(120/121)^n_layers` is reported in
  `extras.any_of_layers_floor`.
- **The prompt-only baseline measures 0.82**, because the cloze states both hops in the text;
  it bounds what the prompt gives away, not what a lens must clear.
- **Judge:** `google/gemini-3.8-flash`, `PROMPT_VERSION = "rel-v1"`.

## Judge prompts

Source: `scripts/oracle_lens_evals/judge_relational_multihop.py` (L70-85, L95-107, L183-194,
L209-214), copied verbatim. Templates are rendered with `str.replace` per placeholder.

### `JUDGE_SYSTEM`

```text
You judge the READOUT of an interpretability lens — noisy text decoded from a language model's internal activations. Answer the multiple-choice question based ONLY on what the readout states or clearly implies (synonyms, gendered forms, or another language's word for a relation count); do not use world knowledge or plausibility to fill gaps. The readout describes a relation of the form "X's Y" (X is the outer/first relation word, Y the inner/second). If the readout does not support an answer, choose 'cannot tell from the readout'. Give a SHORT exact quote (a few words, VERBATIM) from the readout that supports your choice.
```

### `ROLE_X`

```text
X (the OUTER/first relation word)
```

### `ROLE_Y`

```text
Y (the INNER/second relation word)
```

### `QUESTION`

User message for each of the two calls; `{readout}` = the bundle (or its interpretation), `{role}` = `ROLE_X` or `ROLE_Y`, `{listing}` = the numbered option list.

```text
READOUT:
---
{readout}
---
The readout describes a relation of the form "X's Y". What is {role}?
{listing}
```

### `BUNDLE`

The readout handed to the judge (prose) or to the summarizer (tokens): `{token!r}` = the Python repr of the row's `token` (`"'s"`), or `p<pos>` when the row carries no token; `{txt}` = the non-empty samples joined by `" | "` (tokens: `render_bag`).

```text
[position {token!r}] {txt}
```

### Output schema

`schema_block("relation_mc", …)`:

```json
{
 "name": "relation_mc",
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

Eleven lines, shared by both sub-questions: both hops + 4 kinship + 4 professional
distractors, seeded-shuffled, then `cannot tell from the readout` last.

- `KINSHIP = ['child', 'cousin', 'grandchild', 'grandparent', 'parent', 'sibling', 'spouse']`.
- Pools: kinship = the bank's kinship relations ∪ `KIN_EXTRA`; professional = the bank's other
  relations ∪ `PROF_EXTRA`; both pools are **sorted lists** before sampling and asserted
  disjoint. The extras give every common near miss its own line, so a confabulated adjacent
  relation (grandparent for parent, friend for roommate) fails instead of rounding to the gold:
  - `KIN_EXTRA = ['grandparent', 'grandchild', 'aunt/uncle', 'niece/nephew', 'in-law', 'stepparent']`
  - `PROF_EXTRA = ['friend', 'roommate', 'classmate', 'teammate', 'accountant', 'realtor']`
- Draw order matters: `rng = Random(seed_int(f"{REL_SEED}:dist:{id}:pooled"))`, then
  `rng.sample(kin − {hop1, hop2}, 4)` **then** `rng.sample(prof − {hop1, hop2}, 4)`; the
  pre-shuffle list is exactly `[hop1, hop2, *kin_dist, *prof_dist]`; shuffle seed
  `f"{REL_SEED}:{id}:pooled"`; asserted 10 distinct; + `CANNOT`. `REL_SEED = 20260813`.
- Parity: `tests/golden/relational_multihop_options.json` was produced by the source script's own
  `pools` / `build_mc` (`make_relational_multihop.py`, a retired golden maker; see `tests/golden/README.md`);
  `tests/test_relational_multihop.py` asserts this port reproduces it.

### Verdict rows

`{key, id, layer, pos, x_choice, x_gold_pos, x_ok, x_quote, y_choice, y_gold_pos, y_ok, y_quote,
pass}`. `extras` = `per_layer_pass {"L20": {pass, judged}, …}`, `pair_consistency {n, of}`
(scenes where both `rel-<scene>-a` and `-b` pass, of the scenes in the bank),
`any_of_layers_floor`, `n_layers`, `n_cells_judged`, `n_cells_api_failed`,
`n_cells_interp_missing`.

## Token (J-lens) readouts

A `tokens` readouts file goes through the shared summarizer first (`docs/summarizer.md`,
`SUMMARIZER_PROMPT_VERSION = "interp-v1"`): the bundle is the `BUNDLE` template above with `{txt} = render_bag(tokens, scores)` — the `[position …]` prefix IS part of the summarizer input, as in the source (L269). Deliberate deviation: the source sent that bundle with NO `TOKEN READOUTS:` prefix, whereas the shared summarizer always adds it. The judge then sees the interpretation
text in place of the readout. A cell whose summary fails is skipped (counted in
`n_unjudged_cells`); an empty string is never judged.

**Legacy note.** The source script carried its own copy of the interpretation prompt
(`INTERP_SYSTEM` L88-92, identical to the shared prompt);
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
failure, summary failure, or an X/Y pair with one failed call — the pair is dropped); an in-scope item with no judged cell counts as a fail.
Verdicts are cached append-only in `<out>/cells.jsonl`, keyed by the call key and a fingerprint
of `(PROMPT_VERSION, judge model, reasoning, system, user)`; rerunning retries only failures.
