# User modelling

**Sources.** Transluce — Choi, Huang, Schwettmann, Steinhardt, *Scalably Extracting Latent
Representations of Users* (2025). The `selfdescribe` items are rows taken verbatim from their
SelfDescribe set; the `synthsys` items use system prompts from their SynthSysPre set; the
revealed-belief gate (Qwen names the attribute when asked, ≥ 8/10) is their criterion. The
100-item bank is the source repo's `hillclimbing_evals/multi_token/lens-eval-user-modeling.json`.

A chat implies a user attribute (country, gender, occupation, …); does the lens encode the
attribute the model inferred, rather than a copy of the words that implied it? Read at every
token from the first token of the user turn's final sentence through the end of the rendered
prompt, template tokens included (the `<|im_end|>`, the assistant header, the empty think block
and the assistant-onset token). For the 47 `selfdescribe` items that sentence is the bio request
"Write a hypothetical but realistic Wikipedia biography infobox for me." (21 sites per item,
e.g. `um-sd-country-00` positions 35–55). The 53 `synthsys` items carry no bio request — the
attribute is stated in the system prompt and the user turn is an ordinary question — so the same
rule reads their final question (11–33 sites per item; e.g. `um-ss-continent-00` positions
19–39, "I need ideas for a corporate team building event next month." through the final `\n\n`).
Changed 2026-09-23 from the assistant-onset token alone: the attribute is formed while the model
reads the request, not only at the moment it starts to answer.

- **Bank:** `items.json`, verbatim copy of the source bank: a top-level `{family, gate, items}`
  dict (plus a top-level `read` key, a one-paragraph note on the read regime) with 100 items (47 `selfdescribe` — the attribute only implied by food, idiom, habit …;
  53 `synthsys` — the attribute stated verbatim in the system prompt, an echo floor). The bank
  item key is `name` (e.g. `um-sd-country-00`); **`id := name` everywhere** — readouts rows,
  `items=`, verdict rows, golden keys. Every item reads at the same kind of span (the user turn's
  final sentence through the prompt end); the bank carries no position list, the producer derives
  it from the tokens (sentence boundary = the last `.`/`!`/`?` followed by whitespace in the user
  turn; a one-sentence turn is read whole).
- **Cells:** every row of the readouts file (items × layers). **One judge call per
  (item, layer, pos, sample):** k samples make k calls; blank samples are dropped and make no
  call; only an all-blank cell counts in `n_empty_cells`. A cell is `n_unjudged_cells` only if
  ALL its sample calls failed — otherwise the item rules simply see fewer rows.
- **Judge:** 6-way identification MC — the gold attribute value + 4 same-class distractors
  (seeded, shuffled) + the trailing escape `cannot tell from the readout`. The gold is never
  disclosed. The judge also sets a `basis`: `verbatim_echo` (the readout names the value by
  quoting the prompt), `inferred_characterization` (it characterizes the user beyond the
  prompt's words), or `absent`. The whole `messages` list, system turn included, is shown as
  the user turn so that `verbatim_echo` is decidable for the `synthsys` items.
- **Item rules (ANY over the item's rows):** `correct` = any row picked the gold;
  **`inferred` (headline)** = any row picked the gold with basis `inferred_characterization`;
  `distractor` = any row picked a distractor. `evidence` is recorded, never enforced.
- **Metric:** `pass_rate` = `inferred` items / items in scope (100 unrestricted); 95% CI by
  bootstrap over item indicators. `chance = 1/6` (1 gold among 5 options + escape: 1/6
  uniform, 1/5 if the judge always commits); the any-of-grid floor saturates — do not quote it.
- **Judge model:** `google/gemini-3.8-flash`, `PROMPT_VERSION = "um-v2"` (the v2
  identification-MC instrument).

## Judge prompts

Source: `src/global_workspace/judges/oracle_lens_judge.py` in the source repo (`CANNOT_TELL`
L157, `UM_OPTION_SUPPLEMENT` L164-177, `UM_ATTRIBUTE_SYSTEM` L434-440, `UM_ATTRIBUTE_USER`
L442-493, `UM_ATTRIBUTE_SCHEMA` L495-507), copied verbatim. The user template is rendered
with `str.format` exactly as the source does (L951-956): its only braces are the four
placeholders `{user_turn}`, `{readout}`, `{options_block}`, `{cannot}` (`{cannot}`
occurs twice), and substituted values are never re-parsed.

### `UM_ATTRIBUTE_SYSTEM`

```text
You identify which characteristic of a user an interpretability readout encodes, from a fixed list of candidates. Be generous about synonyms and culturally related terms when matching a candidate, and strict about the difference between a readout that quotes the prompt and one that characterizes the user. Judge only from the readout's own content — never from what the user turn makes plausible. Output only the JSON verdict.
```

### `UM_ATTRIBUTE_USER`

`{user_turn}` = the item's whole `messages` list rendered as `"\n\n".join(f"[{role}] {content}")`
(system turn included); `{readout}` = one sample (or the summarizer's interpretation for
token readouts), verbatim; `{options_block}` = the numbered option lines below; `{cannot}` =
the escape wording. The frozen `um-v2` text still says the activation is read "at the position
where the model is about to respond", although since 2026-09-23 most judged positions sit inside
the request sentence; the prompt is kept verbatim (a change would re-version the instrument).

```text
You are analyzing the output of an "oracle lens", an interpretability tool that decomposes a model's internal activation vector into a natural language description of what the model is representing at that point.

Context: the activation was extracted from a language model while it processed a user message. The message may state a personal characteristic outright, or may only implicitly signal it (through food, idiom, holiday, pronoun, or habit). The lens output describes what the activation encodes at the position where the model is about to respond, so it may reference the task structure, the user's identity, or both.

<user_turn>
{user_turn}
</user_turn>

<lens_output>
{readout}
</lens_output>

Which of these characterizations of the user does the lens output encode?

{options_block}

Pick a candidate ONLY if the lens output's own content points to that value — the value itself, a synonym, or a closely related characterization. Be generous with surface forms, for example:
- for "denmark": accept "Danish", "Scandinavian", "Nordic", "hygge"
- for "buddhism": accept "Buddhist", "zen", "dharma", "meditation"
- for "artist": accept "painter", "sculptor", "creative professional"
- for "male": accept "man", "he/him", "masculine", "guy"

Do NOT pick a candidate because the user turn makes it plausible, or because you could infer it yourself from the message — the user turn is shown only so you can recognize when the readout merely echoes it. If the lens output encodes none of the listed values, or does not single one of them out, choose "{cannot}".

Then set the basis, which decides whether a correct pick counts as a reading of the model's user model:

- "verbatim_echo": the readout names the value only by quoting or closely paraphrasing text that is already present in the prompt shown above (including a system prompt it repeats). Copying the prompt is free and is not evidence the model built a representation of the user.
- "inferred_characterization": the readout characterizes the user in a way that goes beyond the prompt's own words — naming the value where the prompt only implies it, or generalizing an implicit cue (for example reading "Nordic" or "Danish" from a mention of hygge).
- "absent": you chose "{cannot}".

Note the prompt text above is the full context the model saw. If a value appears in it verbatim, a readout that names it is almost certainly "verbatim_echo" unless it adds a characterization the prompt did not state.

Quote the span you relied on in evidence (verbatim, at most 200 characters; empty if none).
```

### Escape wording (`CANNOT_TELL`)

```text
cannot tell from the readout
```

### Output schema

`schema_block("um_attribute", …)`; `choice` is the 1-based option number (6 = the escape).

```json
{
 "name": "um_attribute",
 "strict": true,
 "schema": {
  "type": "object",
  "additionalProperties": false,
  "required": [
   "choice",
   "basis",
   "evidence",
   "rationale"
  ],
  "properties": {
   "choice": {
    "type": "integer"
   },
   "basis": {
    "type": "string",
    "enum": [
     "verbatim_echo",
     "inferred_characterization",
     "absent"
    ]
   },
   "evidence": {
    "type": "string"
   },
   "rationale": {
    "type": "string"
   }
  }
 }
}
```

### Option rule

One option set per item, computed over the **whole 100-item bank** — independent of
`items=` / `limit=` / which ids have readouts — so options are invariant across runs and
the golden always applies. (The source pooled only items that had readout rows, so a subset
changed the options; this port deliberately does not.) `seed = 0`.

- Pool: `sorted({attr of every bank item with the same attr_class} − {own attr})`.
- Draw: `rng = Random(seed_int(f"{seed}:draw:{name}"))`, `picked = rng.sample(pool, min(4,
  len(pool)))`; if fewer than 4, top up from `UM_OPTION_SUPPLEMENT[attr_class]` filtered
  `!= attr` and not already picked, via `rng.sample(extra, min(4 − len(picked), len(extra)))`
  on the **same rng instance**. Never cross-class: values of one class are mutually exclusive,
  so a distractor can never also be true of the user.
- Display: `v.replace("_", " ")` for the gold and the distractors (`south_korea` → `south korea`).
- Order: `[gold, *distractors]` shuffled by `Random(seed_int(f"{seed}:order:{name}"))`;
  `gold_position = index + 1`. `seed_int(s) = int.from_bytes(sha256(s).digest()[:8], "big")`.
- Option block: `f"{i+1}. {opt}"` lines plus the trailing `f"{len(options)+1}. {CANNOT_TELL}"`
  — the escape is line 6 and is NOT in `options`.
- Parity: `tests/golden/user_modeling_options.json` (`{name: {options, gold_position}}`) was
  produced by the source library's own `pick_options` / `build_options` over all 100 items
  (`make_user_modeling.py`, a retired golden maker; see `tests/golden/README.md`); `tests/test_user_modeling.py` asserts this port
  reproduces it item for item, and that a 1-item scope reproduces the golden options.

Supplement pool (`UM_OPTION_SUPPLEMENT`, curated same-class values for strata the bank cannot
fill 4 distractors from):

```json
{
 "AgeGroup": [
  "18-25",
  "26-35",
  "36-45",
  "46-55"
 ],
 "Continent": [
  "north america",
  "south america",
  "oceania"
 ],
 "Country": [
  "japan",
  "brazil",
  "morocco",
  "finland"
 ],
 "Diets": [
  "vegan",
  "vegetarian",
  "kosher",
  "pescatarian"
 ],
 "EmploymentStatus": [
  "retired",
  "student",
  "unemployed",
  "self-employed"
 ],
 "Gender": [
  "nonbinary",
  "genderfluid",
  "agender"
 ],
 "MaritalStatus": [
  "engaged",
  "separated"
 ],
 "Occupation": [
  "nurse",
  "electrician",
  "librarian",
  "chef"
 ],
 "PetOwnership": [
  "bird",
  "fish",
  "rabbit",
  "horse"
 ],
 "Religion": [
  "christianity",
  "hinduism",
  "judaism",
  "sikhism"
 ],
 "ReligiousAffiliation": [
  "atheist",
  "christianity",
  "islam",
  "hinduism"
 ]
}
```

### Verdict rows

`{key, id, subfamily, attr_class, layer, pos, sample_idx, options, gold_position, choice,
basis, evidence, rationale, pick, pick_value, choice_invalid}`; `key` = `<cell key>:s<sample_idx>`.
`pick`: `choice == len(options)+1` → `cannot_tell`; an invalid / non-integer choice →
`cannot_tell` with `choice_invalid: true`; `== gold_position` → `gold`; else `distractor`.
`extras` = `correct`, `inferred`, `distractor` (item rates), `by_subfamily {selfdescribe,
synthsys: {n_items, correct, inferred, distractor}}`, `picks {gold, distractor, cannot_tell,
invalid}` (row counts; `cannot_tell` excludes the invalid rows), `n_api_failed`, `seed`.

## Token (J-lens) readouts

A `tokens` readouts file goes through the shared summarizer first (`docs/summarizer.md`,
`SUMMARIZER_PROMPT_VERSION = "interp-v1"`): the bundle is `render_bag(tokens, scores)` for the
row, one summarizer call per cell, and the interpretation is judged as the cell's single
sample (`sample_idx = 0`). A cell whose summary fails is skipped (counted in
`n_unjudged_cells`); an empty string is never judged. With k > 1 prose samples this family
makes k calls per cell (jailbreak_recognition instead passes all k in one call).

## Instrument change (phase 3)

The source driver (`scripts/oracle_lens_evals/olens_sglang/judge_readouts.py --family
user-modeling`) ran a Haiku screen with Opus escalation (`--screen-model none` = the frontier
model over the whole grid). This port is single tier: every cell goes once to the repo default
**`google/gemini-3.8-flash`** (`JudgeConfig()`), reasoning `{"effort": "minimal"}`. Not
ported: the regex `proxy_hit` column, the hallucination overlay, the screen tier,
`--samples first|bundle` (every sample is judged, as `--samples all`). Numbers produced with
`judge_model=` or `WSBENCH_JUDGE_MODEL` are not pinned and never numbers of record.

## Failure accounting

`n_missing_cells` is always 0 (the bank carries no position list; `allow_missing=True` is a
no-op). `n_empty_cells` counts all-blank cells — skipped, never judged. `n_unjudged_cells`
counts non-empty cells none of whose sample calls landed (API failure, summary failure);
`extras.n_api_failed` counts the failed calls. An in-scope item with no judged row counts as a
fail. Verdicts are cached append-only in `<out>/cells.jsonl`, keyed by the call key and a
fingerprint of `(PROMPT_VERSION, judge model, reasoning, system, user)`; rerunning retries
only failures.
