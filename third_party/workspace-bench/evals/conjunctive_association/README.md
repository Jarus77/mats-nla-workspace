# Conjunctive association

In-house family (no external source). A vignette implies a compound state (state × content ×
relation) without naming it; does the lens *state* the whole composition?

- **Bank:** `items.json`, 100 items, verbatim copy of the source repo's
  `…/multi_token/conjunctive_association/items.json`. Fields used: `id`, `gold_label`,
  `contrast_label`, `mc_options` (the frozen 10-way list).
- **Cells:** one (layer, item) bag per layer present in the readouts file; **one call per item**
  over a blob of every bag. Items missing any selected layer are excluded with a warning
  (`extras.n_items_excluded`, outside the denominator).
- **Read regime (producer side, since 2026-09-22):** the stimulus is chat-rendered with
  `"\n\nSummarize the preceding story in one sentence."` appended, and the bag holds only the
  suffix tokens and the chat-tail specials after them (19 sites per item, layers
  20/28/36/44/52/60) — not every position of the render as before. Story positions are never
  read, so a lens that echoes the current token cannot hand the judge the story for it to
  compose the answer itself.
- **Blob** (copied from the source's `score_lens_readouts.py` L26-46 as `norm_token` /
  `BPE_JUNK` / `build_bags`): per (layer, item) every row's samples are `norm_token`'d (strip
  `Ġ Ċ ▁`, `_` → space) and joined with `" | "` in **file order** (not sorted by `pos`), rows
  joined the same way; then `blob += f"\n[L{layer}] " + text` over ascending layers. Note
  `norm_token` is applied to prose samples too, so AO text loses its underscores. The blob is
  sliced to `char_cap` FIRST, then non-printables are stripped. `char_cap` defaults to
  **200000** (the full-blob instrument of record; the source default 24000 is not carried) and
  is set with `opts=char_cap=N`.
- **Item pass** = the judge picks the gold description.
- **Metric:** `pass_rate` = passes / n_items, n_items = in-scope items minus excluded-layer
  items; a failed call stays in the denominator as an `api_fail` row. `chance = 1/11` (single
  call, no grid).
- **Judge:** `google/gemini-3.8-flash`, `PROMPT_VERSION = "comp-v1"`.

## Judge prompts

Source: `scripts/oracle_lens/latent_eval/judge_mc.py` (L43-62, L76-85, L190), copied verbatim.
Templates are rendered with `str.replace`.

### `SYSTEM`

```text
You judge the READOUT of an interpretability lens: text decoded from a language model's internal activations while it read a passage you cannot see (possibly labeled by layer; may contain noise, fragments, or other languages). You are given ONE multiple-choice question — which of the listed compositional descriptions the readout STATES. Pick an option ONLY if the readout names its content directly, as the direct word(s) or an EXACT synonym ("mourning" names grief; "wife" names spouse; "his little boy" names her young child). A faithful translation of the option into another language counts as naming it. Do NOT pick an option you can merely INFER from evidence, causes, scene details, or narrative ("small sneakers" do not name "her young child"), do NOT count related-but-different states ("regret" is not grief; "worried" is not relieved) or generic words too broad to name it ("sad", "upset"), and do NOT use world knowledge or plausibility to fill gaps. If the readout does not state any listed option, choose the "cannot tell from the readout" option. Give a SHORT exact quote (a few words, VERBATIM) from the readouts that supports your choice.
```

### `QUESTION`

`{listing}` = the numbered option list.

```text
Which of the following does the readout state?
{listing}
```

### `USER`

User message; `{r}` = the item's whole blob, `{q}` = the rendered question.

```text
READOUTS:
{r}

{q}
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

Eleven lines: the item's frozen `mc_options` deduped order-preserving (gold asserted present),
seeded-shuffled with `seeded_shuffle(opts, f"{COMP_SEED}:{id}")`, then
`cannot tell from the readout` last. `COMP_SEED = 20260805`. Parity:
`tests/golden/conjunctive_association_options.json` was produced by the source script's own
`build_question` (`make_conjunctive_association.py`, a retired golden maker; see `tests/golden/README.md`);
`tests/test_conjunctive_association.py` asserts this port reproduces it.

### Verdict rows

`{id, choice, gold_pos, contrast_pos, pick, correct, quote, quote_ok}`; `pick` ∈ gold /
contrast (the flipped sibling) / distractor / cannot_tell / invalid / api_fail. `quote_ok`
(normalized quote is a substring of the blob) is recorded, not enforced. `extras` =
`breakdown {gold, contrast, distractor, cannot_tell, invalid, api_fail}`, `char_cap`,
`n_items_excluded`.

## Token (J-lens) readouts

A `tokens` readouts file goes through the shared summarizer first (`docs/summarizer.md`,
`SUMMARIZER_PROMPT_VERSION = "interp-v1"`): each (layer, item) bag — the `" | "` blob of `norm_token`'d tokens across positions — is summarized before blobbing, keyed `<id>__L<layer>`. The judge then sees the interpretation
text in place of the readout. A cell whose summary fails is skipped (counted in
`n_unjudged_cells`); an empty string is never judged.

**Legacy note.** The source script carried its own copy of the interpretation prompt
(`JLENS_SUMMARIZER_SYSTEM` / `JLENS_SUMMARIZER_USER` from `global_workspace/judges/oracle_lens_judge.py` L1021-1039, via `summarize_token_bundles`);
this repo uses the one shared prompt. The original is quoted here for provenance only — it is
not what this family sends:

```text
You convert raw interpretability readouts into plain language. You will see the top-k vocabulary tokens decoded from ONE internal activation of a language model — word fragments, other scripts, and noise tokens are normal. Describe what the activation appears to encode, naming the concrete concepts the tokens collectively point to. Ground every claim in the tokens shown; if they are noise, say so. Never mention tokens, lenses, or this instruction. Two sentences at most. Output only the JSON.
```

```text
Top-k tokens decoded from one activation (separated by " | "):

{bundle}

Describe in plain language what this activation appears to encode.
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
