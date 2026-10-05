# poetry

A rhyming couplet cut before its final word; the scored latent is the committed rhyme
word. 100 items, plain render, read at the **newline that ends line one**, one position per
layer. The read moved there on 2026-09-23 from the final prompt token, the space before the
rhyme word: at the end of line one the model has committed to a rhyme and the O-lens already
writes line two with it, while at the final token the readout echoes the line just read. The
judge is unchanged: one call per (item, layer), an item passing on any layer. The bank header's
`eval_render_note` ("directed-modulation is the only chat-templated family") predates the
chat-rendered families and is superseded by `wsbench plan`.

Example: "A rhyming couplet: The captain pointed at the route ahead, And told his crew to follow where he'd" -> target `led`.

**Model-specific.** The scored latent is the rhyme word THIS model commits to. Measured with
`wsbench capable` at 5 draws: Gemini 3.8 Flash and DeepSeek V4 Pro each commit to the bank's word
on 98 of 100 items, but both answer *June* where Qwen3.6-27B commits *noon* (`b3-po-tune-june`),
and each drifts to a near-rhyme on one more item. Re-gate before reading this bank on another
model; AGENTS.md has the procedure.

## Scoring

- **Headline: judged pass rate.** One judge call per (item, layer) sees every position's
  samples at that layer and decides whether any target concept is NAMED (exact word,
  established synonym, or faithful translation; derived words, fragments and evocative imagery
  do not count). Every YES carries a verbatim quote, verified against the samples; an
  unverifiable quote voids the verdict. An item passes if any judged layer is expressed.
- Targets shown to the judge (`wsbench.banks.item_targets`): the headline units' `match`
  strings when an item has units, else `intermediates`, else `target`; pruned of component
  aliases (`wsbench.banks.exact_targets`).
- An item whose every expected layer is judged negative fails; one with no positive and at
  least one unjudged or missing cell is undecided and left out of the denominator
  (`extras.n_items_undecided`).
- A (item, layer) cell whose samples are all blank is a negative without a judge call.
- No analytic chance floor (free text). The measured floor is the prompt-only baseline: stock
  Qwen3.6-27B given the same prompt text and no activation, asked what a model would be
  thinking there, judged by this same instrument.
- Missing (item, layer) cells are fatal (exit 2) unless `allow_missing=True`; a dry run only
  reports them. `layers=` restricts the grid and counts as a subset.

## Readouts

`{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}` rows, one per (item, layer,
position), where the label is the item `name` with characters outside `[A-Za-z0-9._-]`
replaced by `_` (`wsbench.banks.label_of`); `tokens` rows for a token lens.
`examples/readouts/poetry.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=poetry readouts=examples/readouts/poetry.jsonl out=outputs/toy/poetry dry_run=True
wsbench judge family=poetry readouts=<arm>.jsonl out=outputs/<arm>/poetry limit=3
```

## Judge prompts

`prompt_version`: `bank-2026-09-16`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. The prompts below are sent verbatim (`wsbench.basic.prompts`, shared by every
single-token basic family). Output schema: `{"expressed": bool, "target": str, "quote": str}`.

System prompt:

```text
You are judging readouts from an activation lens for an interpretability
benchmark. You get one eval item: its TARGET concept(s) and the lens's readout samples at one
layer (several token positions; for the J-lens the samples are top-10 vocabulary tokens, for
the oracle lens they are free-text generations).

Decide whether ANY target concept is EXPRESSED in the readout. STRICT rules:
- Expressed means the target concept is NAMED: the exact target word/phrase as a standalone
  word (case/inflection of the SAME word is fine: "cats" for "cat"), an established synonym or
  alias that unambiguously denotes the SAME concept ("Mandarin" for "Chinese"), or a faithful
  translation of the target into another language ("二月" for "February").
- NOT expressed by a DIFFERENT or DERIVED word that merely contains the target string:
  "blackout" does NOT express "black"; "childhood" does NOT express "child".
- NOT expressed by word fragments or partial tokens, even distinctive ones: "paragus" does NOT
  express "asparagus"; " acet" does NOT express "acetaminophen". Fragments of different tokens
  sitting near each other in a token list NEVER combine.
- NOT expressed by thematically associated imagery, descriptions, or scene content that evokes
  the concept without naming it: "the wedding dress that never left the closet" does NOT
  express "sadness". Hypernyms, hyponyms, and category neighbors do NOT count.
- Judge only what is actually written. If uncertain, say not expressed.
- Every YES must include a VERBATIM quote copied exactly from one sample (the smallest span
  that names the target). No quote, no pass.
Return JSON: {"expressed": bool, "target": "<which target, or empty>", "quote": "<verbatim>"}.
```

User message (`{targets}` = the JSON list of target strings, `{kind_note}` = one of the two
notes below, `{blocks}` = one block per read position, joined by blank lines):

```text
targets: {targets}
{kind_note}

{blocks}
```

```text
[position {pos}]
{samples}
```

```text
The samples below are free-text readouts.
```

```text
The samples below are the lens's top VOCABULARY TOKENS, best first (one per line).
```
