# chain_intermediates

A two- or three-step arithmetic chain ("Halve it, rounding down." three times) whose START
NUMBER is given last, answered with no chain of thought. With the seed last nothing can be
computed before the seed token, so every intermediate has to be computed inside the read window
and is never written. Can a lens name it?

120 items (source bank `items_final.json`, nocot-bench's `chain` construct re-ordered; items
unchanged): 36 depth-3 (two intermediates) and 84 depth-2 (one), each gated on Qwen3.6-27B at
10/10 correct with no rollout stating an intermediate; no intermediate appears as a numeral in
the prompt, and start, intermediates and answer are pairwise distinct. Chat render, thinking
off.

Example: `Apply the steps below, in order, to the starting number given at the end. ... The starting number is 23.` →
intermediates [11, 5], answer 2.

## Scoring

- **Headline: pass rate, free recall.** One prompt-blind call per (item, layer) on the readout
  at the LAST token of the render (`cells=last`, the default): the assistant onset `\n\n` after
  the empty think block, not the user turn's last token. `wsbench plan` reads the chat render
  from the " What" token through the end of the render; `cells=last` judges the render's final
  token of those. The judge names the number(s) the readout presents as a computed
  value, ranked, at most three, or none. An item passes when the top-named value is one of its
  intermediates at any layer. The top value is credited only when it appears in the readout as a
  whole number (5 does not match 15) or the judge's quote is verbatim in it (a Chinese numeral is
  credited through the quote); a token bag is verified against its tokens, never its scores.
- **Every-cell read (`opts=cells=all`).** The same judge over every (layer, position) row the
  producer put in the file — the intended use is the question span, "What is the final number?
  Reply with only the final number, nothing else." plus the chat-template tokens after it, where
  the model has both intermediates and is about to answer. An item passes when ANY cell's top
  value is an intermediate; each row lists `hits_at`, and `extras.per_layer_hit_rate` and
  `extras.cells_per_item` size the grid. An any-of-grid rule over many cells inflates the pass
  rate on its own: read it against `null_top1_near`, taken over the same cells.
- **Token lenses** are judged as their top-k bag (the prompt's "bag of loose numerals" rule);
  no summarizer.
- **Floors.** No analytic floor for free recall. `extras.null_top1_near` is the magnitude-matched
  decoy null: how often the top value lands within ±3 of an intermediate without being
  one (the item's start and answer are not decoys), weighted by (number of intermediates) / (size of the decoy set), so it is on the same
  scale as the pass rate per layer. The prompt-only baseline is reported beside it but is not
  item-blind here (the stock model can compute the chain). `extras.any_of_3_rate` (some named
  value is an intermediate) and `extras.committed_rate` (calls naming anything) are diagnostics;
  `extras.per_depth` splits the pass rate by chain depth.
- An item with no hit and an unjudged or missing cell is undecided and leaves the denominator;
  an empty cell is a negative. Rows at positions other than the last token per (item, layer)
  are ignored and counted (`extras.n_rows_not_last_token`). Missing cells are fatal (exit 2)
  unless `allow_missing=True`.

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}` rows over the plan's positions
(" What" through the end of the render, prompt positions only; the max-pos row per (item, layer)
is taken as the last token — the render's final `\n\n`, not the user turn's last token. The source
gen dirs label the seed-last prompts `<name>-last`; strip the suffix when converting,
`sed 's/-last"/"/'`). `examples/readouts/chain_intermediates.jsonl` is a toy file.

```
wsbench judge family=chain_intermediates readouts=examples/readouts/chain_intermediates.jsonl out=outputs/toy/chain dry_run=True
wsbench judge family=chain_intermediates readouts=<arm>.jsonl out=outputs/<arm>/chain_intermediates limit=3
```

## Judge prompts

`prompt_version`: `chain-free-2026-09-16`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. Output schema: `{"states_value": bool, "values": [int, ...], "basis":
"arithmetic" | "stated_result" | "numeral_bag" | "none", "quote": str}`.

`SYSTEM`:

```text
You are reading the output of a "lens" that tries to verbalize what a language model is
computing internally. The model is silently solving a short arithmetic chain: a starting number
and two or three dependent steps, answered with no chain of thought. You do NOT get to see the
chain — only the lens output.

The lens output may be free text, bullet points, or a bag of loose tokens, possibly in several
languages (Chinese numerals like 十四 mean fourteen). It is often vague, generic, or about
something else entirely, and it frequently contains numbers that are not results of anything.

Your job: name the number(s) that this output presents as a COMPUTED VALUE — a result, an
intermediate, a total, a "the answer is X". Rank them, most clearly a computed result first,
at most three.

Rules:
- Do NOT list numbers that appear as operands, quantities, labels, list indices, years, counts
  of items, or parts of unrelated prose. Only numbers the output presents as something that was
  worked out.
- If the output states arithmetic with both operands and a result ("16 - 2 = 14"), list the
  RESULT (14), not the operands.
- A bag of loose numerals with no arithmetic around them still counts: if the bag is dominated
  by one value or a tight cluster, list what it points at, most frequent or most prominent first.
- If the output presents no number as a computed value, set states_value false and return an
  empty list. That is a normal and common answer — do not invent one.
- Never do any arithmetic of your own. You do not know the task, so you cannot know what the
  right answer is; only report what the text itself puts forward.
```

`USER`:

```text
Lens output:
<<<
{readout}
>>>

Which number(s) does this output present as a computed value?
```
