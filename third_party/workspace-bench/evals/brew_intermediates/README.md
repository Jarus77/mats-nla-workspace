# brew_intermediates

A potion changes colour by a ten-rule table each time it is stirred; the prompt lists the rules,
then "stir, stir", then the START colour last, and asks for the final colour with no chain of
thought. The colour after the FIRST stir is computed inside the read window and never written.
Can a lens name it, and does the readout name it more than the colours that were never on the
trajectory?

100 items (source bank `items_final.json`, items unchanged, each carrying its 24 pinned read
cells and their regions from the source's `prompts_full.json`): a palette of twelve
single-token colours, depth 2, every item gated on Qwen3.6-27B (greedy correct, 10/10 sampled
correct, no leak, no single-lookup shortcut). Chat render with the prefill `Answer:`.

Example: start `blue` → after one stir `green` (the gold) → answer
`black`; candidates ['red', 'black', 'green', 'purple', 'blue'].

## Read cells and regions

Every item has 24 pinned prompt positions, each labelled by region:

| region | cells | role |
|---|---|---|
| `stir` | 6 | the "stir, stir" line: the start colour has not been read yet, so nothing can be computed. The negative control. |
| `start` | 6 | the start-colour line |
| `question` | 9 | the question line |
| `emit_asst`, `emit_think`, `emit_colon` | 1 each | the newline after `assistant`, the `\n\n` after `</think>`, and the prefilled `Answer:` colon: where the model commits. The headline cells. |

By default (`opts=regions=headline`) the judge reads the emission and stir cells: 9 cells × 11
layers ≈ 10k calls per arm. `opts=regions=all` adds the start and question cells (24 cells,
≈ 26k calls). In the source repo's committed full-grid run (`results/brew_intermediates/metrics.json`,
uncorrected per-cell lifts) the emission cells carried the signal (s3d: gold named in 38% of
emission cells vs 20% for an off colour, lift 0.18) while the compute region's lift was 0.03
and the stir region's 0.02, so the cut keeps the headline and its control and drops only cells
with no lift; the option keeps the full grid one flag away.

## Scoring

- **Judge.** One prompt-blind multi-select call per non-empty cell: the readout and five
  candidate colours (the gold, the start, the answer and two off-trajectory table colours; the
  item's `options_adjacent`, order seeded per cell). The judge marks EVERY candidate the text
  names, in any language or spelling, plus a `primary` colour and a `basis` (diagnostics).
- **Screen.** A cell whose text has no colour word in any spelling (English words, Chinese
  colour characters) is recorded as naming nothing without a call (`extras.n_screened_cells`).
- **Headline: pass rate.** Over an item's emission cells at every layer, G = cells naming the
  gold and O_j = cells naming off colour j. The item passes iff `K·G > ΣO` (the gold is named in
  more cells than the mean off colour; ties fail), the source bundle's rule. Naming the start or
  the answer beside the gold is not penalised.
- **Floors.** No analytic floor. Per item, `null` is the same rule with each off colour in the
  gold's role (averaged) and `baseline` the fraction of the K+1 colours that clear the rule:
  what a lens with no idea which colour is the intermediate scores. `pass − null =
  (K+1)/K · (pass − baseline)`; compare the headline against the baseline. `extras.stir_pass_rate`
  is the same rule on the stir cells (should sit at the null), `extras.cell_gold_rate` /
  `cell_off_rate` the per-region cell rates, `cell_primary_gold_rate` the judge's single
  `primary` pick per cell, and the lucky-guessing baseline covers the option lists (it asks
  whether the five colours themselves give the intermediate away; its scale is one guess per
  cell, so `cell_primary_gold_rate` is what it compares with). The prompt-only baseline is NOT a
  floor here: the table and the start colour are in the prompt, so the stock model computes the
  intermediate (measured 0.96) — read it as what the text gives away and use `baseline` / `null`
  instead. The source repo's caveat stands:
  the intermediate is usually the model's runner-up answer colour, so this bank cannot separate
  a held step from a runner-up output.
- An item whose emission cells are all judged is decided (an empty cell names nothing); any
  unjudged or missing emission cell leaves it undecided, because a missing cell can flip the rule
  either way. Missing cells are fatal
  (exit 2) unless `allow_missing=True`.

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}` rows at the pinned positions
(`eval_positions` in the bank); rows at other positions are ignored. Token lenses are judged as
their bag. `examples/readouts/brew_intermediates.jsonl` is a toy file.

```
wsbench judge family=brew_intermediates readouts=examples/readouts/brew_intermediates.jsonl out=outputs/toy/brew dry_run=True
wsbench judge family=brew_intermediates readouts=<arm>.jsonl out=outputs/<arm>/brew_intermediates limit=4
wsbench judge family=brew_intermediates readouts=<arm>.jsonl out=outputs/<arm>/brew_all opts=regions=all
```

## Judge prompts

`prompt_version`: `brew-2026-09-16`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. Output schema: `{"named": [colour, ...], "primary": colour | "none", "basis":
"explicit" | "implied" | "none"}` over the palette ['gray', 'pink', 'blue', 'black', 'brown', 'white', 'red', 'green', 'purple', 'gold', 'orange', 'teal'].

`SYSTEM`:

```text
You read short texts produced by an interpretability tool and report which colours they mention. You are given a text and a list of candidate colours. You do not know the task the text came from, and you should not guess it. Report only what the text itself says.
```

`USER`:

```text
Text:
<readout>
{readout}
</readout>

Candidate colours: {options}

1. named: list EVERY candidate colour the text names or clearly refers to, in any language or
   spelling (e.g. a Chinese colour word, a plural, a capitalised form, a token fragment that is
   unambiguously that colour word). A colour the text mentions only as part of another word
   (e.g. "goldfinch", "redistribute") does not count. If none, return an empty list.
2. primary: the ONE candidate colour the text most clearly puts forward as the colour of the
   thing it describes, or "none" if it does not put one forward.
3. basis: "explicit" if the colour words appear plainly, "implied" if you had to interpret
   (e.g. a translation or fragment), "none" if no candidate is named.
```
