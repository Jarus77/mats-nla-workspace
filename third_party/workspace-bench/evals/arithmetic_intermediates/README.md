# arithmetic_intermediates

`Compute (271 - 322) * 14. Reply with only the final number, nothing else.` The model answers
with no chain of thought; the intermediate (−51) is never written. Each variant fixes one
expression shape and one frozen (layer, position) cell, pre-registered from a sweep in the
source repo. **Since 2026-09-23 the readouts of record carry every position of the rendered
prompt at the two pre-registered layers of record, 56 and 60 (42 for the NLA and SAE arms, their
only layer), and the run of record is judged with `opts=cells=all`, one batched call per item and
layer** — Camila: "new readouts for arithmetic (all tokens not just the equals sign)" and, at 06:14
UTC, "for user modeling and arithmetic let's just batch it — feed it everything at once!"; keeping
the layers at 56/60 rather than
widening to the ladder is the orchestrator's decision (Camila specified the tokens, not the
layers). Until then one frozen cell per variant was read and judged (`cells=frozen`, still the
judge default and the pre-registered comparison point). `wsbench plan` emits the regime of
record: chat render, every position, layers 56 and 60, with each item's frozen cell kept in
`extra`. Can a lens assert the intermediate anywhere in the prompt?

596 items across 14 variants (the source bank's `<variant>.json` files, items unchanged, each
carrying its variant's cell and role; 10 variants were dropped by Agam on 2026-09-16: addmulx,
dec16, halflead, halftrail, halves, maxsel, mulmul, negdec, negdiv8, negdiv8x). Every item was
gated on Qwen3.6-27B (≥ 8/10 sampled correct, no leak: no intermediate appears as a numeral in
the prompt). Bare prompt (no system prompt, no few-shot) through the chat template with an empty
think block — `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`,
34–42 tokens; the frozen cells −7 / −8 are the assistant-turn `<|im_start|>` and the newline after
`<|im_end|>`.

| variant | shape | items | tolerance | cell | role |
|---|---|---|---|---|---|
| `absval` | `abs(a - b) * c` | 25 | rel5pct | L56, -7 | structural |
| `addmul` | `(a + b) * c` | 43 | rel5pct | L60, -8 | comparison |
| `floordiv` | `floor(a / b) + c` | 21 | exact | L60, -7 | structural |
| `frac` | `(p / q) * (p*q*k)` | 116 | rel5pct | L60, -8 | structural |
| `fracadd` | `(a / b) + (c / b),  b odd, b \| (a+c)` | 63 | rel5pct | L56, -8 | structural |
| `fraccomp` | `((a + b) / c) * (c*e)` | 44 | rel5pct | L56, -8 | structural |
| `fracint` | `(a / b) + (c / b),  b \| a, b \| c` | 32 | rel5pct | L56, -8 | comparison |
| `fracsmall` | `(a / b) + (c / b),  a,c single-digit, b \| (a+c)` | 16 | rel5pct | L56, -8 | comparison |
| `muladd` | `(a * b) + c` | 35 | rel5pct | L60, -8 | structural |
| `mulmid` | `(a * b) + (c * d),  products 60-100` | 33 | rel5pct | L56, -8 | comparison |
| `sign` | `(a - b) * c` | 25 | exact | L56, -8 | comparison |
| `signpair` | `(a - b) * c` | 23 | rel5pct | L56, -8 | structural |
| `subsub` | `(a - b) - c` | 60 | rel5pct | L56, -8 | comparison |
| `subsubx` | `a - (b - c)` | 60 | rel5pct | L56, -8 | comparison |

`role` is the source's reading: a *structural* variant has no single-token route to the
intermediate and can support a claim against token lenses; a *comparison* variant is
single-token reachable and is reported, not averaged into structural claims.

Example: `(271 - 322) * 14` → intermediate -51, answer -714.

## Scoring

- **Judge: free recall.** Prompt-blind. Under `cells=frozen` one call per item on the variant's
  frozen cell; under `cells=all` **one call per (item, layer)** that lists every non-empty position
  of that layer, in position order, as numbered entries `[k] pos=<offset> token=<token>: <readout>`
  (a token lens's scored bag as `tokens: ...`) and is answered per entry, under the same
  instructions. For each cell the judge names the numbers the readout presents as computed values, ranked, at most
  three, or none (the chained-intermediates judge with its task sentence changed to a bare
  arithmetic expression). A named value is kept only when the readout writes it (any numeral to
  six decimals, list markers and step labels excluded) or, for the top-ranked value only, the
  judge's quote is verbatim, contains a numeral and is not itself a bare number (a Chinese
  numeral is credited that way). No regex or numeric matcher scores
  anything: the numerals are read only to verify what the judge named.
- **Pass** (`value` in the source's terms): some kept value lies within the variant's tolerance
  of the headline intermediate (`exact`, or a relative 5%; the source's `tolerance_ok`, sign-
  aware). `extras.top1_rate` is the stricter top-ranked-value rate. Every rule is applied per cell
  (per entry of a batched call) exactly as for a single-cell call: the named value must be written
  in THAT entry's readout.
- **Floors.** No analytic floor. `cross` is the permutation null: the same rule against the
  intermediates of the item's `null_set` (the other items of its variant whose intermediates the
  source kept apart by about four 2% tolerances; the 13 pairs that sit within the 5% used since
  2026-09-23 were dropped from the null sets),
  averaged over that set; `net = value −
  cross`. `extras.per_variant` gives n / value / cross / net / tolerance / role per variant,
  `extras.per_role` the structural vs comparison split; `extras.committed_rate` the calls that
  named anything. The prompt-only baseline is not item-blind here (the stock model computes the
  expression); its frozen entry is stamped `arith-free-2026-09-16`, stale against the current
  prompt version, so `report` does not draw it until it is re-measured.
- **Every-position read (`opts=cells=all`, the run of record since 2026-09-23).** The same
  judge over every (layer, position) row in the file, packaged as one call per (item, layer)
  (`arith-free-batched-2026-09-23`, below); an item passes when ANY cell names a value within
  its tolerance, and the permutation null is taken the same way, so `net` stays comparable. An
  entry the reply skips, repeats or malforms is an unjudged cell (`rows[*].unjudged_at`,
  `extras.n_calls_failed` counts whole calls that failed); a reply with no usable entry is
  re-queued on the next run. `extras.bands` reports the any-cell accuracy at two fixed bands beside the
  variant's own tolerance: `exact` and within 5% (`rel5pct`) of the intermediate. `extras.cell_hit_rate` and
  `extras.per_layer_hit_rate` are the per-cell numbers, and each row lists `hits_at`. An
  any-of-grid rule over hundreds of cells inflates the pass rate for that reason alone: read it
  against `cross`, never alone.
- One cell per item (the default, `cells=frozen`): rows at other layers or positions are ignored and counted
  (`extras.n_rows_off_cell`); `layers=L` overrides the layer for every item. An empty cell is a
  negative; an unjudged cell leaves its item undecided. Missing cells are fatal (exit 2) unless
  `allow_missing=True`.

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}`, one row per (layer, position)
cell, `pos` the offset from the end of the rendered prompt as in the bank (−1 the final `\n\n`,
`cell.pos` −7 / −8 the frozen cells), so a file written for `cells=all` serves `cells=frozen` too.
- **Regime of record since 2026-09-23:** every position of the render (34–42 per item) at layers
  56 and 60 — 44,044 cells over the 596 items — judged with `opts=cells=all` (one batched call per
  item and layer, ≈ 1.2k calls per arm); the NLA and SAE arms
  carry layer 42 only (22,022 cells) and are judged with `layers=42`, an off-layer diagnostic that
  is `complete=False` by contract. k = 1 sample per cell for prose lenses.
- **Until 2026-09-23:** one row per item at the variant's frozen cell (layer 56 or 60, −7 or −8),
  judged `cells=frozen`; those numbers are still reproducible from an all-position file with the
  default `cells=frozen`.

Token lenses are judged as their bag and verified against the tokens.
`examples/readouts/arithmetic_intermediates.jsonl` holds a few real O-lens rows of four items at
several positions of both layers, frozen cells included (one empty cell among them).

```
wsbench judge family=arithmetic_intermediates readouts=examples/readouts/arithmetic_intermediates.jsonl out=outputs/toy/arith dry_run=True
wsbench judge family=arithmetic_intermediates readouts=<arm>.jsonl out=outputs/<arm>/arithmetic_intermediates opts=cells=all   # the run of record
wsbench judge family=arithmetic_intermediates readouts=<arm>.jsonl out=outputs/<arm>/arithmetic_intermediates-frozen           # the pre-2026-09-23 frozen cell
```

## Judge prompts

`prompt_version`: `arith-free-2026-09-23` for the per-cell call (`cells=frozen`) and
`arith-free-batched-2026-09-23` for the batched `cells=all` call — the two share their
instructions and differ only in packaging, so they carry their own tags. Judge: the repo default
(`google/gemini-3.8-flash`), temperature 0. Per-cell output schema: `{"states_value": bool,
"values": [number, ...], "basis": "arithmetic" | "stated_result" | "numeral_bag" | "none",
"quote": str}`.

`SYSTEM`:

```text
You are reading the output of a "lens" that tries to verbalize what a language model is
computing internally. The model is silently evaluating a short arithmetic expression: two or
three operations on given numbers, answered with no chain of thought. You do NOT get to see the
expression — only the lens output.

The lens output may be free text, bullet points, or a bag of loose tokens, possibly in several
languages (Chinese numerals like 十四 mean fourteen). It is often vague, generic, or about
something else entirely, and it frequently contains numbers that are not results of anything.

Your job: name the number(s) that this output presents as a COMPUTED VALUE — a result, an
intermediate, a total, a "the answer is X". Rank them, most clearly a computed result first,
at most three. Decimals and negative numbers are values too; report them exactly as written.

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

### Batched `cells=all` call — `arith-free-batched-2026-09-23`

Camila (2026-09-23 06:14 UTC): "for user modeling and arithmetic let's just batch it — feed it
everything at once!" One call per (item, layer) lists every non-empty position of that layer, in
position order, as entries `[k] pos=<offset> token=<JSON-quoted token>: <readout>` separated by
blank lines (a token lens's bag as `tokens: tok (score) | ...`), and the judge answers one result
per entry. Output schema: `{"entries": [{"k": int, "values": [number, ...], "basis": "arithmetic" |
"stated_result" | "numeral_bag" | "none", "quote": str}, ...]}`. From 2026-09-23 02:00 to 06:14
UTC `cells=all` was one call per cell with the per-cell prompt above (≈ 407k calls per full run;
never completed).

`SYSTEM_BATCH`:

```text
You are reading the outputs of a "lens" that tries to verbalize what a language model is
computing internally. The model is silently evaluating a short arithmetic expression: two or
three operations on given numbers, answered with no chain of thought. You do NOT get to see the
expression — only the lens outputs.

You are given several lens outputs at once, one per position the lens was read at, as numbered
entries "[k] pos=<offset from the end of the prompt> token=<the token at that position>:" followed
by that entry's output (a bag of loose tokens is written as "tokens: ..."). Judge every entry on
its own; the entries do not explain each other, and an entry's header is not part of its output.

A lens output may be free text, bullet points, or a bag of loose tokens, possibly in several
languages (Chinese numerals like 十四 mean fourteen). It is often vague, generic, or about
something else entirely, and it frequently contains numbers that are not results of anything.

Your job, for EACH entry: name the number(s) that its output presents as a COMPUTED VALUE — a
result, an intermediate, a total, a "the answer is X". Rank them, most clearly a computed result
first, at most three. Decimals and negative numbers are values too; report them exactly as written.

Rules, applied to every entry on its own:
- Do NOT list numbers that appear as operands, quantities, labels, list indices, years, counts
  of items, or parts of unrelated prose. Only numbers the output presents as something that was
  worked out.
- If the output states arithmetic with both operands and a result ("16 - 2 = 14"), list the
  RESULT (14), not the operands.
- A bag of loose numerals with no arithmetic around them still counts: if the bag is dominated
  by one value or a tight cluster, list what it points at, most frequent or most prominent first.
- If an output presents no number as a computed value, return an empty list for that entry. That
  is a normal and common answer — do not invent one.
- Never do any arithmetic of your own. You do not know the task, so you cannot know what the
  right answer is; only report what each text itself puts forward.
- Answer with exactly one result per entry, carrying the entry's k, in the order given; the
  quote is the shortest verbatim span of THAT entry supporting its top value (empty if none).
```

`USER_BATCH`:

```text
Lens outputs, one per read position:

{entries}

For each entry, which number(s) does its output present as a computed value?
```
