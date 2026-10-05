# Read sites for the association families (conjunctive, role-bound)

Since 2026-09-22 the readouts files of `conjunctive_association` and `role_bound_association` are
produced from a **summarize-prompt render**, read only at the prompt's suffix and the chat tail.
This page is the producer contract; the judges are unchanged.

## Why

Both families used to be judged over *every* position of the bare stimulus render (conjunctive:
one blob of all positions × layers; role-bound: any-of over a 17-position window). Under that
regime a lens that merely echoes the current token hands the judge the story itself, and the judge —
not the lens — does the interpretation. A position sweep on the s3d-rl600 O-lens (Gemini 3.8 Flash
judge) showed most of the score came from the chat tail rather than the story positions
(conjunctive: story-only blob 0.06, chat-tail-only 0.10, full 0.15), but the instrument permitted it. Appending a summarize
request gives the lens a read site with a fixed semantic job, and reading only there closes the
loophole.

## Render

```
<|im_start|>user
{stimulus}

Summarize the preceding story in one sentence.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

i.e. the chat template with `enable_thinking=False` (empty think block), no system prompt, no
prefill. Story positions are unaffected by a suffix that follows them (causal attention), so a
producer that already holds a bare-render capture only has to capture the new tail.

## Sites

The 19 positions from the first suffix token through the end of the render, at layers
20 / 28 / 36 / 44 / 52 / 60:

| rel (from end) | token | conjunctive single-site pass | role-bound per-cell pass |
|---|---|---|---|
| 18 | `Sum` | 0.00 | 0.000 |
| 17 | `mar` | 0.00 | 0.000 |
| 16 | `ize` | 0.06 | 0.104 |
| 15 | ` the` | 0.01 | 0.052 |
| 14–10 | ` preceding` ` story` ` in` ` one` ` sentence` | 0.00 | 0.000 |
| 9 | `.` | 0.09 | 0.086 |
| 8 | `<|im_end|>` | 0.00 | 0.002 |
| 7 | `\n` | 0.03 | 0.102 |
| 6 | `<|im_start|>` | 0.01 | 0.072 |
| 5 | `assistant` | 0.00 | 0.005 |
| 4 | `\n` | 0.07 | 0.088 |
| 3 | `<think>` | 0.00 | 0.000 |
| 2 | `\n\n` | 0.00 | 0.000 |
| 1 | `</think>` | 0.00 | 0.000 |
| 0 | `\n\n` | 0.07 | 0.042 |

(s3d-rl600, k=1, T=1.0; conjunctive = one 11-way MC over the site's six layer bags; role-bound =
all three MCs right in one (site, layer) cell.) The content sits at the newline / punctuation
positions *after* the specials; the special tokens themselves read as nothing. All 19 sites are
shipped so the judged grid is the same for every lens; the families' `layers=` / `items=` keys
still restrict.

## Numbers under the old and new regimes (s3d-rl600, Gemini 3.8 Flash)

| family | bare render, old sites | summarize render, 19 sites |
|---|---|---|
| role_bound_association (item = any cell) | 0.40 | 0.63 |
| conjunctive_association (one blob per item) | 0.15 | 0.14 |

Role-bound gains (best single cell 11 → 18 items passing / 100; agent/patient accuracy per cell
0.06–0.12 → 0.12–0.17); conjunctive is flat, with ~83 % "cannot tell" either way.

## Producing a readouts file

One row per (item, layer, site) in the readout contract of the [top-level README](../README.md), `pos`
indexing the render above, `token` the site's token, e.g.

```
{"id": "q01_a", "layer": 52, "pos": 68, "token": "\n\n", "samples": ["- After a long day, Laura packed boxes in the kitchen, meticulously clearing the house before…"]}
```

(The files under `examples/readouts/` are dry-run fixtures; `wsbench plan` is the read site of
record.)
