# Exp 23: naming and carrying base rates at random read points (Neel's "random prompts" check)

Question (Neel, 27 Oct): the forced-choice banks select read points by state and prompt the readers with the state in
mind; does anything we report survive at random read points in random documents, where nothing is prompted and the
readers are read once each? Two base rates: how often does a reader *name* the structural state the cursor is in, and
how far does its unprompted readout *carry* that state (a classifier on the readout text alone).

Pipeline: `build23.py` (laptop) draws 100 random read points per kind (Python, JavaScript, JSON, LaTeX, Markdown) from
random public documents (package directories on the build machine and arXiv e-prints), 150 to 1,200 characters of
context, read point just after a word character, seed 2323; state labels (`lib23.labels`) come from the prefix alone,
the same definitions as the exp 19 generators. Bank sha256
`edfb346fc5603fe9f68ab436139ae824dc70e5ecdba2e87d9627e8485e275be0`, 500 items, tracked. `modal23.py` (app `w2gd`)
captures layer 42 and runs Oracle Lens (k=1, layer 42, no prompt) and the NLA (raw explanation, k=1) once per item.
`analyze23.py` computes the tables on the laptop. Cost about $4 GPU.

## Naming (`results/table23.md`)

Strict, state-specific phrasing (`lib23.names_state`: "inside a string literal", "in math mode", "argument list",
"list item", "heading", and so on): share of readouts naming the state when it holds and when it does not.

| state | n holds / not | Oracle Lens: when holds | when not | NLA: when holds | when not |
|---|---|---:|---:|---:|---:|
| in a string | 134 / 366 | 0.00 | 0.00 | 0.03 | 0.01 |
| in math | 8 / 492 | 0.00 | 0.00 | 0.00 | 0.01 |
| inside parentheses | 85 / 415 | 0.00 | 0.00 | 0.13 | 0.01 |
| inside braces (depth ≥ 2) | 46 / 454 | 0.00 | 0.00 | 0.04 | 0.00 |
| in a list item | 18 / 482 | 0.00 | 0.01 | 0.50 | 0.09 |
| in a heading | 4 / 496 | 0.00 | 0.00 | 0.00 | 0.00 |

A loose count of any structural vocabulary at all (the earlier `MENTION_ANY` regex, which matches "open", "math",
"level") puts the NLA at 0.43 to 0.60 and Oracle Lens at 0.09 to 0.27 per kind; those numbers are content words, not
statements about the read point, and are kept in the table file only for the record.

## Carrying (AUROC; readable = probe on the layer-42 activation; readout = bag-of-tokens classifier on the readout text alone, grouped by file)

| pool | state | n yes / no | readable | Oracle Lens readout | NLA readout |
|---|---|---|---:|---:|---:|
| JavaScript | inside parentheses | 38 / 62 | 0.67 | 0.59 | 0.64 |
| JavaScript | inside braces | 22 / 78 | 0.58 | 0.62 | 0.57 |
| JSON | inside braces | 22 / 78 | 0.49 | 0.53 | 0.47 |
| Python | inside parentheses | 27 / 73 | 0.66 | 0.54 | 0.46 |
| code (py + js) | in a string | 35 / 165 | 0.86 | 0.72 | 0.79 |
| code (py + js) | inside parentheses | 65 / 135 | 0.69 | 0.54 | 0.57 |
| code (py + js) | inside braces | 24 / 176 | 0.69 | 0.66 | 0.68 |
| all kinds | in a string | 134 / 366 | 0.96 | 0.84 | 0.92 |
| all kinds | inside parentheses | 85 / 415 | 0.76 | 0.67 | 0.68 |
| all kinds | inside braces | 46 / 454 | 0.73 | 0.67 | 0.71 |

The "all kinds" string row is confounded with document kind (99 of the 100 JSON points are inside a string, none of
the LaTeX or Markdown points are), so its 0.96 / 0.84 / 0.92 is mostly JSON-versus-not; the code-only pool (0.86 /
0.72 / 0.79, 35 positives) is the honest number. Math, list and heading states have too few positives (8, 18, 4) to
score.

## Reading

1. **Neither reader names the structural state at a random read point.** Oracle Lens: 0 of 500 readouts name any of
   the six states. The NLA names the state in 3% of in-string points, 13% of in-parentheses points and 4% of in-brace
   points, against 0 to 1% when the state does not hold; the one state it names often is "list item" (9 of 18,
   against 9% elsewhere), which is also the one state that is a visible token at the line start rather than a count
   over the prefix. This is the base rate behind the forced-choice "states" cells of exps 18, 19 and 22, which were
   low on every natural row: it is not an artefact of the contrastive banks.
2. **The unprompted readouts still carry the state, weakly, at random points.** In code, the string state is readable
   at 0.86 from the activation and recoverable at 0.72 (Oracle Lens) and 0.79 (NLA) from the readout text alone, on a
   random 35-versus-165 split with no prompting. Parenthesis and brace depth are only weakly readable at a random
   point (0.67 to 0.69; the contrastive banks pick clean deep-versus-shallow pairs) and the readouts carry about that
   much or less (0.54 to 0.68).
3. **The gap between readable and carried is small where the state is readable at all** (string: 0.86 against 0.72 /
   0.79), which is the exp 20 result at base rate: what the activation holds linearly, the free readout mostly keeps,
   and neither reader says it.
4. **What this does not show.** 100 points per kind and one readout per reader give wide intervals (a 0.72 AUROC on
   35 positives has a 95% interval of roughly ±0.10); the bag-of-tokens classifier is the only decorrelated readout
   measure here (no reconstructor); the strict naming regex can miss unusual phrasings, so the naming rates are lower
   bounds on a loose reading and the loose regex an upper bound.
