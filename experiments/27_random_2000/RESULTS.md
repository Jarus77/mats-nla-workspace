# Exp 27: random read points at 2,000 points, with a blind judge on the in-string readouts

Question: exp 23 found that neither reader names the structural state at a random read point (Oracle Lens never where the state holds, 0 of 500,
NLA 3 to 13% on string, parentheses and braces) while the unprompted readouts still carry the state weakly, but its
intervals were about plus or minus 0.10 and its strict naming regex could in principle miss unusual phrasings. Do the
naming and carrying base rates hold at four times the points and with intervals of about plus or minus 0.05, and does
a blind judge, shown the readout alone, find namings of the string state that the regex missed?

Pipeline: `build27.py` runs exp 23's builder (imported, not copied) at seed 2727 and 400 random read points per kind
(Python, JavaScript, JSON, LaTeX, Markdown; 2,000 in all) in random public documents (package directories on the
build machine and arXiv e-prints), the same context window (150 to 1,200 characters, read point just after a word
character) and the same state labels from the prefix (`lib23.labels`); `results/items_base.json` is tracked and its
sha256 is in `results/build_info.json`. `modal27.py` (app `f3qn`) captures layer 42 and runs Oracle Lens (k=1, layer
42, no prompt) and the NLA (raw explanation, k=1) once per item, as in exp 23. `analyze27.py` computes the exp 23
tables (strict naming with Wilson intervals, loose vocabulary for the record, carrying with the kind-confound caveat)
with exp 23's numbers side by side. `judge27.py` then shows a blind judge (Gemini 3.8 Flash via OpenRouter, T=0) the
readout text alone, never the document, for every in-string point (about 500) plus an equal number of matched
not-in-string points (seed 27), for both readers, and asks whether the readout states that the current position is
inside a string literal; the same pass covers exp 23's 134 in-string readouts plus 134 matched negatives, so exp 23's
open item closes here. `run27.sh` chains the steps; `analyze27.py` runs again to merge the judge table.

Cost: two GPU passes of about 25 minutes each (H100 for capture plus Oracle Lens, H200 for the NLA), about $6; judge
about 2,300 short calls, about $3.

Bank as built (27 Oct, 22:20 UTC, before any reader ran): 1,939 points, not 2,000: 400 each for Python, JavaScript, JSON and Markdown and 339 for LaTeX, because the public arXiv e-print pool on this machine has 339 `.tex` files over 4 KB and the builder takes one point per file. sha256 `e95bddd36db82014f7decc6b12e90f65163345e58e396e870be69c65dd7d8156`, seed 2727.

## Predictions written before the run

1. **Naming.** Strict naming rates within 0.05 of exp 23's: Oracle Lens 0.00 to 0.01 on every state; NLA in_string
   about 0.03, paren_depth about 0.13, brace_depth about 0.04, list_depth about 0.5 (on a larger n than exp 23's 18),
   with when-not rates at 0.00 to 0.01 (list_depth about 0.09). The Wilson intervals on the when-holds cells are now
   about plus or minus 0.05 (about plus or minus 0.02 for Oracle Lens at 0 of about 500).
2. **Judge.** The blind judge's yes-rate on in-string readouts is under 0.10 for Oracle Lens and under 0.20 for the
   NLA, against under 0.03 on the matched negatives, for exp 27 and for exp 23's 134 + 134: the judge does not find
   naming that the regex missed, so the regex rates stand as the naming rates and not as lower bounds in any
   material sense.
3. **Carrying.** As in exp 23: code (py + js) in_string text bag about 0.72 for Oracle Lens and about 0.79 for the
   NLA against the probe at about 0.86; parenthesis and brace depth weakly readable (0.6 to 0.7) and carried at about
   that or less; the all-kinds string row again confounded with document kind (JSON) and reported with the caveat.

## Results (27 Oct, 23:17 UTC; `results/table27.md`, `results/judge27.json`, `results/judged27.jsonl`; 1,939 points; two GPU passes of 19 and 16 minutes, about $5; judge 2,480 calls, $1.10)

**Strict naming (state-specific phrasing), share of readouts naming the state when it holds, with exp 23 beside:**

| state | n holds | Oracle Lens | NLA | NLA when the state does not hold | exp 23 NLA (n) |
|---|---:|---:|---:|---:|---:|
| in_string | 486 | 0.00 | 0.02 [0.01, 0.03] | 0.01 | 0.03 (134) |
| paren_depth | 318 | 0.00 | 0.08 [0.05, 0.11] | 0.01 | 0.13 (85) |
| brace_depth | 185 | 0.00 | 0.01 | 0.01 | 0.04 (46) |
| list_depth | 109 | 0.01 | 0.53 [0.44, 0.62] | 0.10 | 0.50 (18) |
| in_math | 22 | 0.00 | 0.09 | 0.01 | 0.00 (8) |
| heading_level | 15 | 0.00 | 0.20 | 0.01 | 0.00 (4) |

**Blind judge on the in-string readouts** (the judge sees only the readout and says whether it states that the
position is inside a string literal; every in-string point plus matched negatives, both banks pooled):

| reader | n in-string / not | judge yes when in string | when not | strict regex when in string |
|---|---|---:|---:|---:|
| Oracle Lens | 598 / 604 | 0.00 [0.00, 0.01] | 0.00 | 0.00 |
| NLA | 602 / 614 | 0.22 [0.18, 0.25] | 0.02 | 0.02 |

What the judge counts that the regex did not: the NLA's own wording, for example "a truncated word mid-quotation",
"a truncated default value string mid-entry", "within the quoted API description". The regex looked for "string
literal" phrasing and the NLA rarely uses it.

**Carrying** (AUROC; readable = probe on the activation; text bag = classifier on the readout text alone):

| pool | state | n yes / no | readable | Oracle Lens readout | NLA readout | exp 23 |
|---|---|---|---:|---:|---:|---|
| code (py + js) | in_string | 102 / 698 | 0.86 | 0.72 | 0.81 | 0.86 / 0.72 / 0.79 |
| code (py + js) | paren_depth | 227 / 573 | 0.75 | 0.60 | 0.61 | 0.69 / 0.54 / 0.57 |
| code (py + js) | brace_depth | 102 / 698 | 0.80 | 0.73 | 0.72 | 0.69 / 0.66 / 0.68 |
| md | list_depth | 109 / 291 | 0.73 | 0.73 | 0.83 | n/a |
| tex | in_math | 22 / 317 | 0.85 | 0.75 | 0.85 | n/a |

The "all kinds" rows (string 0.95 / 0.87 / 0.93) remain confounded with document kind and are in `table27.md` for
the record only.

### Scoring the predictions

| prediction | result |
|---|---|
| 1. strict naming within 0.05 of exp 23 | holds on every state (largest change paren_depth 0.13 to 0.08, inside the exp 23 interval) |
| 2. blind judge: Oracle Lens under 0.10, NLA under 0.20 when in string, under 0.03 on negatives | Oracle Lens holds (0.00); NLA misses by a hair (0.22 [0.18, 0.25]); negatives hold (0.02) |
| 3. carrying as in exp 23 (code in_string 0.86 / 0.72 / 0.79) | holds (0.86 / 0.72 / 0.81) |

## Reading

1. **The sample-size answer.** Four times the points reproduce exp 23's numbers within their old intervals, and the
   new intervals are about plus or minus 0.03 on the string state. Nothing in exp 23 was a small-sample accident.
2. **The regex under-counted the NLA.** A blind judge finds the NLA stating that the position is inside a string
   on about one readout in five when it is, against one in fifty when it is not; the strict regex found one in
   fifty either way. The NLA says it in its own words ("mid-quotation", "quoted"), not in ours. Exp 23's sentence
   "neither reader names structural state" must be read as: Oracle Lens never does when the state holds (0 of 598 by the
   judge; by the regex 1 of 1,939, a list item, and 16 further hits all on points where the named state does not
   hold); the NLA does so in a minority of cases, one in five for strings, half the
   time for list items, and almost never for depth.
3. **Carrying is unchanged** and still ordered as before: the activation, then the NLA's readout, then Oracle
   Lens's, with the code-only string row the honest number (0.86 / 0.81 / 0.72).
4. **What this changes for the account.** The NLA's partial naming at random points (string one in five, list
   items one in two) is more than the exp 19 and 22 "states" cells showed on their banks (0.02 to 0.25 by the same
   judge on 40 items x 8 samples), which is consistent with those banks' read points being chosen at the state's
   defining token rather than at random. The two-type picture stands: Oracle Lens states nothing anywhere; the NLA
   states some structural facts some of the time, in a describing register, without being asked.
