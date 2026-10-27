# Exp 27: random read points at 2,000 points, with a blind judge on the in-string readouts

Question: exp 23 found that neither reader names the structural state at a random read point (Oracle Lens 0 of 500,
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

## Results

(pending: `run27.sh` writes `results/table27.md`, `results/table27.json`, `results/judge27.json`)
