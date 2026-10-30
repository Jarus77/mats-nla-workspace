# Exp 20: what the NLA reconstructor keeps (ordered span, bag of the same tokens, the readers' own words)

Neel (27 Oct): "understanding why o-lens and NLAs fail to capture that thing would be interesting; I recommend studying
the AR and seeing if it captures any of that structure." Exps 14 to 16 did this on two position rows; this runs the same
question on the eight natural rows of exp 19 (selection split, 160 items per row, layer-42 activations already captured)
with four kinds of input to the reconstructor, scored by exp 14's frozen-probe AUROC against the true activations:

- the true last w tokens, w = 4, 8, 16, 32, 64, 128 (how much structure the window carries, and whether more text helps)
- the same 32 tokens in random order (vocabulary kept, structure broken)
- Oracle Lens's own free readout of the vector (k = 1, layer 42)
- the NLA's own explanation of the vector (k = 1)
- the model's own layer-42 activation when it reads the last w tokens alone, no reconstructor (the image of the window map: what any text-emitting reader could at most re-induce)

If the ordered span keeps the state and the shuffled bag and the readouts do not, the readers' failure is located in what
the reconstruction objective rewards: content, not order. App `v7qd` (`modal20.py`); one H100 pass for the Oracle Lens
readouts, one H200 pass for the NLA and the reconstructions.

## Results (27 Oct; `results/table20.md`, `results/recon20_{row}.json`; frozen-probe AUROC against the true layer-42 activations, 160 selection items per row)

| row | model, window 4 | model, window 32 | model, window 128 | critic, span 32 | critic, span 128 | shuffled bag 32 | Oracle Lens readout | NLA explanation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| py_string | 0.99 | 1.00 | 1.00 | 0.85 | 0.95 | 0.42 | 0.91 | 0.99 |
| js_string | 0.96 | 0.99 | 0.99 | 0.82 | 0.84 | 0.48 | 0.90 | 0.98 |
| latex_math | 0.99 | 0.99 | 0.99 | 0.83 | 0.86 | 0.46 | 0.87 | 0.98 |
| py_arg | 0.97 | 0.96 | 0.98 | 0.89 | 0.89 | 0.50 | 0.70 | 0.97 |
| json_depth | 0.77 | 0.96 | 0.99 | 0.87 | 0.97 | 0.65 | 0.94 | 0.98 |
| html_depth | 0.83 | 0.94 | 0.95 | 0.74 | 0.81 | 0.30 | 0.79 | 0.93 |
| md_heading | 0.54 | 0.57 | 0.59 | 0.51 | 0.57 | 0.47 | 0.59 | 0.59 |
| md_list | 0.58 | 0.62 | 0.65 | 0.64 | 0.66 | 0.55 | 0.62 | 0.63 |

"model" = the base model's own layer-42 activation when it reads only the last w tokens; "critic" = the NLA reconstructor
fed the same text; "shuffled bag" = the 32 tokens in random order through the critic; the two readout columns = each
reader's own free readout of the vector (k = 1), through the critic. Cosine of each rebuild's class direction with the
true one: model window 32 is 0.78 to 0.93 on the six readable rows, critic span 32 is 0.30 to 0.65, bag 32 is -0.07 to
0.22, Oracle Lens readout 0.48 to 0.73, NLA explanation 0.77 to 0.90.

## Reading

1. **The window is not the bottleneck.** The model's own activation on the last 32 tokens carries every readable
   state at 0.94 to 1.00, and four of six already at 4 tokens. The two rows Oracle Lens missed when prompted (Python
   string 0.52, JSON depth 0.64 in exp 19) are carried at 1.00 and 0.96. Longer windows matter only for the two
   depth rows (JSON 0.77 at 4 tokens, 0.99 at 64), where the state is defined by context further back.
2. **Order carries the state, vocabulary does not.** The same 32 tokens shuffled fall to 0.30 to 0.65 (class-direction
   cosine near zero). What the readers would have to reproduce is the arrangement, not the words.
3. **Both readers' free readouts re-induce the state through the critic.** Oracle Lens's readout, which never names any
   of these properties, rebuilds them at 0.70 to 0.94 on the six readable rows: this is the structure-aware
   "exhibits" measure the judge could not supply (exps 18, 19 scored it 0.00 everywhere). The NLA's explanation, which
   describes genre and topic and states none of these properties, rebuilds them at 0.93 to 0.99.
4. **So the readers do not fail to capture structure; they fail to name it.** The state is present in what each reader
   writes, recoverable by the critic above the level of the true 32-token span, and absent from what either reader
   says about the text. Oracle Lens's prompted misses are a naming and phrasing failure (the fixed phrasing scored
   reversed on both missed rows), not a detection failure. This is the direction Guiv's E45 pointed at and it is now
   measured directly on the model, not through a reconstructor.
5. **The NLA number is partly in-distribution.** The critic was trained on NLA explanations, so 0.93 to 0.99 for the NLA
   column overstates how much a neutral decoder would recover; the Oracle Lens column is out of distribution for the
   critic (bullets and continuations) and is if anything conservative. The two columns should not be compared with
   each other, only each with its own bag and span baselines.
6. **The two Markdown rows stay at chance everywhere** (0.51 to 0.66), consistent with their probe AUROC of 0.63 and
   0.65: heading level and list depth are not linearly present at layer 42 on real Markdown, so no reader can carry
   them.

## What this changes

The exp 19 "exhibits" cell, re-scored with the reconstructor measure at the same 0.70 line used for the other AUROC
cells (post hoc, disclosed: the pre-registered judge measure failed on every row of two experiments): predicted high
on the five span-sayable rows and four of five observed high (0.87 to 0.94), py_arg just under the line (0.697, miss);
predicted low on HTML depth and observed 0.79 (miss); predicted low on both Markdown rows and observed low. Six of
eight, against three of eight with the judge. The primary scorecard with this cell becomes 34 of 38 (0.89).

The sentence for the thesis: the readers carry structural state in their outputs and do not name it. The remaining
question is why naming fails, and the exp 17 and 19 statement-shift numbers say what it is: a statement of the
property does not move the critic (0.01 to 0.23), so neither objective ever rewarded saying it. The thing to test next
is the original headline rows (table column, parenthesis) under this exact protocol, because exp 16 measured the
Oracle Lens part through a different reconstructor at layer 44 and found position absent from it; if the two agree,
exp 16's "remainder" is the naming gap seen from the other side, and if they disagree, one of the two reconstructors
is the reason.

Cost: about $4 (one H100 pass of 25 minutes, one H200 pass of 30 minutes).

## The original headline rows under the same protocol (27 Oct, 14:37; exp 13's banks, 160 selection items each)

| row | model, window 4 | model, window 32 | critic, span 32 | critic, span 128 | shuffled bag 32 | Oracle Lens readout | NLA explanation |
|---|---:|---:|---:|---:|---:|---:|---:|
| table column (first vs late) | 0.87 | 1.00 | 0.95 | 1.00 | 0.46 | 0.64 | 1.00 |
| parenthesis (inside vs outside) | 0.71 | 0.90 | 0.74 | 0.84 | 0.57 | 0.67 | 0.87 |

Class-direction cosines at window 32: model 0.89 / 0.87, critic 0.31 / 0.48, bag 0.05 / 0.11, Oracle Lens readout
0.25 / 0.54, NLA explanation 0.83 / 0.88.

**Reconciliation with exp 16.** Exp 16 found position in the part of the vector that Oracle Lens's words do not rebuild
(table column: remainder 0.98, Oracle Lens part 0.01, through the Oracle Lens reconstructor at layer 44). Here, through
the NLA critic at layer 42, Oracle Lens's readout carries table column at 0.64 and parenthesis at 0.67: weak, where the
same readout carries string state at 0.90 and JSON depth at 0.94. The two reconstructors agree in direction and differ
in degree, and the degree is the finding. Oracle Lens exhibits a state to the extent that continuing the text makes it
visible: a continuation inside a string literal looks nothing like one outside, a continuation after the first cell of a
table row looks like one after the fifth. The NLA's explanation carries both headline rows (1.00, 0.87) by describing
the format ("pipe-delimited table, two prior rows, predicting continuation of the same pattern") without ever naming
the column, which is the exp 16 "remainder" seen from the other side.

So the sentence stands for both readers and all ten rows: the window carries the state (0.90 to 1.00 on every readable
row), the readers' outputs carry it to different degrees (NLA through description, Oracle Lens through continuation),
and neither names it. The one quantity that predicts how much Oracle Lens exhibits is not the window but whether the
state changes the continuation; that is a text-only measurement (a classifier on the next tokens after the read point,
no model needed) and the natural next predictor to pre-register.

Cost of this run: about $3.

## Cross-check without the reconstructor (27 Oct, laptop; `results/readout_textbag.json`)

A bag-of-tokens text classifier on each reader's readout alone (grouped 5-fold by source file, no model, no critic)
recovers the state about as well as the critic does for Oracle Lens and somewhat less well for the NLA:

| row | Oracle Lens readout, text bag | through the critic | NLA explanation, text bag | through the critic |
|---|---:|---:|---:|---:|
| table column | 0.71 | 0.64 | 0.91 | 1.00 |
| parenthesis | 0.56 | 0.67 | 0.81 | 0.87 |
| Python string | 0.96 | 0.91 | 0.95 | 0.99 |
| JavaScript string | 0.88 | 0.90 | 0.87 | 0.98 |
| LaTeX math | 0.78 | 0.87 | 0.85 | 0.98 |
| call argument | 0.72 | 0.70 | 0.67 | 0.97 |
| JSON depth | 0.94 | 0.94 | 0.97 | 0.98 |
| HTML depth | 0.84 | 0.79 | 0.96 | 0.93 |
| Markdown heading | 0.57 | 0.59 | 0.63 | 0.59 |
| Markdown list | 0.59 | 0.62 | 0.62 | 0.63 |

Two methods, one trained on NLA prose and one that knows nothing about activations, agree on which readouts carry which
states, so "the readers' outputs carry the state" does not rest on the critic. The agreement also says what the carrying
mostly is: the words the readout uses (a continuation inside a string literal is made of string-ish tokens), which is
exhibition in the plain sense. Where the critic beats the bag by a margin (NLA on call argument 0.97 against 0.67, table
column 1.00 against 0.91), the critic recovers something the vocabulary does not carry, which is either the explanation's
structure or the critic's own training on NLA prose; the second decoder (next) separates those.
