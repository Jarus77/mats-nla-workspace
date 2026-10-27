# Exp 21: does the state change the text that follows? A text-only predictor of Oracle Lens exhibition (exploratory, negative)

Exp 20 left one candidate predictor for how much Oracle Lens's readout exhibits a state: whether the state changes the
continuation of the text. This measures it with no model: for every read point, the true next tokens of the source
(re-located in the same local files; the table-column row's continuation regenerated from exp 13's cell pools, labelled
synthetic), a bag-of-tokens classifier over the first n continuation tokens, grouped 5-fold by source file, AUROC.
Scored against exp 20's Oracle Lens readout column. Laptop only; `build21.py`, `lib21.py` (tested).

**Status: exploratory and post hoc.** The Oracle Lens numbers had been seen before this was run, so this is a fit,
not a prediction; it was meant to pick the operationalisation to pre-register on the public-source rebuild. It did not
find one.

## Result (`results/table21.md`)

| row | next 4 tokens | next 8 | next 32 | Oracle Lens readout (exp 20) |
|---|---:|---:|---:|---:|
| table column (synthetic continuation) | 0.85 | 1.00 | 0.98 | 0.64 |
| Python string | 0.90 | 0.93 | 0.64 | 0.91 |
| JavaScript string | 0.90 | 0.85 | 0.64 | 0.90 |
| LaTeX math | 0.92 | 0.85 | 0.55 | 0.87 |
| call argument | 0.89 | 0.86 | 0.60 | 0.70 |
| JSON depth | 0.94 | 0.97 | 0.98 | 0.94 |
| HTML depth | 0.97 | 0.99 | 1.00 | 0.79 |
| Markdown heading | 0.57 | 0.53 | 0.57 | 0.59 |
| Markdown list | 0.61 | 0.72 | 0.72 | 0.62 |

Spearman with the Oracle Lens readout: +0.78 at 4 tokens over nine rows, but +0.54 over the seven readable rows
(n = 7, not significant) and near zero or negative at every other length. The nine-row number is carried by the two
Markdown rows, which are at chance for every reader because the state is not readable, not because the continuation
does not change.

## Reading

1. **The continuation does change with the state on every readable row** (0.85 to 0.97 at four tokens), including
   table column, where Oracle Lens exhibits it weakly (0.64). So "the lens exhibits what continuing the text makes
   visible" is wrong as stated: the continuation after a first cell is separable from one after a late cell, and the
   lens's readout still carries little of it.
2. **Length matters in a way a bag of tokens cannot use.** String and math states are visible in the next two to eight
   tokens and wash out by 32; depth states are visible at every length. The lens's readout is 1 to 3 short bullets, so
   if a continuation predictor exists it is about the first few tokens, which is where the weak +0.54 sits.
3. **No predictor of exhibition is pre-registered from this.** The honest statement for the write-up is that what
   Oracle Lens exhibits is bounded by what is readable and by what the window carries (exp 20) and is otherwise not
   yet predicted; the contrast that any future predictor must explain is string state 0.91 against table column 0.64
   with both fully present in the window and both separable in the continuation.

Cost: none (laptop).
