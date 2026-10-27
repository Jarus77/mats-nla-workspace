# Exp 25 scorecard: exp 22 with `ol_exhibits` observed by the reconstructor measure (exp 20's, line 0.70)

Every other cell, prediction and threshold is exp 22's (`score22.json`); only the `ol_exhibits` column changes.

**Accuracy over 56 scored cells: 0.82 [0.70, 0.90]** (declared bar 0.80 over at least 36 cells).

| row | nla_prompted | nla_states | nla_names_complement | ol_prompted | ol_exhibits | ol_states | jlens_prompted |
|---|---|---|---|---|---|---|---|
| py_string | H / 0.88 ok | L / 0.12 ok | L / 0.00 ok | H / 0.91 ok | H / 0.86 ok | L / 0.00 ok | L / 0.56 ok |
| js_string | H / 0.90 ok | L / 0.14 ok | L / 0.00 ok | L / 0.87 MISS | L / 0.84 MISS | L / 0.00 ok | L / 0.66 ok |
| latex_math | H / 0.82 MISS | L / 0.09 ok | L / 0.40 MISS | H / 0.93 ok | H / 0.87 ok | L / 0.00 ok | L / 0.56 ok |
| py_arg | H / 0.65 MISS | L / 0.02 ok | H / 0.71 ok | H / 0.82 ok | H / 0.51 MISS | L / 0.00 ok | L / 0.56 ok |
| json_depth | H / 0.96 ok | L / 0.25 MISS | H / 0.68 ok | H / 0.88 ok | H / 0.97 ok | L / 0.01 ok | L / 0.79 MISS |
| md_heading | L / 0.76 ok | L / 0.04 ok | H / 1.00 ok | L / 0.58 ok | L / 0.65 ok | L / 0.00 ok | L / 0.53 ok |
| md_list | L / 0.69 ok | L / 0.01 ok | H / 0.00 MISS | L / 0.49 ok | L / 0.48 ok | L / 0.00 ok | L / 0.51 ok |
| code_lang | H / 1.00 ok | L / 0.70 MISS | H / 0.78 ok | H / 0.99 ok | H / 0.98 ok | L / 0.00 ok | H / 0.93 ok |

Confusion per reader (predicted x observed):

- nla: HH 8, HL 3, LH 3, LL 10
- ol: HH 9, HL 1, LH 2, LL 12
- jlens: HH 1, HL 0, LH 1, LL 6

**Primary (free-readout cells, J-lens on the six uncued rows): 32 of 38, 0.84 [0.70, 0.93]** (exp 22 with the judge-scored cell: 30 of 38, 0.79 [0.64, 0.89]).

**Secondary (prompted forced-choice cells): 13 of 16, 0.81 [0.57, 0.93]** (unchanged from exp 22).

All 56 cells, the exp 18 convention: 46 of 56, 0.82 [0.70, 0.90] (exp 22: 44 of 56, 0.79 [0.66, 0.87]).

With vs without code_lang: primary 32 of 38 vs 29 of 34; secondary 13 of 16 vs 11 of 14.

Per row, the `ol_exhibits` cell (judge exhibited-correct rate -> reconstructor AUROC of the Oracle Lens readout):

- py_string: judge 0.00 -> recon 0.86 (predicted high, ok)
- js_string: judge 0.00 -> recon 0.84 (predicted low, MISS)
- latex_math: judge 0.00 -> recon 0.87 (predicted high, ok)
- py_arg: judge 0.03 -> recon 0.51 (predicted high, MISS)
- json_depth: judge 0.00 -> recon 0.97 (predicted high, ok)
- md_heading: judge 0.00 -> recon 0.65 (predicted low, ok)
- md_list: judge 0.25 -> recon 0.48 (predicted low, ok)
- code_lang: judge 0.78 -> recon 0.98 (predicted high, ok)
