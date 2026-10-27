# Exp 28 table: 47 rows with a stated rate and an eight-paraphrase statement shift

Shift = ((rec(A) - rec(B)) . d_hat) / gap through the NLA critic at layer 42; `fixed shift` is the source experiment's fixed pair recomputed here, `source's fixed shift` is the value the source recorded (exp 14's NLA ruler for the exp 12/13 rows). span = frozen-probe AUROC of the critic's rebuild of the true 32-token span; readable = grouped 5-fold probe AUROC on the true activations.

| row | exp | type | stated | fixed shift | paraphrase mean [min, max] | SD | span | readable | used z | source's fixed shift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ij2_lang (language row) | 12 | self-inducing | 0.81 | 0.62 | 0.49 [0.39, 0.63] | 0.09 | 1.00 | 1.00 |  | 0.72 |
| ij2_lang (language row) | 18 | self-inducing | 0.95 | 0.51 | 0.36 [0.20, 0.55] | 0.10 | 1.00 | 0.99 |  | 0.51 |
| code_lang (language row) | 22 | self-inducing | 0.70 | 0.30 | 0.24 [0.19, 0.32] | 0.04 | 0.99 | 0.99 |  | 0.30 |
| bk_caps | 12 | self-inducing | 0.07 | 0.31 | 0.20 [0.14, 0.29] | 0.05 | 0.99 | 0.98 |  |  |
| bk_quote | 12 | structural | 0.14 | 0.32 | 0.19 [0.02, 0.24] | 0.07 | 0.66 | 0.72 |  |  |
| tb_sentcount | 13 | structural | 0.00 | 0.11 | 0.17 [0.05, 0.29] | 0.08 | 0.61 | 0.92 |  |  |
| bk_code (language row) | 12 | self-inducing | 0.82 | 0.28 | 0.16 [0.06, 0.23] | 0.06 | 1.00 | 0.98 |  |  |
| py_arg | 22 | structural | 0.02 | 0.21 | 0.15 [0.05, 0.23] | 0.07 | 0.88 | 0.95 | -1.2 | 0.21 |
| nj2_tense | 12 | self-inducing | 0.00 | 0.19 | 0.15 [0.12, 0.22] | 0.03 | 0.84 | 0.80 |  |  |
| json_depth | 22 | structural | 0.25 | 0.14 | 0.15 [0.09, 0.20] | 0.04 | 0.92 | 1.00 | -1.7 | 0.13 |
| bk_listidx (counter-row) | 12 | structural | 0.25 | 0.22 | 0.14 [0.08, 0.23] | 0.05 | 1.00 | 1.00 |  | 0.23 |
| nj3_predpos | 12 | structural | 0.31 | 0.21 | 0.14 [0.06, 0.22] | 0.05 | 0.74 | 0.75 |  |  |
| bk_quote | 18 | structural | 0.08 | 0.23 | 0.14 [0.07, 0.20] | 0.05 | 0.68 | 0.82 | 3.8 | 0.23 |
| json_depth | 19 | structural | 0.14 | 0.15 | 0.14 [0.08, 0.17] | 0.03 | 0.87 | 0.99 | -2.0 | 0.14 |
| ij_listnum | 12 | structural | 0.07 | 0.23 | 0.12 [0.06, 0.20] | 0.05 | 1.00 | 1.00 |  |  |
| py_arg | 19 | structural | 0.03 | 0.22 | 0.10 [0.03, 0.21] | 0.06 | 0.89 | 0.99 | 3.2 | 0.23 |
| agr_number | 12 | structural | 0.28 | 0.19 | 0.09 [0.04, 0.16] | 0.04 | 1.00 | 1.00 |  |  |
| agr_gender | 12 | structural | 0.09 | 0.14 | 0.08 [0.03, 0.17] | 0.05 | 1.00 | 1.00 |  |  |
| tb_codeline | 13 | structural | 0.00 | 0.05 | 0.06 [-0.00, 0.12] | 0.04 | 0.74 | 1.00 |  |  |
| csv_col | 18 | structural | 0.13 | 0.11 | 0.05 [0.01, 0.08] | 0.02 | 0.71 | 1.00 | 0.3 | 0.11 |
| tb_tablecol (counter-row) | 13 | structural | 0.41 | 0.07 | 0.05 [0.01, 0.09] | 0.02 | 0.95 | 1.00 |  | 0.07 |
| call_arg (counter-row) | 18 | structural | 0.33 | 0.08 | 0.05 [0.02, 0.09] | 0.02 | 1.00 | 1.00 | 9.1 | 0.09 |
| json_depth | 18 | structural | 0.25 | 0.14 | 0.04 [-0.03, 0.09] | 0.04 | 0.99 | 1.00 | 2.6 | 0.13 |
| bk_indent | 12 | structural | 0.00 | 0.04 | 0.04 [0.01, 0.07] | 0.03 | 1.00 | 1.00 |  |  |
| latex_math | 19 | structural | 0.15 | 0.05 | 0.04 [0.00, 0.07] | 0.03 | 0.83 | 0.99 | 25.6 | 0.05 |
| stance_word | 18 | structural | 0.00 | 0.14 | 0.04 [0.02, 0.07] | 0.02 | 0.46 | 0.45 |  | 0.14 |
| html_depth | 19 | structural | 0.00 | 0.04 | 0.04 [-0.03, 0.11] | 0.05 | 0.74 | 0.99 | -4.6 | 0.04 |
| tb_wordsleft | 13 | structural | 0.00 | 0.03 | 0.03 [-0.00, 0.07] | 0.03 | 0.47 | 0.64 |  |  |
| fmt_qa | 13 | structural | 0.04 | 0.01 | 0.03 [-0.01, 0.14] | 0.05 | 0.60 | 0.93 |  |  |
| md_list | 19 | structural | 0.04 | 0.06 | 0.03 [-0.03, 0.06] | 0.03 | 0.64 | 0.65 |  | 0.07 |
| html_depth | 18 | structural | 0.00 | 0.01 | 0.02 [0.01, 0.04] | 0.01 | 0.77 | 1.00 | -1.0 | 0.02 |
| fmt_listdepth | 13 | structural | 0.04 | 0.03 | 0.02 [0.01, 0.03] | 0.01 | 0.65 | 1.00 |  |  |
| md_heading | 22 | structural | 0.04 | 0.06 | 0.02 [-0.01, 0.07] | 0.03 | 0.62 | 0.72 |  | 0.06 |
| bk_wordpos | 12 | structural | 0.00 | 0.04 | 0.02 [-0.04, 0.08] | 0.04 | 0.82 | 0.88 |  |  |
| latex_math | 22 | structural | 0.09 | 0.02 | 0.01 [-0.04, 0.06] | 0.04 | 0.85 | 1.00 | 24.7 | 0.02 |
| py_string | 19 | structural | 0.19 | 0.04 | 0.01 [-0.01, 0.04] | 0.02 | 0.85 | 1.00 | 6.6 | 0.04 |
| nj1_col_break | 12 | structural | 0.00 | -0.01 | 0.01 [-0.02, 0.03] | 0.01 | 0.48 | 0.94 |  |  |
| latex_math | 18 | structural | 0.00 | 0.01 | 0.01 [-0.01, 0.02] | 0.01 | 1.00 | 1.00 | 22.1 | 0.01 |
| py_string | 18 | structural | 0.03 | 0.01 | 0.00 [-0.01, 0.01] | 0.01 | 0.98 | 1.00 | -12.2 | 0.01 |
| js_string | 22 | structural | 0.14 | 0.01 | 0.00 [-0.01, 0.03] | 0.01 | 0.77 | 0.98 | -2.0 | 0.01 |
| md_heading | 19 | structural | 0.04 | 0.03 | 0.00 [-0.06, 0.08] | 0.05 | 0.51 | 0.63 |  | 0.03 |
| js_string | 19 | structural | 0.08 | 0.02 | 0.00 [-0.03, 0.06] | 0.03 | 0.82 | 0.99 | 0.7 | 0.01 |
| md_heading | 18 | structural | 0.00 | -0.00 | -0.00 [-0.02, 0.01] | 0.01 | 0.68 | 1.00 |  | -0.01 |
| tb_paren (counter-row) | 13 | structural | 0.30 | -0.04 | -0.00 [-0.02, 0.03] | 0.02 | 0.74 | 0.90 |  | -0.05 |
| py_string | 22 | structural | 0.12 | -0.01 | -0.01 [-0.04, 0.02] | 0.02 | 0.84 | 1.00 | 10.7 | -0.02 |
| nj1_col | 12 | structural | 0.00 | -0.09 | -0.02 [-0.06, 0.01] | 0.02 | 0.44 | 0.67 |  |  |
| md_list | 22 | structural | 0.01 | -0.02 | -0.03 [-0.06, 0.00] | 0.02 | 0.50 | 0.49 |  | -0.03 |

### Correlations of the shift with the NLA stated rate (permutation p, two-sided)

| rows | n | Spearman | p | Pearson | p |
|---|---:|---:|---:|---:|---:|
| all rows, paraphrase-mean shift | 47 | 0.40 | 0.006 | 0.69 | 0.000 |
| all rows, fixed-pair shift | 47 | 0.47 | 0.001 | 0.68 | 0.000 |
| without the language rows | 43 | 0.23 | 0.136 | 0.19 | 0.224 |
| self-inducing rows only | 6 | 0.54 | 0.302 | 0.57 | 0.202 |
| structural rows only | 41 | 0.28 | 0.078 | 0.26 | 0.101 |

### Bands of the paraphrase-mean shift

| band | n | mean stated | min | max |
|---|---:|---:|---:|---:|
| < 0.10 | 32 | 0.09 | 0.00 | 0.41 |
| 0.10 to 0.25 | 13 | 0.22 | 0.00 | 0.82 |
| >= 0.25 | 2 | 0.88 | 0.81 | 0.95 |

AUROC of the shift for stated >= 0.30 (8 vs 39 rows): paraphrase mean 0.76, fixed pair 0.74.

### Binomial logistic fits of the stated rate

| model | params | log-lik | AIC | coefficients |
|---|---:|---:|---:|---|
| intercept only | 1 | -2679.2 | 5360.4 | -1.67 |
| shift | 2 | -1526.3 | 3056.5 | -2.73, 10.06 |
| type | 2 | -1683.2 | 3370.3 | -2.20, 2.44 |
| shift plus type | 3 | -1437.3 | 2880.5 | -2.63, 6.98, 1.08 |

AIC(type) - AIC(shift) = 313.8 (negative: the type indicator fits better).

Paraphrase spread: 8 of 47 rows have a min-to-max range over 0.15; median range 0.097, median SD 0.033.
