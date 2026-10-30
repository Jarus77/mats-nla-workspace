Checkpoints in training order: textnla, av_base, iter_000100, iter_000200, iter_000300, iter_000400, iter_000500, iter_000600, iter_000700, iter_000800. Text bag tokeniser: qwen.

### states

States (judge): share of the 8 mining readouts per item in which Gemini 3.8 Flash finds an explicit claim about the row's property (exp 22's cell)

| row | textnla | av_base | iter_000100 | iter_000200 | iter_000300 | iter_000400 | iter_000500 | iter_000600 | iter_000700 | iter_000800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| py_string | 0.47 | 0.21 | 0.21 | 0.12 | 0.24 | 0.13 | 0.23 | 0.18 | 0.21 | 0.19 |
| js_string | 0.49 | 0.19 | 0.21 | 0.29 | 0.21 | 0.16 | 0.20 | 0.17 | 0.16 | 0.24 |
| latex_math | 0.17 | 0.10 | 0.13 | 0.04 | 0.10 | 0.00 | 0.16 | 0.08 | 0.09 | 0.13 |
| py_arg | 0.24 | 0.12 | 0.08 | 0.04 | 0.08 | 0.02 | 0.01 | 0.02 | 0.06 | 0.03 |
| json_depth | 0.23 | 0.11 | 0.23 | 0.20 | 0.20 | 0.21 | 0.18 | 0.21 | 0.23 | 0.19 |
| md_heading | 0.30 | 0.02 | 0.05 | 0.04 | 0.03 | 0.03 | 0.00 | 0.03 | 0.02 | 0.03 |
| md_list | 0.19 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 | 0.01 | 0.01 | 0.00 |
| code_lang | 0.50 | 0.55 | 0.77 | 0.74 | 0.77 | 0.71 | 0.72 | 0.77 | 0.74 | 0.76 |

### names

Names (strict regex): share of the k=1 selection readouts in which the row's state regex fires (exp 23/26 phrasing)

| row | textnla | av_base | iter_000100 | iter_000200 | iter_000300 | iter_000400 | iter_000500 | iter_000600 | iter_000700 | iter_000800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| py_string | 0.39 | 0.08 | 0.03 | 0.09 | 0.05 | 0.03 | 0.06 | 0.08 | 0.06 | 0.04 |
| js_string | 0.37 | 0.12 | 0.12 | 0.06 | 0.05 | 0.06 | 0.06 | 0.05 | 0.08 | 0.01 |
| latex_math | 0.12 | 0.04 | 0.02 | 0.03 | 0.03 | 0.02 | 0.05 | 0.03 | 0.06 | 0.03 |
| py_arg | 0.18 | 0.12 | 0.08 | 0.07 | 0.09 | 0.03 | 0.01 | 0.06 | 0.08 | 0.07 |
| json_depth | 0.67 | 0.10 | 0.04 | 0.06 | 0.13 | 0.08 | 0.07 | 0.08 | 0.16 | 0.09 |
| md_heading | 0.43 | 0.42 | 0.26 | 0.46 | 0.32 | 0.44 | 0.41 | 0.30 | 0.48 | 0.36 |
| md_list | 0.78 | 0.44 | 0.44 | 0.49 | 0.44 | 0.54 | 0.57 | 0.56 | 0.51 | 0.47 |
| code_lang | 0.42 | 0.46 | 0.62 | 0.67 | 0.73 | 0.69 | 0.73 | 0.72 | 0.75 | 0.69 |

### carries

Carries (critic): k=1 readouts rebuilt by the final NLA reconstructor, scored by the frozen probe against the true layer-42 activations (exp 14/25 measure)

| row | textnla | av_base | iter_000100 | iter_000200 | iter_000300 | iter_000400 | iter_000500 | iter_000600 | iter_000700 | iter_000800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| py_string | 0.92 | 0.97 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| js_string | 0.89 | 0.95 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| latex_math | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| py_arg | 0.84 | 0.87 | 0.93 | 0.94 | 0.95 | 0.91 | 0.94 | 0.94 | 0.94 | 0.92 |
| json_depth | 0.91 | 0.97 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| md_heading | 0.64 | 0.67 | 0.70 | 0.68 | 0.69 | 0.69 | 0.68 | 0.69 | 0.68 | 0.68 |
| md_list | 0.47 | 0.45 | 0.47 | 0.48 | 0.48 | 0.48 | 0.51 | 0.50 | 0.49 | 0.49 |
| code_lang | 0.99 | 0.95 | 0.98 | 0.98 | 0.99 | 0.98 | 0.98 | 0.98 | 0.99 | 0.98 |

### textbag

Text bag: grouped 5-fold AUROC of a bag-of-tokens classifier on the k=1 readout text alone (exp 20/23 cross-check)

| row | textnla | av_base | iter_000100 | iter_000200 | iter_000300 | iter_000400 | iter_000500 | iter_000600 | iter_000700 | iter_000800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| py_string | 0.93 | 0.92 | 0.96 | 0.94 | 0.92 | 0.92 | 0.92 | 0.92 | 0.96 | 0.92 |
| js_string | 0.91 | 0.92 | 0.88 | 0.95 | 0.94 | 0.93 | 0.89 | 0.92 | 0.97 | 0.93 |
| latex_math | 0.88 | 0.91 | 0.91 | 0.89 | 0.91 | 0.90 | 0.89 | 0.86 | 0.92 | 0.88 |
| py_arg | 0.65 | 0.53 | 0.52 | 0.60 | 0.72 | 0.62 | 0.56 | 0.55 | 0.52 | 0.50 |
| json_depth | 0.88 | 0.97 | 0.98 | 0.99 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.97 |
| md_heading | 0.70 | 0.60 | 0.60 | 0.65 | 0.50 | 0.61 | 0.58 | 0.60 | 0.58 | 0.54 |
| md_list | 0.53 | 0.60 | 0.54 | 0.63 | 0.56 | 0.51 | 0.57 | 0.56 | 0.54 | 0.59 |
| code_lang | 0.93 | 0.86 | 0.95 | 0.99 | 0.98 | 0.95 | 0.97 | 0.95 | 0.95 | 0.98 |

### fve

FVE of the critic rebuilds of the k=1 readouts against the true activations (per-row predict-the-mean baseline)

| row | textnla | av_base | iter_000100 | iter_000200 | iter_000300 | iter_000400 | iter_000500 | iter_000600 | iter_000700 | iter_000800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| py_string | -0.05 | 0.18 | 0.53 | 0.54 | 0.55 | 0.57 | 0.56 | 0.58 | 0.58 | 0.59 |
| js_string | -0.06 | 0.14 | 0.51 | 0.51 | 0.53 | 0.53 | 0.53 | 0.55 | 0.56 | 0.55 |
| latex_math | 0.05 | 0.24 | 0.55 | 0.58 | 0.60 | 0.61 | 0.60 | 0.61 | 0.61 | 0.62 |
| py_arg | -0.13 | -0.06 | 0.32 | 0.34 | 0.36 | 0.36 | 0.37 | 0.39 | 0.41 | 0.40 |
| json_depth | -0.73 | -0.53 | -0.06 | -0.04 | 0.02 | 0.02 | 0.04 | 0.07 | 0.09 | 0.06 |
| md_heading | 0.00 | 0.28 | 0.59 | 0.60 | 0.62 | 0.61 | 0.63 | 0.62 | 0.64 | 0.66 |
| md_list | -0.09 | 0.15 | 0.51 | 0.53 | 0.54 | 0.56 | 0.53 | 0.55 | 0.57 | 0.58 |
| code_lang | -0.01 | 0.08 | 0.41 | 0.44 | 0.43 | 0.44 | 0.45 | 0.46 | 0.47 | 0.48 |

### cos

Cosine between the rebuilt and the true class-mean direction

| row | textnla | av_base | iter_000100 | iter_000200 | iter_000300 | iter_000400 | iter_000500 | iter_000600 | iter_000700 | iter_000800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| py_string | 0.56 | 0.79 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.92 | 0.91 |
| js_string | 0.61 | 0.79 | 0.88 | 0.88 | 0.88 | 0.87 | 0.88 | 0.87 | 0.89 | 0.88 |
| latex_math | 0.79 | 0.81 | 0.88 | 0.88 | 0.89 | 0.89 | 0.89 | 0.88 | 0.89 | 0.89 |
| py_arg | 0.62 | 0.69 | 0.85 | 0.85 | 0.85 | 0.86 | 0.86 | 0.86 | 0.87 | 0.87 |
| json_depth | 0.60 | 0.67 | 0.77 | 0.77 | 0.78 | 0.78 | 0.80 | 0.80 | 0.80 | 0.80 |
| md_heading | 0.56 | 0.72 | 0.85 | 0.85 | 0.86 | 0.85 | 0.86 | 0.86 | 0.86 | 0.87 |
| md_list | 0.42 | 0.51 | 0.72 | 0.71 | 0.81 | 0.81 | 0.78 | 0.77 | 0.76 | 0.81 |
| code_lang | 0.64 | 0.66 | 0.80 | 0.82 | 0.81 | 0.80 | 0.83 | 0.82 | 0.83 | 0.83 |

### Changes from av_base to iter_000400

- states: structural rows mean -0.03; code_lang +0.16; textnla minus av_base on the readable rows +0.17
- names: structural rows mean -0.02; code_lang +0.23; textnla minus av_base on the readable rows +0.26
- carries: structural rows mean +0.02; code_lang +0.03; textnla minus av_base on the readable rows -0.04
- textbag: structural rows mean +0.00; code_lang +0.08; textnla minus av_base on the readable rows +0.00

### Pre-registered checks (RESULTS.md)

- p1_structural_states_flat: FAILS
- p2_code_lang_states_rise: holds
- p3_carrying_rises_and_textnla_near_base: FAILS
- p4_textbag_stable_on_strings: FAILS

### Consistency with exp 25 (iter_000400 carrying)

- py_string: exp 25 nla_readout 0.99, this run 0.99 (new samples, seed 0)
- js_string: exp 25 nla_readout 0.97, this run 0.97 (new samples, seed 0)
- latex_math: exp 25 nla_readout 1.00, this run 1.00 (new samples, seed 0)
- py_arg: exp 25 nla_readout 0.93, this run 0.91 (new samples, seed 0)
- json_depth: exp 25 nla_readout 0.99, this run 0.99 (new samples, seed 0)
- md_heading: exp 25 nla_readout 0.70, this run 0.69 (new samples, seed 0)
- md_list: exp 25 nla_readout 0.49, this run 0.48 (new samples, seed 0)
- code_lang: exp 25 nla_readout 0.98, this run 0.98 (new samples, seed 0)
