# Phase A (existing data) — PILOT / EXPLORATORY

## A1 Oracle Lens by layer (exp 08 col pilot, centred, 60 items)

| cell | AUROC | win | shift | z_shift (band) |
|---|---|---|---|---|
| OL@40|count_qual|raw|q90 | 0.47 | 0.38 | -0.34 | -0.7 (iso~proxy(jrem1)), -1.3 (nonj1~proxy(jrem1)) |
| OL@40|count_qual|raw|q99 | 0.46 | 0.43 | -0.44 | -0.6 (iso~proxy(jrem1)), -1.2 (nonj1~proxy(jrem1)) |
| OL@40|count_qual|jrem1|q90 | 0.50 | 0.47 | -0.03 | 0.3 (iso), -0.2 (nonj1) |
| OL@40|count_qual|jrem1|q99 | 0.49 | 0.52 | -0.08 | 0.2 (iso), -0.3 (nonj1) |
| OL@40|count_qual|jrem4|q90 | 0.49 | 0.37 | -0.13 | -0.0 (iso~proxy(jrem1)), -0.6 (nonj1~proxy(jrem1)) |
| OL@40|count_qual|jrem4|q99 | 0.48 | 0.37 | -0.20 | -0.1 (iso~proxy(jrem1)), -0.6 (nonj1~proxy(jrem1)) |
| OL@40|count_num|raw|q90 | 0.49 | 0.37 | -0.12 | -1.0 (iso~proxy(jrem1)), -0.8 (nonj1~proxy(jrem1)) |
| OL@40|count_num|raw|q99 | 0.49 | 0.37 | -0.13 | -0.6 (iso~proxy(jrem1)), -0.5 (nonj1~proxy(jrem1)) |
| OL@40|count_num|jrem1|q90 | 0.50 | 0.47 | -0.04 | -0.2 (iso), -0.1 (nonj1) |
| OL@40|count_num|jrem1|q99 | 0.49 | 0.37 | -0.08 | -0.3 (iso), -0.2 (nonj1) |
| OL@40|count_num|jrem4|q90 | 0.50 | 0.50 | 0.03 | 0.4 (iso~proxy(jrem1)), 0.5 (nonj1~proxy(jrem1)) |
| OL@40|count_num|jrem4|q99 | 0.51 | 0.57 | 0.08 | 0.6 (iso~proxy(jrem1)), 0.7 (nonj1~proxy(jrem1)) |
| OL@40|decision|raw|q90 | 0.53 | 0.68 | 0.23 | 0.8 (iso~proxy(jrem1)), 1.0 (nonj1~proxy(jrem1)) |
| OL@40|decision|raw|q99 | 0.53 | 0.67 | 0.30 | 0.7 (iso~proxy(jrem1)), 0.9 (nonj1~proxy(jrem1)) |
| OL@40|decision|jrem1|q90 | 0.50 | 0.48 | 0.02 | -0.1 (iso), 0.1 (nonj1) |
| OL@40|decision|jrem1|q99 | 0.51 | 0.53 | 0.02 | -0.1 (iso), 0.1 (nonj1) |
| OL@40|decision|jrem4|q90 | 0.51 | 0.52 | 0.03 | -0.0 (iso~proxy(jrem1)), 0.2 (nonj1~proxy(jrem1)) |
| OL@40|decision|jrem4|q99 | 0.51 | 0.52 | 0.05 | -0.0 (iso~proxy(jrem1)), 0.2 (nonj1~proxy(jrem1)) |
| OL@42|count_qual|raw|q90 | 0.48 | 0.38 | -0.24 | -0.7 (iso) |
| OL@42|count_qual|raw|q99 | 0.46 | 0.35 | -0.32 | -0.6 (iso) |
| OL@42|count_qual|jrem1|q90 | 0.49 | 0.48 | -0.06 | -0.1 (iso), -0.2 (nonj1) |
| OL@42|count_qual|jrem1|q99 | 0.50 | 0.52 | -0.01 | 0.1 (iso), 0.0 (nonj1) |
| OL@42|count_qual|jrem4|q90 | 0.49 | 0.37 | -0.10 | -0.4 (iso), -0.6 (nonj4) |
| OL@42|count_qual|jrem4|q99 | 0.49 | 0.43 | -0.14 | -0.3 (iso), -0.6 (nonj4) |
| OL@42|count_num|raw|q90 | 0.49 | 0.42 | -0.07 | -0.9 (iso) |
| OL@42|count_num|raw|q99 | 0.48 | 0.45 | -0.11 | -0.9 (iso) |
| OL@42|count_num|jrem1|q90 | 0.50 | 0.40 | -0.01 | -0.2 (iso), -0.3 (nonj1) |
| OL@42|count_num|jrem1|q99 | 0.50 | 0.47 | -0.03 | -0.3 (iso), -0.4 (nonj1) |
| OL@42|count_num|jrem4|q90 | 0.51 | 0.58 | 0.08 | 0.9 (iso), 1.2 (nonj4) |
| OL@42|count_num|jrem4|q99 | 0.51 | 0.53 | 0.07 | 0.6 (iso), 0.7 (nonj4) |
| OL@42|decision|raw|q90 | 0.50 | 0.57 | 0.05 | -0.1 (iso) |
| OL@42|decision|raw|q99 | 0.50 | 0.55 | 0.05 | -0.2 (iso) |
| OL@42|decision|jrem1|q90 | 0.49 | 0.43 | -0.10 | -0.7 (iso), -0.4 (nonj1) |
| OL@42|decision|jrem1|q99 | 0.48 | 0.40 | -0.14 | -0.6 (iso), -0.4 (nonj1) |
| OL@42|decision|jrem4|q90 | 0.49 | 0.42 | -0.08 | -0.7 (iso), -0.3 (nonj4) |
| OL@42|decision|jrem4|q99 | 0.49 | 0.43 | -0.10 | -0.6 (iso), -0.1 (nonj4) |
| OL@44|count_qual|raw|q90 | 0.48 | 0.47 | -0.19 | -0.9 (iso~proxy(jrem1)), -0.7 (nonj1~proxy(jrem1)) |
| OL@44|count_qual|raw|q99 | 0.48 | 0.42 | -0.26 | -0.9 (iso~proxy(jrem1)), -0.7 (nonj1~proxy(jrem1)) |
| OL@44|count_qual|jrem1|q90 | 0.49 | 0.53 | -0.08 | -0.5 (iso), -0.3 (nonj1) |
| OL@44|count_qual|jrem1|q99 | 0.50 | 0.50 | -0.07 | -0.4 (iso), -0.1 (nonj1) |
| OL@44|count_qual|jrem4|q90 | 0.51 | 0.57 | 0.07 | 0.1 (iso~proxy(jrem1)), 0.4 (nonj1~proxy(jrem1)) |
| OL@44|count_qual|jrem4|q99 | 0.51 | 0.57 | 0.06 | 0.0 (iso~proxy(jrem1)), 0.2 (nonj1~proxy(jrem1)) |
| OL@44|count_num|raw|q90 | 0.49 | 0.35 | -0.07 | -0.6 (iso~proxy(jrem1)), -0.7 (nonj1~proxy(jrem1)) |
| OL@44|count_num|raw|q99 | 0.48 | 0.35 | -0.12 | -0.7 (iso~proxy(jrem1)), -0.8 (nonj1~proxy(jrem1)) |
| OL@44|count_num|jrem1|q90 | 0.49 | 0.45 | -0.04 | -0.3 (iso), -0.4 (nonj1) |
| OL@44|count_num|jrem1|q99 | 0.49 | 0.43 | -0.03 | -0.2 (iso), -0.2 (nonj1) |
| OL@44|count_num|jrem4|q90 | 0.51 | 0.52 | 0.03 | 0.2 (iso~proxy(jrem1)), 0.3 (nonj1~proxy(jrem1)) |
| OL@44|count_num|jrem4|q99 | 0.51 | 0.58 | 0.10 | 0.6 (iso~proxy(jrem1)), 0.7 (nonj1~proxy(jrem1)) |
| OL@44|decision|raw|q90 | 0.51 | 0.65 | 0.06 | 1.0 (iso~proxy(jrem1)), 0.1 (nonj1~proxy(jrem1)) |
| OL@44|decision|raw|q99 | 0.51 | 0.63 | 0.07 | 0.8 (iso~proxy(jrem1)), 0.0 (nonj1~proxy(jrem1)) |
| OL@44|decision|jrem1|q90 | 0.50 | 0.53 | -0.01 | 0.5 (iso), -0.3 (nonj1) |
| OL@44|decision|jrem1|q99 | 0.50 | 0.58 | -0.00 | 0.5 (iso), -0.2 (nonj1) |
| OL@44|decision|jrem4|q90 | 0.49 | 0.37 | -0.07 | 0.1 (iso~proxy(jrem1)), -0.6 (nonj1~proxy(jrem1)) |
| OL@44|decision|jrem4|q99 | 0.49 | 0.47 | -0.09 | 0.2 (iso~proxy(jrem1)), -0.5 (nonj1~proxy(jrem1)) |

WorkspaceBench pass (exp 06): {"NLA@42": {"typo_mt": 0.73, "multilingual_mt": 0.42, "multihop_mt": 0.15, "basic_readout_mt": 0.57, "pooled": 0.4675}, "OL@42": {"typo_mt": 0.76, "multilingual_mt": 0.28, "multihop_mt": 0.1, "basic_readout_mt": 0.45, "pooled": 0.3975}, "OL@40": {"typo_mt": 0.72, "multilingual_mt": 0.26, "multihop_mt": 0.08, "basic_readout_mt": 0.42, "pooled": 0.37}, "OL@44": {"typo_mt": 0.7, "multilingual_mt": 0.26, "multihop_mt": 0.1, "basic_readout_mt": 0.48, "pooled": 0.385}, "JLens@42": {"typo_mt": 0.07, "multilingual_mt": 0.0, "multihop_mt": 0.0, "basic_readout_mt": 0.01, "pooled": 0.02}, "OL grid (any-of-11)": {"typo_mt": 0.86, "multilingual_mt": 0.65, "multihop_mt": 0.34, "basic_readout_mt": 0.73, "pooled": 0.645}}

OL grid pass by layer (exp 01): {"basic_readout_mt": {"20": 0.33, "24": 0.4, "28": 0.43, "32": 0.43, "36": 0.6, "40": 0.59, "44": 0.66, "48": 0.64, "52": 0.78, "56": 0.8, "60": 0.77}, "multihop_mt": {"20": 0.09, "24": 0.08, "28": 0.08, "32": 0.09, "36": 0.17, "40": 0.12, "44": 0.12, "48": 0.13, "52": 0.26, "56": 0.31, "60": 0.24}, "multilingual_mt": {"20": 0.34, "24": 0.4, "28": 0.44, "32": 0.48, "36": 0.62, "40": 0.76, "44": 0.78, "48": 0.81, "52": 0.85, "56": 0.89, "60": 0.89}, "typo_mt": {"20": 0.73, "24": 0.78, "28": 0.71, "32": 0.68, "36": 0.73, "40": 0.72, "44": 0.7, "48": 0.66, "52": 0.74, "56": 0.78, "60": 0.76}}

## A2 consensus (>=2 of 4 samples) vs single-sample mention rates (strict regex)

| reader | condition | single | items any | consensus |
|---|---|---|---|---|
| NLA | natural | 0.000 | 0.000 | 0.000 |
| NLA | real|jrem1|q90|+ | 0.008 | 0.017 | 0.017 |
| NLA | real|jrem1|q90|- | 0.000 | 0.000 | 0.000 |
| NLA | real|jrem1|q99|+ | 0.000 | 0.000 | 0.000 |
| NLA | real|jrem1|q99|- | 0.000 | 0.000 | 0.000 |
| NLA | real|jrem4|q90|+ | 0.004 | 0.017 | 0.000 |
| NLA | real|jrem4|q90|- | 0.000 | 0.000 | 0.000 |
| NLA | real|jrem4|q99|+ | 0.000 | 0.000 | 0.000 |
| NLA | real|jrem4|q99|- | 0.004 | 0.017 | 0.000 |
| NLA | real|raw|q90|+ | 0.004 | 0.017 | 0.000 |
| NLA | real|raw|q90|- | 0.000 | 0.000 | 0.000 |
| NLA | real|raw|q99|+ | 0.000 | 0.000 | 0.000 |
| NLA | real|raw|q99|- | 0.004 | 0.017 | 0.000 |
| NLA | null|iso_0|q99|+ | 0.000 | 0.000 | 0.000 |
| NLA | null|iso_0|q99|- | 0.000 | 0.000 | 0.000 |
| NLA | null|nonj1_0|q99|+ | 0.004 | 0.017 | 0.000 |
| NLA | null|nonj1_0|q99|- | 0.000 | 0.000 | 0.000 |
| OL | natural | 0.000 | 0.000 | 0.000 |
| OL | real|jrem1|q90|+ | 0.004 | 0.017 | 0.000 |
| OL | real|jrem1|q90|- | 0.000 | 0.000 | 0.000 |
| OL | real|jrem1|q99|+ | 0.000 | 0.000 | 0.000 |
| OL | real|jrem1|q99|- | 0.000 | 0.000 | 0.000 |
| OL | real|raw|q99|+ | 0.000 | 0.000 | 0.000 |
| OL | real|raw|q99|- | 0.000 | 0.000 | 0.000 |

exp 07 (k=8, unsteered) strict self-consistency: {"nj1_col|NLA@42": {"sc": "1.00", "single": "0.000", "consensus": "0.000"}, "nj1_col|OL@42": {"sc": "1.00", "single": "0.003", "consensus": "0.000"}, "nj1_col|OL@40": {"sc": "1.00", "single": "0.003", "consensus": "0.000"}, "nj1_col|OL@44": {"sc": "0.99", "single": "0.006", "consensus": "0.025"}, "nj1_col_break|NLA@42": {"sc": "1.00", "single": "0.000", "consensus": "0.000"}, "nj1_col_break|OL@42": {"sc": "1.00", "single": "0.000", "consensus": "0.000"}, "nj1_col_break|OL@40": {"sc": "1.00", "single": "0.000", "consensus": "0.000"}, "nj1_col_break|OL@44": {"sc": "1.00", "single": "0.000", "consensus": "0.000"}}

## A3 punctuation strata (shift = mean LO(+) − LO(−); n per stratum)

col strata counts: {"ends": {"punct": 4, "word": 56}, "sentence_start": {"False": 56, "True": 4}, "chars_remaining": {"6-20": 19, "<=5": 17, ">20": 24}, "next_is_nl": {"False": 51, "True": 9}}

| cell | all | ends=stop | ends=punct | ends=word | sent_start=True | sent_start=False |
|---|---|---|---|---|---|---|
| OL@42|count_qual|reg_jrem4|q90 | -0.10 (n60) | - | 0.27 (n4) | -0.12 (n56) | -0.42 (n4) | -0.07 (n56) |
| OL@42|count_qual|reg_raw|q90 | -0.25 (n60) | - | 0.39 (n4) | -0.29 (n56) | -0.67 (n4) | -0.22 (n56) |
| OL@42|count_num|reg_jrem4|q90 | 0.03 (n60) | - | -0.04 (n4) | 0.04 (n56) | 0.20 (n4) | 0.02 (n56) |
| OL@42|count_num|reg_raw|q90 | -0.07 (n60) | - | -0.03 (n4) | -0.07 (n56) | -0.22 (n4) | -0.06 (n56) |
| OL@42|decision|reg_jrem4|q90 | -0.05 (n60) | - | 0.28 (n4) | -0.07 (n56) | 0.12 (n4) | -0.06 (n56) |
| OL@42|decision|reg_raw|q90 | 0.02 (n60) | - | 0.42 (n4) | -0.01 (n56) | -0.12 (n4) | 0.03 (n56) |
| OL@42|f07_count|reg_jrem4|q90 | -0.08 (n60) | - | 0.09 (n4) | -0.09 (n56) | -0.18 (n4) | -0.07 (n56) |
| OL@42|f07_count|reg_raw|q90 | -0.06 (n60) | - | 0.01 (n4) | -0.06 (n56) | -0.19 (n4) | -0.05 (n56) |
| OL@42|f07_break|reg_jrem4|q90 | -0.32 (n60) | - | -0.44 (n4) | -0.31 (n56) | -0.55 (n4) | -0.30 (n56) |
| OL@42|f07_break|reg_raw|q90 | -0.19 (n60) | - | -0.10 (n4) | -0.20 (n56) | -0.33 (n4) | -0.18 (n56) |
| OL@42|nl_cont|reg_jrem4|q90 | 1.72 (n60) | - | 1.72 (n4) | 1.72 (n56) | 2.02 (n4) | 1.70 (n56) |
| OL@42|nl_cont|reg_raw|q90 | 2.87 (n60) | - | 2.36 (n4) | 2.91 (n56) | 2.98 (n4) | 2.87 (n56) |
| NLA@42|count_qual|reg_jrem4|q90 | 0.05 (n60) | - | 0.19 (n4) | 0.04 (n56) | -0.20 (n4) | 0.07 (n56) |
| NLA@42|count_qual|reg_raw|q90 | -0.18 (n60) | - | 0.28 (n4) | -0.21 (n56) | -0.88 (n4) | -0.13 (n56) |
| NLA@42|count_num|reg_jrem4|q90 | 0.09 (n60) | - | 0.20 (n4) | 0.08 (n56) | -0.27 (n4) | 0.12 (n56) |
| NLA@42|count_num|reg_raw|q90 | 0.15 (n60) | - | 0.37 (n4) | 0.14 (n56) | -0.37 (n4) | 0.19 (n56) |
| NLA@42|decision|reg_jrem4|q90 | 0.04 (n60) | - | -0.02 (n4) | 0.05 (n56) | 0.11 (n4) | 0.04 (n56) |
| NLA@42|decision|reg_raw|q90 | 0.07 (n60) | - | -0.10 (n4) | 0.08 (n56) | 0.13 (n4) | 0.06 (n56) |
| NLA@42|f07_count|reg_jrem4|q90 | 0.35 (n60) | - | 0.29 (n4) | 0.35 (n56) | 0.48 (n4) | 0.34 (n56) |
| NLA@42|f07_count|reg_raw|q90 | 0.58 (n60) | - | 0.59 (n4) | 0.58 (n56) | 0.66 (n4) | 0.58 (n56) |
| NLA@42|f07_break|reg_jrem4|q90 | -0.04 (n60) | - | 0.02 (n4) | -0.05 (n56) | -0.07 (n4) | -0.04 (n56) |
| NLA@42|f07_break|reg_raw|q90 | -0.03 (n60) | - | 0.05 (n4) | -0.03 (n56) | -0.11 (n4) | -0.02 (n56) |

tense strata counts: {"ends": {"punct": 3, "stop": 11, "word": 66}, "sentence_start": {"False": 68, "True": 12}}

| cell | all | ends=stop | ends=punct | ends=word | sent_start=True | sent_start=False |
|---|---|---|---|---|---|---|
| OL@42|tense_tok|jrem4|q90 | 0.34 (n80) | 0.58 (n11) | 0.77 (n3) | 0.28 (n66) | 0.68 (n12) | 0.28 (n68) |
| OL@42|f07_tense|jrem4|q90 | 2.02 (n80) | 2.09 (n11) | 2.36 (n3) | 1.99 (n66) | 2.24 (n12) | 1.98 (n68) |
| model|tense_tok|jrem4|q90 | -0.02 (n80) | -0.02 (n11) | -0.04 (n3) | -0.02 (n66) | -0.02 (n12) | -0.02 (n68) |
| model|tense_broad|jrem4|q90 | -0.03 (n80) | -0.03 (n11) | -0.04 (n3) | -0.02 (n66) | -0.03 (n12) | -0.02 (n68) |
| OL@42|tense_tok|jrem4|r108 | 0.62 (n80) | 1.06 (n11) | 1.36 (n3) | 0.51 (n66) | 1.21 (n12) | 0.51 (n68) |
| OL@42|f07_tense|jrem4|r108 | 3.71 (n80) | 3.74 (n11) | 3.96 (n3) | 3.69 (n66) | 3.94 (n12) | 3.67 (n68) |
| model|tense_tok|jrem4|r108 | -0.04 (n80) | -0.01 (n11) | -0.11 (n3) | -0.04 (n66) | -0.00 (n12) | -0.05 (n68) |
| model|tense_broad|jrem4|r108 | -0.05 (n80) | -0.03 (n11) | -0.10 (n3) | -0.05 (n66) | -0.03 (n12) | -0.05 (n68) |
| OL@42|tense_tok|raw|q90 | 1.85 (n80) | 3.13 (n11) | 2.96 (n3) | 1.59 (n66) | 3.27 (n12) | 1.60 (n68) |
| OL@42|f07_tense|raw|q90 | 7.91 (n80) | 8.99 (n11) | 8.24 (n3) | 7.72 (n66) | 9.08 (n12) | 7.70 (n68) |
| model|tense_tok|raw|q90 | 0.54 (n80) | 0.64 (n11) | 0.41 (n3) | 0.52 (n66) | 0.63 (n12) | 0.52 (n68) |
| model|tense_broad|raw|q90 | 0.45 (n80) | 0.54 (n11) | 0.32 (n3) | 0.45 (n66) | 0.51 (n12) | 0.44 (n68) |
| OL@42|tense_tok|raw|r108 | 2.06 (n80) | 3.44 (n11) | 3.11 (n3) | 1.79 (n66) | 3.57 (n12) | 1.80 (n68) |
| OL@42|f07_tense|raw|r108 | 8.72 (n80) | 9.80 (n11) | 8.57 (n3) | 8.54 (n66) | 9.86 (n12) | 8.51 (n68) |
| model|tense_tok|raw|r108 | 0.61 (n80) | 0.71 (n11) | 0.42 (n3) | 0.60 (n66) | 0.69 (n12) | 0.59 (n68) |
| model|tense_broad|raw|r108 | 0.52 (n80) | 0.59 (n11) | 0.33 (n3) | 0.51 (n66) | 0.56 (n12) | 0.51 (n68) |

## A5 probe vs readers by layer (AUROC on the same steered pairs)

| feature | layer | dir | dose | probe | J-Lens | OL | NLA |
|---|---|---|---|---|---|---|---|
| col | 40 | raw | q90 | 1.00 | 0.76 | 0.47 | nan |
| col | 40 | raw | q99 | 1.00 | 0.84 | 0.46 | nan |
| col | 40 | jrem1 | q90 | 1.00 | 0.54 | 0.50 | nan |
| col | 40 | jrem1 | q99 | 1.00 | 0.55 | 0.49 | nan |
| col | 40 | jrem4 | q90 | 1.00 | 0.50 | 0.49 | nan |
| col | 40 | jrem4 | q99 | 1.00 | 0.50 | 0.48 | nan |
| col | 42 | raw | q90 | 1.00 | 0.73 | 0.48 | 0.48 |
| col | 42 | raw | q99 | 1.00 | 0.81 | 0.46 | 0.48 |
| col | 42 | jrem1 | q90 | 1.00 | 0.54 | 0.49 | 0.52 |
| col | 42 | jrem1 | q99 | 1.00 | 0.56 | 0.50 | 0.54 |
| col | 42 | jrem4 | q90 | 1.00 | 0.50 | 0.49 | 0.50 |
| col | 42 | jrem4 | q99 | 1.00 | 0.50 | 0.49 | 0.51 |
| col | 44 | raw | q90 | 1.00 | 0.74 | 0.48 | nan |
| col | 44 | raw | q99 | 1.00 | 0.81 | 0.48 | nan |
| col | 44 | jrem1 | q90 | 1.00 | 0.59 | 0.49 | nan |
| col | 44 | jrem1 | q99 | 1.00 | 0.62 | 0.50 | nan |
| col | 44 | jrem4 | q90 | 1.00 | 0.52 | 0.51 | nan |
| col | 44 | jrem4 | q99 | 1.00 | 0.53 | 0.51 | nan |
| lang | 40 | raw | q90 | 1.00 | 0.72 | 0.77 | nan |
| lang | 40 | jrem4 | q90 | 0.97 | 0.51 | 0.58 | nan |
| lang | 42 | raw | q90 | 1.00 | 0.68 | 0.76 | nan |
| lang | 42 | jrem4 | q90 | 0.98 | 0.50 | 0.55 | nan |
| lang | 44 | raw | q90 | 1.00 | 0.76 | 0.77 | nan |
| lang | 44 | jrem4 | q90 | 0.97 | 0.53 | 0.54 | nan |
| tense | 40 | raw | q90 | 0.99 | 0.86 | 0.55 | nan |
| tense | 40 | jrem4 | q90 | 0.96 | 0.50 | 0.52 | nan |
| tense | 42 | raw | q90 | 0.99 | 0.79 | 0.55 | nan |
| tense | 42 | jrem4 | q90 | 0.95 | 0.50 | 0.51 | nan |
| tense | 44 | raw | q90 | 0.99 | 0.88 | 0.52 | nan |
| tense | 44 | jrem4 | q90 | 0.96 | 0.51 | 0.51 | nan |
| col | 40 | raw | q90 | 1.00 | 0.50 | nan | nan |
| col | 42 | raw | q90 | 1.00 | 0.51 | nan | nan |
| col | 44 | raw | q90 | 1.00 | 0.51 | nan | nan |

rank correlation (probe AUROC vs reader AUROC across cells): {"NLA|count_num|auroc": {"rho": "nan", "n": 6}, "NLA|count_qual|auroc": {"rho": "nan", "n": 6}, "NLA|decision|auroc": {"rho": "nan", "n": 6}, "OL|count_num|auroc": {"rho": "-0.03", "n": 18}, "OL|count_qual|auroc": {"rho": "-0.13", "n": 18}, "OL|decision|auroc": {"rho": "-0.06", "n": 18}, "OL|label|auroc": {"rho": "0.77", "n": 12}, "jlens_eol_auroc": {"rho": "0.22", "n": 18}, "jlens_label_auroc": {"rho": "0.28", "n": 15}}
