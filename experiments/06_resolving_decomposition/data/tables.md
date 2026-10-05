## Pass rates (regex mt-regex-2026-09-23, L42, k=1 unless noted; 95% item bootstrap CI)
| arm | typo_mt | multilingual_mt | multihop_mt | basic_readout_mt | pooled (n=400) |
|---|---|---|---|---|---|
| NLA@42 | 0.73 [0.64,0.82] | 0.42 [0.32,0.52] | 0.15 [0.09,0.22] | 0.57 [0.47,0.67] | 0.47 [0.42,0.52] |
| OL@42 | 0.76 [0.67,0.84] | 0.28 [0.19,0.37] | 0.10 [0.05,0.17] | 0.45 [0.35,0.55] | 0.40 [0.35,0.45] |
| OL@40 | 0.72 [0.63,0.80] | 0.26 [0.18,0.35] | 0.08 [0.03,0.14] | 0.42 [0.33,0.52] | 0.37 [0.33,0.42] |
| OL@44 | 0.70 [0.61,0.79] | 0.26 [0.17,0.35] | 0.10 [0.05,0.16] | 0.48 [0.39,0.58] | 0.39 [0.34,0.43] |
| JLens@42 | 0.07 [0.02,0.13] | 0.00 [0.00,0.00] | 0.00 [0.00,0.00] | 0.01 [0.00,0.03] | 0.02 [0.01,0.04] |
| OL grid (any-of-11) | 0.86 [0.79,0.92] | 0.65 [0.56,0.74] | 0.34 [0.25,0.43] | 0.73 [0.64,0.82] | 0.65 [0.60,0.69] |
| T_qwen | 0.45 [0.35,0.55] | 0.40 [0.30,0.50] | 0.44 [0.34,0.54] | 0.53 [0.43,0.63] | 0.46 [0.41,0.51] |
| T_qwen_k3 | 0.68 [0.58,0.77] | 0.65 [0.55,0.74] | 0.64 [0.54,0.73] | 0.81 [0.73,0.88] | 0.69 [0.65,0.74] |
| T_wsb | 0.85 [0.78,0.92] | 0.58 [0.48,0.68] | 0.82 [0.74,0.89] | 0.77 [0.69,0.85] | 0.76 [0.71,0.80] |
| T_gem | 0.66 [0.56,0.75] | 0.73 [0.64,0.82] | 0.75 [0.66,0.83] | 0.87 [0.80,0.93] | 0.75 [0.71,0.79] |
| T_gem_wsb | 0.90 [0.84,0.96] | 0.28 [0.19,0.38] | 0.77 [0.68,0.85] | 0.55 [0.45,0.65] | 0.62 [0.58,0.67] |
| T_any | 1.00 [1.00,1.00] | 0.87 [0.80,0.93] | 0.98 [0.95,1.00] | 0.98 [0.95,1.00] | 0.96 [0.94,0.98] |
| Resolve(NLA) Qwen | 0.75 [0.66,0.84] | 0.61 [0.51,0.70] | 0.26 [0.18,0.35] | 0.65 [0.56,0.74] | 0.57 [0.52,0.62] |
| Resolve(NLA) Gemini | 0.77 [0.69,0.85] | 0.63 [0.53,0.72] | 0.32 [0.23,0.41] | 0.68 [0.59,0.77] | 0.60 [0.55,0.65] |
| Resolve(OL@42) Qwen | 0.72 [0.63,0.81] | 0.61 [0.51,0.70] | 0.18 [0.11,0.26] | 0.60 [0.50,0.69] | 0.53 [0.48,0.57] |
| Resolve(OL@42) Gemini | 0.73 [0.64,0.81] | 0.59 [0.49,0.68] | 0.28 [0.20,0.37] | 0.64 [0.55,0.73] | 0.56 [0.51,0.61] |
| T_qwen per-sample rates s0/s1/s2 | 0.45/0.49/0.50 | 0.40/0.42/0.36 | 0.44/0.43/0.46 | 0.53/0.59/0.58 | |

## Decomposition: NLA@42 vs text-only arms
| text arm | family | NLA@42 pass | text pass | P(text pass ∣ reader pass) | reader-only hits | text-only hits | McNemar p |
|---|---|---|---|---|---|---|---|
| T_qwen | typo_mt | 73/100 | 45/100 | 0.52 | 35 | 7 | 1.5e-05 |
| T_qwen | multilingual_mt | 42/100 | 40/100 | 0.52 | 20 | 18 | 0.87 |
| T_qwen | multihop_mt | 15/100 | 44/100 | 0.73 | 4 | 33 | 1.1e-06 |
| T_qwen | basic_readout_mt | 57/100 | 53/100 | 0.63 | 21 | 17 | 0.63 |
| T_qwen | pooled | 187/400 | 182/400 | 0.57 | 80 | 75 | 0.75 |
| T_qwen_k3 | typo_mt | 73/100 | 68/100 | 0.73 | 20 | 15 | 0.5 |
| T_qwen_k3 | multilingual_mt | 42/100 | 65/100 | 0.90 | 4 | 27 | 3.4e-05 |
| T_qwen_k3 | multihop_mt | 15/100 | 64/100 | 0.87 | 2 | 51 | 3.2e-13 |
| T_qwen_k3 | basic_readout_mt | 57/100 | 81/100 | 0.86 | 8 | 32 | 0.00018 |
| T_qwen_k3 | pooled | 187/400 | 278/400 | 0.82 | 34 | 125 | 2e-13 |
| T_gem | typo_mt | 73/100 | 66/100 | 0.64 | 26 | 19 | 0.37 |
| T_gem | multilingual_mt | 42/100 | 73/100 | 0.90 | 4 | 35 | 3.4e-07 |
| T_gem | multihop_mt | 15/100 | 75/100 | 1.00 | 0 | 60 | 1.7e-18 |
| T_gem | basic_readout_mt | 57/100 | 87/100 | 0.95 | 3 | 33 | 2.3e-07 |
| T_gem | pooled | 187/400 | 301/400 | 0.82 | 33 | 147 | 2.2e-18 |
| T_any | typo_mt | 73/100 | 100/100 | 1.00 | 0 | 27 | 1.5e-08 |
| T_any | multilingual_mt | 42/100 | 87/100 | 1.00 | 0 | 45 | 5.7e-14 |
| T_any | multihop_mt | 15/100 | 98/100 | 1.00 | 0 | 83 | 2.1e-25 |
| T_any | basic_readout_mt | 57/100 | 98/100 | 1.00 | 0 | 41 | 9.1e-13 |
| T_any | pooled | 187/400 | 383/400 | 1.00 | 0 | 196 | 2e-59 |

## Decomposition: OL@42 vs text-only arms
| text arm | family | OL@42 pass | text pass | P(text pass ∣ reader pass) | reader-only hits | text-only hits | McNemar p |
|---|---|---|---|---|---|---|---|
| T_qwen | typo_mt | 76/100 | 45/100 | 0.51 | 37 | 6 | 1.6e-06 |
| T_qwen | multilingual_mt | 28/100 | 40/100 | 0.46 | 15 | 27 | 0.088 |
| T_qwen | multihop_mt | 10/100 | 44/100 | 0.40 | 6 | 40 | 3.1e-07 |
| T_qwen | basic_readout_mt | 45/100 | 53/100 | 0.69 | 14 | 22 | 0.24 |
| T_qwen | pooled | 159/400 | 182/400 | 0.55 | 72 | 95 | 0.088 |
| T_qwen_k3 | typo_mt | 76/100 | 68/100 | 0.72 | 21 | 13 | 0.23 |
| T_qwen_k3 | multilingual_mt | 28/100 | 65/100 | 0.75 | 7 | 44 | 1.2e-07 |
| T_qwen_k3 | multihop_mt | 10/100 | 64/100 | 0.80 | 2 | 56 | 1.2e-14 |
| T_qwen_k3 | basic_readout_mt | 45/100 | 81/100 | 0.96 | 2 | 38 | 1.5e-09 |
| T_qwen_k3 | pooled | 159/400 | 278/400 | 0.80 | 32 | 151 | 1.1e-19 |
| T_gem | typo_mt | 76/100 | 66/100 | 0.67 | 25 | 15 | 0.15 |
| T_gem | multilingual_mt | 28/100 | 73/100 | 0.79 | 6 | 51 | 5.7e-10 |
| T_gem | multihop_mt | 10/100 | 75/100 | 0.90 | 1 | 66 | 9.2e-19 |
| T_gem | basic_readout_mt | 45/100 | 87/100 | 0.96 | 2 | 44 | 3.1e-11 |
| T_gem | pooled | 159/400 | 301/400 | 0.79 | 34 | 176 | 2.7e-24 |
| T_any | typo_mt | 76/100 | 100/100 | 1.00 | 0 | 24 | 1.2e-07 |
| T_any | multilingual_mt | 28/100 | 87/100 | 0.86 | 4 | 63 | 1.1e-14 |
| T_any | multihop_mt | 10/100 | 98/100 | 1.00 | 0 | 88 | 6.5e-27 |
| T_any | basic_readout_mt | 45/100 | 98/100 | 1.00 | 0 | 53 | 2.2e-16 |
| T_any | pooled | 159/400 | 383/400 | 0.97 | 4 | 228 | 3.5e-62 |

## Re-solving ceiling (text-only LLM given only the reader's readout)
| reader | re-solver | family | reader pass | re-solve pass | uplift (reader ✗, re-solve ✓) | loss (reader ✓, re-solve ✗) | McNemar p |
|---|---|---|---|---|---|---|---|
| NLA@42 | Qwen | typo_mt | 73 | 75 | 2 | 0 | 0.5 |
| NLA@42 | Qwen | multilingual_mt | 42 | 61 | 22 | 3 | 0.00016 |
| NLA@42 | Qwen | multihop_mt | 15 | 26 | 11 | 0 | 0.00098 |
| NLA@42 | Qwen | basic_readout_mt | 57 | 65 | 11 | 3 | 0.057 |
| NLA@42 | Qwen | pooled | 187 | 227 | 46 | 6 | 1e-08 |
| NLA@42 | Gemini | typo_mt | 73 | 77 | 6 | 2 | 0.29 |
| NLA@42 | Gemini | multilingual_mt | 42 | 63 | 23 | 2 | 1.9e-05 |
| NLA@42 | Gemini | multihop_mt | 15 | 32 | 19 | 2 | 0.00022 |
| NLA@42 | Gemini | basic_readout_mt | 57 | 68 | 14 | 3 | 0.013 |
| NLA@42 | Gemini | pooled | 187 | 240 | 62 | 9 | 7.3e-11 |
| OL@42 | Qwen | typo_mt | 76 | 72 | 0 | 4 | 0.12 |
| OL@42 | Qwen | multilingual_mt | 28 | 61 | 34 | 1 | 2.1e-09 |
| OL@42 | Qwen | multihop_mt | 10 | 18 | 9 | 1 | 0.021 |
| OL@42 | Qwen | basic_readout_mt | 45 | 60 | 16 | 1 | 0.00027 |
| OL@42 | Qwen | pooled | 159 | 211 | 59 | 7 | 2.4e-11 |
| OL@42 | Gemini | typo_mt | 76 | 73 | 5 | 8 | 0.58 |
| OL@42 | Gemini | multilingual_mt | 28 | 59 | 34 | 3 | 1.2e-07 |
| OL@42 | Gemini | multihop_mt | 10 | 28 | 20 | 2 | 0.00012 |
| OL@42 | Gemini | basic_readout_mt | 45 | 64 | 23 | 4 | 0.00031 |
| OL@42 | Gemini | pooled | 159 | 224 | 82 | 17 | 2.2e-11 |

## Matched-layer (L42) comparisons
| comparison | family | A pass | B pass | A-only | B-only | McNemar p |
|---|---|---|---|---|---|---|
| NLA@42 vs OL@42 | typo_mt | 73 | 76 | 6 | 9 | 0.61 |
| NLA@42 vs OL@42 | multilingual_mt | 42 | 28 | 20 | 6 | 0.0094 |
| NLA@42 vs OL@42 | multihop_mt | 15 | 10 | 10 | 5 | 0.3 |
| NLA@42 vs OL@42 | basic_readout_mt | 57 | 45 | 18 | 6 | 0.023 |
| NLA@42 vs OL@42 | pooled | 187 | 159 | 54 | 26 | 0.0023 |
| NLA@42 vs T_qwen | typo_mt | 73 | 45 | 35 | 7 | 1.5e-05 |
| NLA@42 vs T_qwen | multilingual_mt | 42 | 40 | 20 | 18 | 0.87 |
| NLA@42 vs T_qwen | multihop_mt | 15 | 44 | 4 | 33 | 1.1e-06 |
| NLA@42 vs T_qwen | basic_readout_mt | 57 | 53 | 21 | 17 | 0.63 |
| NLA@42 vs T_qwen | pooled | 187 | 182 | 80 | 75 | 0.75 |
| OL@42 vs T_qwen | typo_mt | 76 | 45 | 37 | 6 | 1.6e-06 |
| OL@42 vs T_qwen | multilingual_mt | 28 | 40 | 15 | 27 | 0.088 |
| OL@42 vs T_qwen | multihop_mt | 10 | 44 | 6 | 40 | 3.1e-07 |
| OL@42 vs T_qwen | basic_readout_mt | 45 | 53 | 14 | 22 | 0.24 |
| OL@42 vs T_qwen | pooled | 159 | 182 | 72 | 95 | 0.088 |
| NLA@42 vs T_gem | typo_mt | 73 | 66 | 26 | 19 | 0.37 |
| NLA@42 vs T_gem | multilingual_mt | 42 | 73 | 4 | 35 | 3.4e-07 |
| NLA@42 vs T_gem | multihop_mt | 15 | 75 | 0 | 60 | 1.7e-18 |
| NLA@42 vs T_gem | basic_readout_mt | 57 | 87 | 3 | 33 | 2.3e-07 |
| NLA@42 vs T_gem | pooled | 187 | 301 | 33 | 147 | 2.2e-18 |
| OL@42 vs T_gem | typo_mt | 76 | 66 | 25 | 15 | 0.15 |
| OL@42 vs T_gem | multilingual_mt | 28 | 73 | 6 | 51 | 5.7e-10 |
| OL@42 vs T_gem | multihop_mt | 10 | 75 | 1 | 66 | 9.2e-19 |
| OL@42 vs T_gem | basic_readout_mt | 45 | 87 | 2 | 44 | 3.1e-11 |
| OL@42 vs T_gem | pooled | 159 | 301 | 34 | 176 | 2.7e-24 |

## Hard reader-only hits (no text-only arm, 8 samples, finds the target)
| reader | family | reader pass | hits also found by T_any | hard reader-only hits (T_any ✗) |
|---|---|---|---|---|
| NLA@42 | typo_mt | 73 | 73 | 0 |
| NLA@42 | multilingual_mt | 42 | 42 | 0 |
| NLA@42 | multihop_mt | 15 | 15 | 0 |
| NLA@42 | basic_readout_mt | 57 | 57 | 0 |
| NLA@42 | pooled | 187 | 187 | 0 |
| OL@42 | typo_mt | 76 | 76 | 0 |
| OL@42 | multilingual_mt | 28 | 24 | 4 |
| OL@42 | multihop_mt | 10 | 10 | 0 |
| OL@42 | basic_readout_mt | 45 | 45 | 0 |
| OL@42 | pooled | 159 | 155 | 4 |
| OL grid (any-of-11) | typo_mt | 86 | 86 | 0 |
| OL grid (any-of-11) | multilingual_mt | 65 | 59 | 6 |
| OL grid (any-of-11) | multihop_mt | 34 | 34 | 0 |
| OL grid (any-of-11) | basic_readout_mt | 73 | 73 | 0 |
| OL grid (any-of-11) | pooled | 258 | 252 | 6 |
