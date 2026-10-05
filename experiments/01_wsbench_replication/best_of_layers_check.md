# Best-of-layers mechanics check (olens_grid / jlens_grid; per-item vs per-family selection)

## olens_grid

| family | any-layer per-item (published rule) | best single layer (per-family) | L40 | L44 | per-layer curve 20..60 |
|---|---|---|---|---|---|
| typo_mt | 0.86 (pub value 0.86) | 0.78 @L24 | 0.72 | 0.70 | 0.73 0.78 0.71 0.68 0.73 0.72 0.70 0.66 0.74 0.78 0.76 |
| multilingual_mt | 0.65 (pub value 0.65) | 0.43 @L56 | 0.26 | 0.26 | 0.02 0.03 0.09 0.10 0.18 0.26 0.26 0.30 0.39 0.43 0.38 |
| multihop_mt | 0.34 (pub value 0.34) | 0.18 @L56 | 0.08 | 0.10 | 0.08 0.05 0.05 0.07 0.14 0.08 0.10 0.08 0.17 0.18 0.13 |
| basic_readout_mt | 0.73 (pub value 0.73) | 0.62 @L56 | 0.42 | 0.48 | 0.22 0.23 0.27 0.30 0.43 0.42 0.48 0.46 0.58 0.62 0.61 |

## jlens_grid

| family | any-layer per-item (published rule) | best single layer (per-family) | L40 | L44 | per-layer curve 20..60 |
|---|---|---|---|---|---|
| typo_mt | 0.15 (pub value 0.15) | 0.08 @L40 | 0.08 | 0.07 | 0.01 0.03 0.05 0.01 0.03 0.08 0.07 0.04 0.01 0.03 0.03 |
| multilingual_mt | 0.08 (pub value 0.08) | 0.06 @L60 | 0.00 | 0.00 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.01 0.06 |
| multihop_mt | 0.10 (pub value 0.10) | 0.06 @L56 | 0.00 | 0.00 | 0.00 0.00 0.01 0.01 0.02 0.00 0.00 0.00 0.00 0.06 0.03 |
| basic_readout_mt | 0.24 (pub value 0.24) | 0.15 @L60 | 0.04 | 0.03 | 0.00 0.00 0.01 0.00 0.00 0.04 0.03 0.01 0.02 0.04 0.15 |

