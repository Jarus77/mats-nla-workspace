## Claim labels by reader (Gemini 3.8 Flash judge; 100 readouts per reader, 25 per family)
| reader | readouts | claims | claims/readout | contradicted | restate | target/latent | inferable | unsupported | text-verifiable precision [95% CI] | unsupported share | readouts with ≥1 contradicted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NLA@42 | 100 | 844 | 8.4 | 0.36 | 0.09 | 0.11 | 0.38 | 0.06 | 0.62 [0.59,0.65] | 0.06 | 97/100 |
| OL@42 | 100 | 675 | 6.8 | 0.35 | 0.06 | 0.18 | 0.14 | 0.28 | 0.52 [0.46,0.59] | 0.28 | 77/100 |
| TextNLA(prompt,Qwen) | 100 | 587 | 5.9 | 0.08 | 0.23 | 0.18 | 0.49 | 0.02 | 0.92 [0.89,0.94] | 0.02 | 34/100 |

## Text-verifiable precision by family
| reader | typo_mt | multilingual_mt | multihop_mt | basic_readout_mt |
|---|---|---|---|---|
| NLA@42 | 0.58 | 0.68 | 0.59 | 0.62 |
| OL@42 | 0.84 | 0.54 | 0.39 | 0.40 |
| TextNLA(prompt,Qwen) | 0.89 | 0.91 | 0.92 | 0.95 |

## Judge agreement: Gemini 3.8 Flash vs kimi-k3 (Fireworks) on 40 claims, stratified by reader x Gemini label
| label set | raw agreement | Cohen's κ [95% bootstrap CI] |
|---|---|---|
| 5-way | 0.78 | 0.72 [0.56,0.87] |
| 3-way (text-true / contradicted / unsupported) | 0.88 | 0.77 [0.57,0.94] |

Confusion (rows Gemini, cols kimi-k3):
| Gemini \ kimi | contradicted | restate | target/latent | inferable | unsupported |
|---|---|---|---|---|---|
| contradicted | 6 | 0 | 0 | 1 | 1 |
| restate | 1 | 6 | 1 | 0 | 0 |
| target/latent | 0 | 0 | 8 | 0 | 0 |
| inferable | 0 | 1 | 2 | 5 | 0 |
| unsupported | 1 | 0 | 0 | 1 | 6 |