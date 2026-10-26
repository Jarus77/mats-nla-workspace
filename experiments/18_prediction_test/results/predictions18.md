Thresholds: readable ≥ 0.8, span ≥ 0.8, statement ≥ 0.3, used z ≥ 3.0, J-enriched ≥ 1.5. Predicted cells: H = predicted high, L = predicted low.

| row | next-token set | constrains next | short value | readable | span (frozen probe) | stmt | used z | J-enr | NLA prompted | NLA states | NLA names complement | OL prompted | OL exhibits | OL states | J-lens prompted |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
| json_depth | '}' | yes | yes | 1.00* | 0.99* | 0.13 | 2.61 | 1.10 | H | L | L | H | H | L | L |
| latex_math | '$' | yes | yes | 1.00* | 1.00* | 0.01 | 22.15* | 1.42 | H | H | L | H | H | L | L |
| py_string | '"' "'" | yes | yes | 1.00* | 0.98* | 0.01 | -12.24* | 1.01 | H | H | L | H | H | L | L |
| call_arg | ',' ')' | yes | yes | 1.00* | 1.00* | 0.09 | 9.05* | 1.04 | H | H | L | H | H | L | L |
| md_heading | none | no | yes | 1.00* | 0.68 | -0.01 | n/a | 1.09 | H | L | L | L | L | L | L |
| csv_col | ',' '\n' | yes | yes | 1.00* | 0.71 | 0.11 | 0.29 | 0.93 | H | L | L | L | L | L | L |
| html_depth | '</' | yes | yes | 1.00* | 0.77 | 0.02 | -0.98 | 1.10 | H | L | L | L | L | L | L |
| stance_word | none | no | yes | 0.45 | 0.46 | 0.14 | n/a | 1.51* | L | L | L | L | L | L | H |
| ij2_lang | none | no | yes | 0.99* | 1.00* | 0.51* | n/a | 1.38 | H | L | L | H | H | L | H |
| bk_quote | '"' | yes | yes | 0.82* | 0.68 | 0.23 | 3.83* | 1.62* | H | H | L | L | L | L | H |

* = passes its threshold.
