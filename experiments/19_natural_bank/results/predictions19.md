Thresholds: readable ≥ 0.8, span ≥ 0.8, statement ≥ 0.3, used z ≥ 3.0, J-enriched ≥ 1.5. Predicted cells: H = predicted high, L = predicted low.

| row | next-token set | constrains next | short value | readable | span (frozen probe) | stmt | used z | J-enr | NLA prompted | NLA states | NLA names complement | OL prompted | OL exhibits | OL states | J-lens prompted |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
| py_string | '"' "'" | yes | yes | 1.00* | 0.85* | 0.04 | 6.56* | 1.17 | H | L | L | H | H | L | L |
| js_string | '"' "'" | yes | yes | 0.99* | 0.82* | 0.01 | 0.70 | 1.22 | H | L | L | H | H | L | L |
| latex_math | '$' | yes | yes | 0.99* | 0.83* | 0.05 | 25.55* | 1.31 | H | L | L | H | H | L | L |
| py_arg | ',' ')' | yes | yes | 0.99* | 0.89* | 0.23 | 3.16* | 1.10 | H | L | H | H | H | L | L |
| json_depth | '}' | yes | yes | 0.99* | 0.87* | 0.14 | -1.98 | 1.29 | H | L | H | H | H | L | L |
| html_depth | '</' | yes | yes | 0.99* | 0.74 | 0.04 | -4.62* | 1.32 | H | L | H | L | L | L | L |
| md_heading | none | no | yes | 0.63 | 0.51 | 0.03 | n/a | 1.48 | L | L | H | L | L | L | L |
| md_list | none | no | yes | 0.65 | 0.64 | 0.07 | n/a | 1.42 | L | L | H | L | L | L | L |

* = passes its threshold.
