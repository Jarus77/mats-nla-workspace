Thresholds: readable ≥ 0.8, span ≥ 0.8, statement ≥ 0.3, used z ≥ 3.0, J-enriched ≥ 1.5. Predicted cells: H = predicted high, L = predicted low.

| row | next-token set | constrains next | short value | readable | span (frozen probe) | stmt | used z | J-enr | NLA prompted | NLA states | NLA names complement | OL prompted | OL exhibits | OL states | J-lens prompted |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
| py_string | '"' "'" | yes | yes | 1.00* | 0.84* | -0.02 | 10.72* | 1.06 | H | L | L | H | H | L | L |
| js_string | '"' "'" | yes | yes | 0.98* | 0.77 | 0.01 | -1.97 | 1.16 | H | L | L | L | L | L | L |
| latex_math | '$' | yes | yes | 1.00* | 0.85* | 0.02 | 24.66* | 1.20 | H | L | L | H | H | L | L |
| py_arg | ',' ')' | yes | yes | 0.95* | 0.88* | 0.21 | -1.24 | 1.06 | H | L | H | H | H | L | L |
| json_depth | '}' | yes | yes | 1.00* | 0.92* | 0.13 | -1.75 | 1.33 | H | L | H | H | H | L | L |
| md_heading | none | no | yes | 0.72 | 0.62 | 0.06 | n/a | 1.40 | L | L | H | L | L | L | L |
| md_list | none | no | yes | 0.49 | 0.50 | -0.03 | n/a | 1.04 | L | L | H | L | L | L | L |
| code_lang | none | no | yes | 0.99* | 0.99* | 0.30 | n/a | 1.59* | H | L | H | H | H | L | H |

* = passes its threshold.
