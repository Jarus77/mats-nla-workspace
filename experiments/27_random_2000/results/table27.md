## Naming the state at random read points (strict, state-specific phrasing; share of readouts naming the state when it holds vs when it does not; 95% Wilson)

| state | n holds / not | Oracle Lens: named when holds | when not | NLA: named when holds | when not | exp 23: OL / NLA when holds (n) |
|---|---|---:|---:|---:|---:|---:|
| in_string | 486 / 1453 | 0.00 [0.00, 0.01] | 0.00 | 0.02 [0.01, 0.03] | 0.01 | 0.00 / 0.03 (134) |
| in_math | 22 / 1917 | 0.00 [0.00, 0.15] | 0.00 | 0.09 [0.03, 0.28] | 0.01 | 0.00 / 0.00 (8) |
| paren_depth | 318 / 1621 | 0.00 [0.00, 0.01] | 0.00 | 0.08 [0.05, 0.11] | 0.01 | 0.00 / 0.13 (85) |
| brace_depth | 185 / 1754 | 0.00 [0.00, 0.02] | 0.00 | 0.01 [0.00, 0.03] | 0.01 | 0.00 / 0.04 (46) |
| list_depth | 109 / 1830 | 0.01 [0.00, 0.05] | 0.00 | 0.53 [0.44, 0.62] | 0.10 | 0.00 / 0.50 (18) |
| heading_level | 15 / 1924 | 0.00 [0.00, 0.20] | 0.00 | 0.20 [0.07, 0.45] | 0.01 | 0.00 / 0.00 (4) |

## Any structural vocabulary at all (loose regex; over-counts content words such as "open", "math", "level")

| kind | n | Oracle Lens | NLA |
|---|---:|---:|---:|
| js | 400 | 0.14 | 0.46 |
| json | 400 | 0.11 | 0.46 |
| md | 400 | 0.15 | 0.55 |
| py | 400 | 0.13 | 0.50 |
| tex | 339 | 0.26 | 0.67 |

## Carrying at random read points (AUROC; readable = probe on the activation; text bag = classifier on the readout alone; the all-kinds string row is confounded with document kind, the code-only row is the honest number)

| kind | state | n yes / no | readable | Oracle Lens readout | NLA readout | exp 23: readable / OL / NLA |
|---|---|---|---:|---:|---:|---:|
| js | in_string | 51 / 349 | 0.92 | 0.78 | 0.80 | n/a |
| js | paren_depth | 132 / 268 | 0.73 | 0.63 | 0.69 | 0.67 / 0.59 / 0.64 |
| js | brace_depth | 98 / 302 | 0.67 | 0.61 | 0.59 | 0.58 / 0.62 / 0.57 |
| json | brace_depth | 83 / 317 | 0.74 | 0.61 | 0.63 | 0.49 / 0.53 / 0.47 |
| md | paren_depth | 78 / 322 | 0.88 | 0.80 | 0.87 | n/a |
| md | list_depth | 109 / 291 | 0.73 | 0.73 | 0.83 | n/a |
| py | in_string | 51 / 349 | 0.80 | 0.60 | 0.77 | n/a |
| py | paren_depth | 95 / 305 | 0.70 | 0.56 | 0.60 | 0.66 / 0.54 / 0.46 |
| tex | in_math | 22 / 317 | 0.85 | 0.75 | 0.85 | n/a |
| code (py+js) | in_string | 102 / 698 | 0.86 | 0.72 | 0.81 | 0.86 / 0.72 / 0.79 |
| code (py+js) | paren_depth | 227 / 573 | 0.75 | 0.60 | 0.61 | 0.69 / 0.54 / 0.57 |
| code (py+js) | brace_depth | 102 / 698 | 0.80 | 0.73 | 0.72 | 0.69 / 0.66 / 0.68 |
| all | in_string | 486 / 1453 | 0.95 | 0.87 | 0.93 | 0.96 / 0.84 / 0.92 |
| all | in_math | 22 / 1917 | 0.91 | 0.96 | 0.94 | n/a |
| all | paren_depth | 318 / 1621 | 0.85 | 0.75 | 0.76 | 0.76 / 0.67 / 0.68 |
| all | brace_depth | 185 / 1754 | 0.85 | 0.78 | 0.77 | 0.73 / 0.67 / 0.71 |
| all | list_depth | 109 / 1830 | 0.92 | 0.85 | 0.91 | n/a |

## Blind judge on the in-string readouts (judge sees the readout only; yes = it states the position is inside a string literal)

| source | reader | n in-string / not | judge yes when in string | when not | regex when in string | regex when not | unparsed |
|---|---|---|---:|---:|---:|---:|---:|
| e27 | ol | 469 / 475 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.00 | 0.00 | 28 |
| e27 | nla | 473 / 482 | 0.22 [0.19, 0.26] | 0.02 [0.01, 0.04] | 0.02 | 0.00 | 17 |
| e23 | ol | 129 / 129 | 0.00 [0.00, 0.03] | 0.00 [0.00, 0.03] | 0.00 | 0.00 | 10 |
| e23 | nla | 129 / 132 | 0.19 [0.13, 0.27] | 0.02 [0.00, 0.05] | 0.03 | 0.01 | 7 |
| pooled | ol | 598 / 604 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.00 | 0.00 | 38 |
| pooled | nla | 602 / 614 | 0.22 [0.18, 0.25] | 0.02 [0.01, 0.03] | 0.02 | 0.01 | 24 |

(judge: google/gemini-3.8-flash, T=0.0; 2480 of 2480 rows judged, 62 unparsed)
