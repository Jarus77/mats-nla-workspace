## Naming the state at random read points (strict, state-specific phrasing; share of readouts naming the state when it holds vs when it does not)

| state | n holds / not | Oracle Lens: named when holds | when not | NLA: named when holds | when not |
|---|---|---:|---:|---:|---:|
| in_string | 134 / 366 | 0.00 | 0.00 | 0.03 | 0.01 |
| in_math | 8 / 492 | 0.00 | 0.00 | 0.00 | 0.01 |
| paren_depth | 85 / 415 | 0.00 | 0.00 | 0.13 | 0.01 |
| brace_depth | 46 / 454 | 0.00 | 0.00 | 0.04 | 0.00 |
| list_depth | 18 / 482 | 0.00 | 0.01 | 0.50 | 0.09 |
| heading_level | 4 / 496 | 0.00 | 0.00 | 0.00 | 0.00 |

## Any structural vocabulary at all (loose regex; over-counts content words such as "open", "math", "level")

| kind | n | Oracle Lens | NLA |
|---|---:|---:|---:|
| js | 100 | 0.09 | 0.43 |
| json | 100 | 0.13 | 0.45 |
| md | 100 | 0.09 | 0.57 |
| py | 100 | 0.11 | 0.46 |
| tex | 100 | 0.27 | 0.60 |

## Carrying at random read points (AUROC; readable = probe on the activation; text bag = classifier on the readout alone)

| kind | state | n yes / no | readable | Oracle Lens readout | NLA readout |
|---|---|---|---:|---:|---:|
| js | paren_depth | 38 / 62 | 0.67 | 0.59 | 0.64 |
| js | brace_depth | 22 / 78 | 0.58 | 0.62 | 0.57 |
| json | brace_depth | 22 / 78 | 0.49 | 0.53 | 0.47 |
| py | paren_depth | 27 / 73 | 0.66 | 0.54 | 0.46 |
| code (py+js) | in_string | 35 / 165 | 0.86 | 0.72 | 0.79 |
| code (py+js) | paren_depth | 65 / 135 | 0.69 | 0.54 | 0.57 |
| code (py+js) | brace_depth | 24 / 176 | 0.69 | 0.66 | 0.68 |
| all | in_string | 134 / 366 | 0.96 | 0.84 | 0.92 |
| all | paren_depth | 85 / 415 | 0.76 | 0.67 | 0.68 |
| all | brace_depth | 46 / 454 | 0.73 | 0.67 | 0.71 |
