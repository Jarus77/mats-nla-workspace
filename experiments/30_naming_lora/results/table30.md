| prediction | measured | line | verdict |
|---|---|---|---|
| P1 gate FVE, statement arm vs control | S 0.507, C 0.512, base 0.504 | |S - C| <= 0.02 | PASS |
| P2 statement accuracy on held-out clusters (arm S) | py_string 0.47, json_depth 0.69, py_arg 0.41, md_list 0.34 | py_string, json_depth, py_arg >= 0.85; md_list <= 0.65 | FAIL |
| P3 transfer rows (arm S) | js_string accuracy 0.31, latex_math naming 0.03 | js_string >= 0.70; latex_math naming < 0.30 | FAIL |
| P4 exp 23 random points, in-string naming (arm S, lib23 strict regex) | when true 0.396, when false 0.260 | >= 0.50 when the state holds; <= 0.15 when it does not | FAIL |
| P5 within-item FVE, naming minus non-naming | skipped: exp 22's k = 8 readouts carry no per-sample FVE | |mean| <= 0.02 | n/a |

Per row (held-out clusters; arms present: base, C, S):

| row | trained | n | base acc | base naming | base FVE | C acc | C naming | C FVE | S acc | S naming | S FVE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| py_string | yes | 32 | 0.00 | 0.00 | 0.539 | 0.00 | 0.00 | 0.542 | 0.47 | 0.59 | 0.545 |
| py_arg | yes | 32 | 0.00 | 0.00 | 0.335 | 0.00 | 0.00 | 0.356 | 0.41 | 0.50 | 0.326 |
| json_depth | yes | 32 | 0.00 | 0.00 | -0.024 | 0.00 | 0.00 | -0.023 | 0.69 | 0.72 | 0.017 |
| md_heading | yes | 32 | 0.00 | 0.00 | 0.645 | 0.00 | 0.00 | 0.635 | 0.47 | 0.81 | 0.619 |
| md_list | yes | 32 | 0.00 | 0.00 | 0.497 | 0.00 | 0.00 | 0.501 | 0.34 | 0.75 | 0.495 |
| code_lang | yes | 32 | 0.00 | 0.00 | 0.437 | 0.00 | 0.00 | 0.468 | 0.28 | 0.28 | 0.461 |
| js_string | no (transfer) | 32 | 0.00 | 0.00 | 0.542 | 0.00 | 0.00 | 0.565 | 0.31 | 0.53 | 0.544 |
| latex_math | no (transfer) | 32 | 0.00 | 0.00 | 0.585 | 0.00 | 0.00 | 0.592 | 0.03 | 0.03 | 0.595 |

Exp 23 random points, in-string naming by lib23's strict regex: base: true 0.015, false 0.005 (statement check, overall accuracy 0.000) / C: true 0.030, false 0.005 (statement check, overall accuracy 0.000) / S: true 0.396, false 0.260 (statement check, overall accuracy 0.184)

Exp 23 random points by the statement check (lib30's substring test of the py_string statements), by the true label:

| arm | in string: says inside / outside / both / nothing | not in string: says inside / outside / both / nothing |
|---|---|---|
| base | 0.00 / 0.00 / 0.00 / 1.00 (n = 134) | 0.00 / 0.00 / 0.00 / 1.00 (n = 366) |
| C | 0.00 / 0.00 / 0.00 / 1.00 (n = 134) | 0.00 / 0.00 / 0.00 / 1.00 (n = 366) |
| S | 0.12 / 0.24 / 0.00 / 0.64 (n = 134) | 0.04 / 0.21 / 0.00 / 0.75 (n = 366) |

Training: arm S 400 steps, final loss 1.327, 13.4 min; arm C 400 steps, final loss 1.378, 13.4 min (wall time from launch to done in pipeline.log, container start included; train30_*.json's wall_s over-accumulates and is not shown)
