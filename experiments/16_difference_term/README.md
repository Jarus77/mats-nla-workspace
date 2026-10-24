# 16 — Difference-term swap (built 24 Oct 2026, not yet run)

Neel's experiment (23 Oct): freeze the OLens part of the activation, replace the remainder from another
prompt, re-run the NLA. Done in raw space with the NLA's reconstructor (the OL-AR output is whitened and
its moments are gated).

Files: `spec16.py`, `lib16.py` (`test_lib16.py`), `modal16.py` (app `pm9`, stages s1 / s2 / analyze), `env.py`.
```
modal run experiments/16_difference_term/modal16.py --stage s1        # H200, ~20 min (4 rows)
modal run experiments/16_difference_term/modal16.py --stage s2        # H100, ~10 min
modal run experiments/16_difference_term/modal16.py --stage analyze   # CPU
modal volume get rfm-results 16_difference_term/ experiments/16_difference_term/results/
```
Prediction: for language the NLA readout follows the OLens part; for table column / parenthesis it
follows the swapped residual (the information OLens's output does not carry).
