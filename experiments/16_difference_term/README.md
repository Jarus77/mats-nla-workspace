# 16 — Difference-term swap (built 24 Oct 2026, run 25 Oct by Guiv on RunPod; see RESULTS.md)

Neel's experiment (23 Oct): freeze the OLens part of the activation, replace the remainder from another
prompt, re-run the NLA. Done in raw space with the NLA's reconstructor (the OL-AR output is whitened and
its moments are gated).

Files: `spec16.py`, `lib16.py` (`test_lib16.py`), `modal16.py` (app `pm9`, stages s1 / s2 / analyze; opaque Modal
function names), `run_local.py` (no-Modal runner, used for the recorded run), `report16.py` (table + figure), `env.py`.
```
modal run experiments/16_difference_term/modal16.py --stage s1        # H200, 24 min for 4 rows (measured)
modal run experiments/16_difference_term/modal16.py --stage s2        # H100, 12 min (measured)
modal run experiments/16_difference_term/modal16.py --stage analyze   # CPU
modal volume get rfm-results 16_difference_term/ experiments/16_difference_term/results/vol/
python experiments/16_difference_term/report16.py --analysis experiments/16_difference_term/results/analysis16_*.json
```
Prediction: for language the NLA readout follows the OLens part; for table column / parenthesis it
follows the swapped residual (the information OLens's output does not carry).
