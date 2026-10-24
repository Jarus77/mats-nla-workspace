# 15 — AR-mediated steering (built 24 Oct 2026, not yet run)

Question: does the model use the table-column / parenthesis state the NLA reports, and can the
reconstructor build a steering vector for that state from text (exhibited vs stated)?

Files: `spec15.py` (design constants), `lib15.py` (pure helpers; `test_lib15.py`), `modal15.py` (app `vx4`,
stages prep / behave / score_nla / score_ol / analyze), `analyze15.py` (`test_analyze15.py`), `env.py`.

Run order (two GPUs max):
```
modal run experiments/15_ar_steering/modal15.py --stage prep                       # H200, ~15 min
modal run experiments/15_ar_steering/modal15.py --stage behave &                   # H100, ~60 min
modal run experiments/15_ar_steering/modal15.py --stage score_nla                  # H200, ~15 min
modal run experiments/15_ar_steering/modal15.py --stage score_ol                   # H100, ~15 min
modal run experiments/15_ar_steering/modal15.py --stage analyze                    # CPU
modal volume get rfm-results 15_ar_steering/ experiments/15_ar_steering/results/
```
Every stage skips a feature whose output exists on `rfm-results:/15_ar_steering/`.
Predictions (proposal §4): class-mean and AR-from-exhibited-text vectors move the model and the NLA
readout; the AR-from-statement vector moves nothing; OLens moves only through continuation; J-lens flat.
