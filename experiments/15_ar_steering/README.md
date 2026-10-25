# 15 — AR-mediated steering (built 24 Oct 2026, run 25 Oct; see RESULTS.md)

Question: does the model use the table-column / parenthesis state the NLA reports, and can the
reconstructor build a steering vector for that state from text (exhibited vs stated)?

Files: `spec15.py` (design constants), `lib15.py` (pure helpers; `test_lib15.py`), `modal15.py` (app `vx4`,
stages prep / behave / score_nla / score_ol / analyze / probe; the Modal function names are opaque on purpose),
`analyze15.py` (`test_analyze15.py`), `report15.py` (table + figure), `run_local.py` (no-Modal runner), `env.py`.

Run order (measured 25 Oct, one GPU at a time with `RFM_MAXC=1`; default cap 2):
```
modal run experiments/15_ar_steering/modal15.py --stage prep                       # H200, ~15 min
modal run experiments/15_ar_steering/modal15.py --stage behave                     # H100, ~70 min (4 chunks)
modal run experiments/15_ar_steering/modal15.py --stage score_nla                  # H200, ~72 min per concept
modal run experiments/15_ar_steering/modal15.py --stage score_ol                   # H100, ~36 min per concept
modal run experiments/15_ar_steering/modal15.py --stage analyze                    # CPU
modal volume get rfm-results 15_ar_steering/ experiments/15_ar_steering/results/vol/
python experiments/15_ar_steering/report15.py --analysis results/analysis15_tb_tablecol.json results/analysis15_tb_paren.json
```
`--stage probe` times two OL scoring conditions (~5 min). `RFM_SCORE_T=2 RFM_SCORE_NULLK=10` would trim a
scoring stage to two phrasings and ten nulls per class (not used in the recorded run).
Every stage skips a feature whose output exists on `rfm-results:/15_ar_steering/`.
Predictions (proposal §4): class-mean and AR-from-exhibited-text vectors move the model and the NLA
readout; the AR-from-statement vector moves nothing; OLens moves only through continuation; J-lens flat.
