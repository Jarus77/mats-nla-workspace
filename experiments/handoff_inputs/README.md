# Inputs for exps 15 and 16 (copied from the sprint's results volume, 24 Oct 2026)

Layout mirrors `rfm-results:/` so the experiment code finds them unchanged:
- `13_new_rows/`: items + L40/42/44 activations for table column and parenthesis (exp 13)
- `07_template_mining/`: the same for the two controls, language and bullet index (exp 07)
- `14_ar_blindspot/`: fresh Oracle Lens readouts and NLA-reconstructor outputs per item (exp 14)
- `03_pipeline_smoke/jspace_basis_L42.pt`: the J-space basis at layer 42 (exp 03)

Modal: `bash experiments/handoff_inputs/upload_to_modal.sh`. Local GPU box: copy the four folders into `/results/`.
