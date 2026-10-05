# Two channels of self-report: code, data, and figures

Replication material for the workshop paper. Everything here was produced by the
code in `code/`; nothing is hand-edited.

## What the experiment does

Inject the Jacobian-lens direction of a concept into the residual stream of a
neutral passage, read the concept's lens rank downstream as ground truth for
presence, and read the model's yes/no and named-concept report from its logits.
Model: Qwen3-8B (Apache-2.0). Lens: Neuronpedia `jacobian-lens`, Qwen3-8B,
Salesforce-wikitext fit (Apache-2.0), used unchanged.

## Data (`data/`)

One JSON line per trial unless noted. Ranks are 0-based: rank 0 is the top of
the vocabulary.

| file | rows | what |
|---|---|---|
| `results_main.jsonl` | 9,000 | main matrix. 50 concepts x 20 passages x gamma {0, 0.05, 0.1, 0.2, 0.4}, concept and isotropic-random arms |
| `results_ext.jsonl` | 4,000 | same design at gamma {0.6, 0.8}, both arms. Not pre-registered |
| `results_controls.jsonl` | 1,200 | gamma 0.4. `concept_all` (every battery concept scored in the slot), `jrandom` (J-space random token injected), `nocase` (countries, case constraint removed) |
| `results_modelB.jsonl` | 2,000 | Qwen3-4B, same prompts, no injection, both elicitations |
| `results_threeway.jsonl` | 3,500 | three-option elicitation that failed validation. Kept because the failure is reported |
| `layer_scan.json` | 250 | gamma 0.4, lens rank at all 35 source layers per trial |
| `geometry.json` | 50 | per concept: norm of `v_c` and cosine to the layer-22 lens direction, at layers 12, 13, 14 |
| `check3_band.json` | - | per-layer top-1 next-token accuracy and excess kurtosis, J-lens and logit lens |
| `blurt_sample.json` | 500 | free generations at gamma 0.4, passages 0 and 1 |

Common fields: `kind`, `concept`, `category`, `passage`, `gamma`, `lens_rank`
(J-lens rank of the concept at source layer 22, last user-turn token), `p_yes`
(forced choice over yes/no token variants), `mass` (total probability on valid
yes/no answers, the coherence measure), `rank_c` (rank of the concept in the
CONCEPT slot). `results_controls.jsonl` adds `rank_all`, `lens_rank_vec`,
`rank_vec`, `rank_c_lower`, `vec_token`, `top5`.

Definitions used throughout: landed = `lens_rank <= 25`; named = `rank_c < 5`;
detection = `p_yes > 0.5 and rank_c < 5`; trials with `mass < 0.5` are excluded
by pre-registration.

## Code (`code/`)

- `mech-interp.ipynb` walks the whole method from loading the lens to the main
  matrix, one step per cell, including the three verification checks.
- `workshop_runs.py` runs the follow-ups headlessly on Modal: `gamma_ext`,
  `controls`, `model_b`, `threeway`, `threeway_v2`, `binary_cs`. Each writes a
  resumable JSONL, so an interrupted run continues where it stopped.
- `jverify.py` is the two-hop verification from Section 4.
- `make_figs.py` regenerates every figure and CSV from `data/`, then
  independently recomputes each plotted number and exits non-zero on a mismatch.

Requires one 24 GB GPU. The main matrix takes 27 minutes on an NVIDIA L4; all
runs together take about two hours.

## Figures (`figures/`)

`fig1` dose response and naming-vs-cutoff; `fig2` band localisation; `fig3`
per-concept landed against named; `fig4` layer scan; `fig5` within-category
slopes; `fig6` bimodality of the random arm. Each `figN_data.csv` holds the
exact numbers plotted.

## Reproducing

```
python code/make_figs.py            # figures from the released data
modal run code/workshop_runs.py::controls   # re-run a control arm (needs the volume)
```

The Modal scripts expect the Qwen3 weights and the lens on a volume mounted at
`/models`; see the paths at the top of `workshop_runs.py`.
