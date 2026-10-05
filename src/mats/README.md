# `mats`: sprint pipeline library

What the prereg (`notes/prereg_v0.md` rev 2 + D7) needs to run the main experiment, in small modules.
The end-to-end example is `experiments/03_pipeline_smoke/smoke_modal.py`.

**Layer convention (K0).** "Layer L" means the output of decoder block L, which is `model.layers[L]`'s
forward-hook output and HF `hidden_states[L+1]`. Every API takes L, never L+1.

**Read layers (D7).** L42 is primary for all three readers, Oracle Lens included (its prompt says
"layer 42"). OL at 40/44 is sensitivity only. The OL layer sweep for natural items covers
20..60 in steps of 4, plus 42. Never select a layer per item: score each layer separately.

| module | what it gives you |
|---|---|
| `capture.py` | `capture(model, seqs, layers, positions)` gives `{L: [N, P, d]}`: batched, right-padded, early exit after the deepest layer. `logits_at`, `next_token_kl`. Arm U hooks are context managers: `steer(model, L, vectors, positions, dose)`, `clamp(model, L, direction, value, positions)` and the generic `edit`. Arm R (read-layer) helpers are pure functions: `steer_vectors`, `read_layer_steer`, `clamp_vectors` (one direction or a subspace), and `doses_from_projection` (q90−m, q99−m, 2(q99−m)). **Steering preserves the norm by default**, h' = (h+Δv)·‖h‖/‖h+Δv‖; pass `preserve_norm=False` for plain addition. Note that OL and the NLA see only unit(input), so they are invariant to this rescaling. `centred_steer_vectors`, h − (v·h)v ± Δv, gives equal norms on both ± sides; it is the construction that removes the dilution artefact (03 RESULTS, "Norm-preserving re-run"). |
| `jspace.py` | `jspace_basis(J, W_U, gain)` returns the top right singular vectors of `W_U diag(g) J`, ported from 02's s8_rstar. Also `project_out`, `project_onto`, `enrichment` (random = 1.0), `j_removed_direction`, the random equal-rank controls (`random_subspace`, `random_direction(kind=nonj/j/iso)`), `load_jlens_layers` (mmap, selected layers only), `norm_gain`, `load_basis_file` (reads 02's `jspace_basis_L{38,42}.pt`), and the NNLS robustness tools `jlens_atoms`, `nn_pursuit`, `project_out_span`. |
| `probes.py` | `fit_logreg`, `cv_auroc` (grouped k-fold; `transform=` for J-removal, frozen or retrained), `mean_diff_direction`. |
| `metrics.py` | `auroc`, `auroc_ci` (cluster bootstrap), `cluster_bootstrap`, `follow_feature` (± steered pairs, optional 0-dose subtraction, paired win rate), `follow_text`, `fraction_of_probe(_ci)`, `paired_null_auroc` (real vs matched null, paired bootstrap), `page_trend_test` (matches scipy's asymptotic test), `fve_guard` (the one-sided §4.4 rule). Null bands: `paired_stats`, `null_band` (percentile, two-sided band, z in null-SD units, no subtraction) and `null_band_contrast`, which compares a real ± contrast against K null contrasts. |
| `readers/nla.py` | NLA AV+AR (unchanged class). Batched additions: `logprob_of_strings_batch`, `paired_logodds_batch`, `paired_logodds_templates`. |
| `readers/olens.py` | `load_olens(model)` returns an `OracleLens` with `base()` (adapter off, for capture), `verbalize(acts, layer=42, k, sampling)`, `logprob_of_strings`, `paired_logodds(acts, layer, "- a", "- b")`, `paired_logodds_templates` and `layer_sweep_logodds`. Its constants are `DEFAULT_LAYER=42`, `SENSITIVITY_LAYERS` and `SWEEP_LAYERS`. |
| `readers/jlens.py` | `load_jlens(model, tok, layers)` returns a `JLensReader` with `scores` (cosine readout of record), `readout(k)`, `paired_logodds(acts, layer, words_a, words_b)` (max-variant score difference) and `rank_of`. |
| `textnla.py` | `TextNLALocal` scores base Qwen teacher-forced, using the NLA warm-start instruction, `mode="prompt"` or `"explanation"`. `TextNLAGemini` uses Gemini 3.8 Flash through OpenRouter. OpenRouter returns no logprobs for Gemini, so LO is the logit of an elicited P(A), with both option orders averaged. |

## Rules baked in

- **Paired log-odds keeps all candidates of one activation in the same forward batch** (OL, NLA, TextNLA). Batch composition alone moves bf16 scores by up to 0.26 nats.
- **Candidates differ only in the feature value**, written in each reader's native format:
  - NLA: explanation register after `<explanation>\n`;
  - OL: `- ` bullets;
  - J-Lens: label-token variants.
- **Capture runs with the OL adapter off** (`with ol.base():`).
- **J-Lens loads only the requested layers**, so the subject model, OL and J-Lens all fit together on one H100. The NLA needs its own H200 container: AV + AR is about 90 GB.

## Typical flow

```python
from mats import capture as C, jspace as JS, probes as P, metrics as M
from mats.readers.olens import load_olens
from mats.readers.jlens import load_jlens

ol = load_olens()                                       # Qwen3.6-27B + OL LoRA
with ol.base():
    acts = C.capture(ol.model, seqs, [42], [[-1]] * len(seqs))
jl = load_jlens(ol.model, ol.tok, layers=[42])
basis = JS.jspace_basis(jl.J[42], jl.W_U, JS.norm_gain(jl.final_norm))
u = JS.j_removed_direction(basis, P.mean_diff_direction(acts[42][:, 0], y), r=384)
doses = C.doses_from_projection(acts[42][:, 0] @ u)
plus, minus = acts[42][:, 0] + doses[1] * u, acts[42][:, 0] - doses[1] * u
lo = lambda A: ol.paired_logodds(A, 42, "- French language", "- German language")
M.follow_feature(lo(plus), lo(minus), clusters=item_ids)
```

## Tests

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python numpy scipy pytest torch
.venv/bin/python -m pytest tests -q        # offline, CPU, a few seconds
```

The tests cover metrics, the J-space math against an SVD, capture/steer/clamp on a toy causal
decoder (checked against `hidden_states[L+1]` with padding), the J-Lens score definitions and the
probes. The GPU paths (OL/NLA/TextNLA scoring) are exercised end to end by
`experiments/03_pipeline_smoke`.
