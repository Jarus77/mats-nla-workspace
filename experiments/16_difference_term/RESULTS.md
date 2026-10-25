# 16 — Difference-term swap (Neel's experiment), run 25 Oct 2026 by Guiv on RunPod

**Question (Neel, 23 Oct drop-in).** Reconstruct the OLens output and the NLA output as activations;
the NLA's will be better; what is in the difference? Freeze the OLens part, replace the remainder from a
different prompt, and see what the readers now report.

**Design (spec16.py).** Layer 42, 200 items per row. R_ol = NLA-AR(OLens bullets, exp 14), R_nla =
NLA-AR(NLA explanation, exp 14), res = h − R_ol (the difference term), resn = R_nla − R_ol. Partner j(i)
of the opposite class from a different cluster (seed 16). Vectors: h, R_nla, R_ol (= ol_only),
swap_res = R_ol_i + res_j, swap_ol = R_ol_j + res_i, swap_resn = R_ol_i + resn_j. Readout: NLA and
OLens@42 prompted log-odds on the frozen phrasings (mean over 4, and the fixed label). Statistic: AUROC
of the readout against i's label and against j's label. Probes retrained on each component. Rows: two
NLA-leads (table column, parenthesis) and two controls (language, list index).

## Headline (mean-phrasing readout, AUROC against the **donor** j's label)

| row | NLA on h | NLA on OL-part alone | NLA, difference term from donor | NLA, OLens part from donor | OL, difference term from donor | OL, OLens part from donor |
|---|---:|---:|---:|---:|---:|---:|
| table column | 1.00 | 0.53 | **0.98** | **0.01** | 0.61 | 0.42 |
| parenthesis | 0.96 | 0.67 | **0.87** | **0.12** | 0.72 | 0.38 |
| language (control) | 1.00 | 1.00 | **0.10** | **0.92** | 0.10 | 0.91 |
| list index (control) | 1.00 | 0.69 | **0.98** | **0.03** | 0.64 | 0.39 |

Reading the bold pair per row: near 1 in the first and near 0 in the second means the readout lives in
the difference term; the reverse means it lives in the part the OLens output reconstructs. Fixed-label
phrasings agree (table column 0.76 / 0.20, parenthesis 0.84 / 0.19, language 0.10 / 0.92, list index
0.84 / 0.20). swap_resn (the NLA-reconstruction difference term) transfers the same way (0.95 / 0.77 /
0.06 / 0.83). Figure: `results/fig16_difference_term.png`.

## What it says

1. **Position-in-text state is in the difference term.** For table column, parenthesis and list index
   the NLA readout follows whichever prompt supplied the difference term (0.98 / 0.87 / 0.98) and ignores
   whichever supplied the OLens part (0.01 / 0.12 / 0.03). The OLens part alone leaves the NLA at or
   near chance for table column (0.53) and weak for parenthesis (0.67). What the NLA reports about position
   is information the OLens output does not carry back into the activation. Neel's prediction, confirmed.
2. **Language is in the OLens part.** The control behaves the opposite way for both readers: the readout
   follows the OLens part (0.92 / 0.91) and ignores the difference term (0.10 / 0.10). The swap design
   itself is sound: a concept the OLens output does carry transfers with it.
3. **OLens's own readout is weak on position and also leans on the difference term.** OLens reads table
   column / parenthesis / list index at 0.72 / 0.70 / 0.99 on the true activation but goes to 0.61 / 0.72
   / 0.64 for the donor when the difference term is swapped, and 0.42 / 0.38 / 0.39 when its own part is
   swapped. The OLens reader's partial signal on position therefore comes from the same component its own
   text fails to reconstruct. Together with exp 14 (OL-AR keeps 1.00 / 0.93 from the true span, 0.66 /
   0.60 from OL bullets): OLens can read position partially, says nothing about it, and what it says does
   not re-induce it.
4. **Reconstruction quality.** FVE of R_ol against h: −0.26 (table column), +0.17 (parenthesis), +0.04
   (language), −0.13 (list index); R_nla: 0.50 / 0.71 / 0.61 / 0.57. Retrained probes still find the
   concept in R_ol for every row (0.88 / 0.76 / 1.00 / 0.93) and in res (1.00 / 0.91 / 0.94 / 1.00): a
   linear trace survives in the OLens part, but it is not what either reader's readout uses. "Difference
   term carries position" is a statement about the readers' readouts, not about linear decodability.

**Predictions (README) vs outcome.** Both confirmed: language follows the OLens part; table column and
parenthesis follow the difference term. Unpredicted: list index (an OLens-reads control in exp 12/13)
behaves like the position rows, i.e. OLens reads it on h (0.99) but from the difference term.

## Caveats

- Both reconstructions are made with the NLA's reconstructor, so "the OLens part" means "what the NLA's
  AR can rebuild from OLens's words". The OL-AR's own space is whitened and its moments gated (exp 14).
- One layer and position; OLens bullets were generated at its trained layer 44 (exp 14) and read at 42.
- Prompted readouts only; partner pairing is one seeded draw.

## Run record

Run by Guiv's agent on RunPod, one NVIDIA H200 NVL (144 GB), checkout a8e41f4 (the handoff commit),
`run_local.py`: s1 24 m 27 s, s2 11 m 56 s, analyze 24 s; 53 min wall clock including provisioning;
estimated USD 3.5 (7.5 with earlier failed provisioning attempts). The pinned `hf` CLI dropped the
`av_base/*` pattern; the same six patterns were completed from the same snapshot with the pinned
`huggingface_hub` (documented in his logs); no code, constant, seed or template changed. CPU tests 12/12.
Deliverables (12 outputs, SHA-256 verified against his manifest) are on `rfm-results:/16_difference_term/`
and in `results/` (analysis JSON, stage prints; `.pt` under `results/vol/`, untracked). Report:
`python report16.py --analysis results/analysis16_*.json`.
