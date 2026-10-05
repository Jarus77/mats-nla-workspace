# Decision log

## 2026-10-05 — D1: problem and design
- Problem: Neel's "Is NLA's extra really the non-workspace stuff?" on Qwen3.6-27B.
- Design: unified source attribution (user chose it over a narrower option).
- Overlap: a concurrent pilot (LittleMeHere/nla-vs-lenses). We differentiate on rigor and on natural non-J content, cite the pilot in paragraph 1, and don't copy blindly.

## 2026-10-05 — D2: revision after Fable critique (handoff #1)
Accepted:
1. **Attribution is a 2×2, not 4 buckets.** Content location (J / non-J) × reader mechanism (tracks activation / tracks text), plus "unsupported". Measured at population level by intervention sensitivity (move the feature with text fixed; move the text with the feature fixed). The old buckets weren't a partition: e.g. character count is non-J *and* deterministic from the text.
2. **Cut (B) split-injection.** Viswanath already has the 70B result, it's the pilot's next step, and it's the most out-of-distribution tool. Dose-response steering along natural probe directions is its on-distribution version. Optional 20-item replication if time allows.
3. **Headline = (A)**: 2–3 natural non-J features plus 2 in-J calibration features, under dose-response steering. Readers are reported as a fraction of what the probe recovers.
4. **Self-report is not a gate for non-J.** The criterion is surviving J-space removal plus absence from J-Lens. Phase 1 showed self-report failure is often an elicitation failure (colours are in J-space yet never named).
5. **(C) plants by moving the feature direction.** Not by adding J-Lens directions (trivial for J-Lens), and not by full-residual patching (that carries the counterfactual prompt along with it).
6. **Doses at p90, p99 and 2×p99** of the natural projection. Report the norm as a fraction of residual norm, plus next-token KL. Use the NLA reconstructor's FVE as an out-of-distribution detector: a dose is void if FVE leaves the pre-registered margin (new kill check).
7. **Scoring by paired log-odds**: candidates differ only in the feature value and are written in each reader's native format; AUROC is measured against a matched null. "Judged precision" is renamed **text-verifiable precision** (unsupported ≠ false).
8. **NLA re-solving ceiling**: a text-only LLM (TextNLA) given only the NLA's own explanation.
9. **Layer confound**: NLA at 42 vs Oracle Lens at 40/44, so report feature probe accuracy at 40/42/43/44.

Lead's caveats on the critique (sent back to Fable in handoff #2):
- **A linear probe is not a true ceiling.** Readers are nonlinear 27B decoders and could exceed it. Call it a reference, and show cases where a reader beats the probe as a finding, not an error.
- **On natural data, a text-tracking reader can get non-J features right** (character count is computable from the text). So results on natural data alone can't show "reads non-J". Only the steering arm can, and the write-up should lean on it.
- **The FVE out-of-distribution detector is partly circular.** The reconstructor was co-trained with the NLA verbalizer, so a high FVE on a steered activation shows the pair handles it, not that the NLA "reads" it. Pair it with a reconstructor-independent check: next-token KL, and whether the probe still decodes the intended value.
- **Paired log-odds for J-Lens** (no decoder): use the difference in J-Lens projection scores between the two candidate tokens. That's only comparable to the decoders after AUROC normalisation, so say so.
