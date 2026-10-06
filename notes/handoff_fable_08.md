# Handoff #8 to Fable — 2026-10-07: OL mechanism (S6) came back mixed; two pre-freeze fixes

Paste everything below the line into the Fable 5.1 chat.

---

The rev-4.1 freeze is ready apart from bookkeeping. Then the OL mechanism tests (S6, on the pilot split) finished, and they expose two rule problems I want your call on before freezing. All numbers are from the pilot split, excluded from the main analysis. Candidates were unchanged since they were fixed (sha256s verified).

## Results
- **Gain ratio** (OL continuation shift / model behavioural shift, at relative norm 0.108, adapter on):

  | Direction | Ratio |
  |---|---|
  | col raw | 0.99 [0.83, 1.20] |
  | col 4r* | 1.06 [0.89, 1.26] |
  | random A / random B (both with a behavioural effect) | 1.65 / 1.89 |
  | tense raw | 3.40 |
  | tense 4r* | undefined: the model's next-token tense readout doesn't move (−0.04 nats, z −1.4/−2.8), while OL tracks it at z +12 to +16 |

  J-removal keeps 0.79 of OL's per-unit effect vs 0.75 of the model's, replicating round 5.
- **Indentation** (non-J, causally inert in the model): sub-threshold. The max z is +2.7/+2.6, but OL's shift grows with dose (AUROC 0.73 at 2× the dose) while the model's indentation readout stays flat. It vanishes with the adapter off.
- **Adapter off:** tense collapses just as col does (0.49–0.51, from 0.65–1.00). Both continuation channels need the LoRA.
- **Sanity:** OL nl_cont on 80 fresh items is 7–12% below round 5 (within noise), and the model's newline effect reproduces.

## Verdicts
- **The literal §10.1 rule says READING,** but only because tense 4r*'s ratio is undefined (−15.5) and drags the comparison mean negative. That's an artefact.
- **With a valid-denominator clause** (a direction counts only if the model's behavioural shift is ≥ +3 z, which the rule already requires of the random directions): **INCONCLUSIVE**. The ratios are [0.99, 1.06, 3.40, 1.65, 1.89]. "Acting" fails the ±25% band, and "reading" fails because col isn't the highest.
- **My read:** col specifically looks like *acting* (OL's newline shift is one-for-one with the model's, with the same retention under J-removal). But OL's continuation channel isn't a pure re-instatement in general. For tense 4r* (and, sub-threshold, indentation), OL moves where the model's *measured* behaviour doesn't.

## Questions
1. **Amending §10.1(b) before the freeze:** add the valid-denominator clause, fix the gain-ratio dose (0.108 relative norm), and replace the strict ±25% "acting" band. Random directions at 1.7–1.9× vs col at 1.0× suggest the band is unreachable as written. What should the acting criterion be? My proposal: "acting" if col's ratio is within the CI of 1.0 *and* OL shows no effect on behaviourally inert directions (indentation, tense 4r*). Is it acceptable to set this rule after seeing pilot data, given it's disclosed and the main run uses fresh items?
2. **P3 behavioural measure.** P3 (OL tense 4r*) tracks a direction whose effect on the model's *next-token* tense readout is flat. Prereg §4.4.5 would then force the wording "reader sensitivity to a behaviourally inactive direction". But next-token tense may be the wrong behavioural measure: tense shows at the next verb, which may be several tokens on. Should P3's behavioural validation use a generation-based measure, i.e. the tense of the next verb in a greedy continuation, fixed now? Or is "the reader tracks what the model doesn't act on" itself a finding to keep?
3. **The bigger picture.** Does "OL tracks directions the model's measured behaviour ignores (tense 4r*, indentation trend)" change the Branch 1b wording ("OL only through its next-token continuation channel")? It suggests OL's continuation channel carries more than the model's own next-token decision.
