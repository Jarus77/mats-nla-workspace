# Handoff #3 to Fable — 2026-10-05 (night)

Paste everything below the line into the Fable 5.1 chat.

---

The pipeline is built and an end-to-end smoke test ran on a toy in-J feature: language identity, French vs German, 40 pairs, steering at the read layer. It cost $1.54. The plumbing works:
- capture matches the NLA's convention;
- the J-basis is consistent across code paths;
- equal-norm random-direction nulls give 0.47–0.54 for all readers;
- the FVE guard passes at q90/q99 (−0.015) and voids 2×q99 (−0.06);
- the L38 arm has next-token KL ≤ 0.008, and 54–86% of the dose reaches 40–44.

Projected sprint compute is ~35 GPU-h. The smoke test also exposed four design problems. I need your critique of the fixes, especially #3, which is the one that could silently invalidate the headline.

## 1. FollowFeature is capped by each item's prior
Cross-item AUROC of the steered log-odds tops out at 0.79–0.91 even though every reader moves on 100% of items under raw steering. Each item's baseline log-odds varies more than the steering shift does.

**Fix:** the primary metric is the *paired* / zero-dose-subtracted version (per-item shift, AUROC or paired win rate), with the cross-item version alongside. Zero-dose subtraction is exposed to norm dilution: J-Lens uses a cosine readout and OL injects `16000·unit(h)`. So I also subtract the matching equal-norm random-direction contrast. The ± contrast is immune to dilution by symmetry. OK?

## 2. The leak detector needs to be two-sided
Under J-removed steering, the J-Lens *anti*-tracks the feature (AUROC 0.35–0.47). The rule "void if > 0.55" never fires on that.

**Fix:** void if |FollowFeature(J-Lens) − 0.5| > 0.05.

## 3. (Critical) "Survives J-removal" cannot distinguish in-J from non-J
- Language identity *is* in J-space: its probe direction is 3–7× enriched relative to random.
- Yet a probe keeps AUROC ≥ 0.998 after removing r* and 4r*. A probe retrained on J-removed vectors does too.
- Removing an equal-rank *random* subspace also leaves it at 1.00.

A strong binary feature is redundantly decodable from almost any large subspace of a 5120-d residual. So the old G2 would have labelled redundant in-J features as non-J. That could include POS and tense, the very features the J-space paper reports as "surviving J-removal".

**Proposed G2, symmetric and calibrated against random subspaces:**
- **(a) J-only probe:** trained on the projection *onto* the top-r J-space, compared with probes on K=10 random equal-rank subspaces.
- **(b) Enrichment** of the probe direction and the top-k INLP directions (k ≤ 8).
- **Non-J** ⇔ the J-only probe is inside the random-subspace 95% band AND enrichment ≤ 1.5, at both r* and 4r*.
- **In-J** ⇔ the J-only probe is above the band AND enrichment ≥ 2.
- The old survival test is reported *relative* to random-subspace removal.

**Questions:**
- Is "information not available within the top-r J-space beyond what a random subspace of the same size gives" the right operational meaning of non-J?
- Does it reintroduce the redundancy problem in reverse? A redundant feature might be decodable from *every* subspace, including the J-only one, and so look "in-J" by (a) while enrichment is ~1. Should enrichment then be the tiebreaker?
- Does this change how we should read the J-space paper's POS/tense claims?

## 4. Oracle Lens candidate templates must be mined
OL's generations at L42 are content continuations in the source language, not label bullets (e.g. "- Le chat dort paisiblement dans son panier…"). So "- French language" is off its output distribution. It still tracks, but probably insensitively.

**Fix:** mine OL's native phrasings per feature from its own generations on natural items, and score with 3 templates per reader, averaged. The NLA's explanations open with the label ("French descriptive sentence…"), so its templates are on-distribution.

**Question:** does template mining risk tuning the scoring to favour one reader? I'd mine on a held-out split and freeze the templates before the steering runs.

## Toy observation (not a result; n=40, plumbing only)
On the J-removed language direction:
- the NLA tracks it (zero-subtracted 0.96, paired win 0.91, fraction of probe 0.19);
- OL anti-tracks (0.06–0.15);
- the J-Lens anti-tracks (0.00).

It's exactly the H3-type contrast. It can't be dilution in a ± contrast. Do you have a mechanistic guess for why OL would *anti*-track the non-J part of a feature it tracks in full (raw direction: OL 0.84)? My guess: the non-J remainder of the probe direction is anti-aligned with whatever OL uses. If so, "OL tracks the J-removed direction" is a poor test of whether OL reads non-J content, and we may need the INLP-subspace version.

## Status
- Feature search (with the revised G2) is still running; results expected tomorrow.
- Both prep gates have passed (K0 and the WorkspaceBench replication).
- All runs so far: under $15 total.
