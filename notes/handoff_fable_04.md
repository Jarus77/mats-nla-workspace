# Handoff #4 to Fable — 2026-10-06

Paste everything below the line into the Fable 5.1 chat.

---

Feature gates are back under your final G2:
- **geometric:** diff-in-means enrichment ≤ 1.5 at r* and 4r*;
- **functional:** J-Lens value-token AUROC inside the shuffled-label null band;
- **causal use:** steering along the J-removed direction vs a random-direction band;
- leak pre-check at L38.

r* = 384 by the readout-agreement rule, so 4r* = 1536. J-Lens = neuronpedia n1000; the camilablank lens was fit on only 25 prompts and differs a lot (cosine 0.76). **Only one clean non-J feature qualified (target was 2–3).** The calibrated test also produced a methodological finding. I need strategy calls.

## Gate table (L42)
| Feature | Probe | Enrichment r*/4r* | J-Lens AUROC [null band] | J-removed steering | Label |
|---|---|---|---|---|---|
| **col**: characters since last newline (line wrap) | R² 0.655 (control task 0.12) | 0.67 / 1.10 | 0.500 [0.499, 0.501] | +0.99 newline log-odds, **49× random**; keeps 79% of the full-direction effect | **NON-J** |
| col_report: same count under an explicit "how many characters" question | R² 0.88 | 0.59 / 0.86 | 0.500 | 3.5× | non-J, **so the planned in-J calibration failed** |
| bracket depth | R² 0.945 | 0.57 / 1.16 | 0.5035 (band top 0.501) | 1.5–1.9× | diffuse-but-J-readable (borderline) |
| predicted next-word POS | acc 0.823 (chance 0.125) | 2.62 / 1.56 | 0.481 | 11× | J-enriched, fails |
| tense | acc 0.845 | 5.6 / 2.3 | 0.52 | 7× (19% of the full effect) | in-J |
| language (French/German) | acc 0.966 | 3.3 / 1.7 | 0.62 | ≈ 0 | in-J |
| topic (reference) | acc 0.849 | 6.0 / 2.3 | 0.83 | — | in-J |

**Other results:**
- **Leak pre-check:** steering J-removed directions at L38 moved no feature label into the J-Lens top-10 at 40/42/44, at q99 or at 2×q99.
- **Self-report** (reported only): col is poorly reportable (free-answer Spearman 0.42, MAE no better than predicting the mean); tense and topic are reportable (87%, 82%).
- **Qwen can't count without chain of thought** (7% exact).
- **Explicit report didn't put the count into J.** Number tokens are near zero in J-Lens on digit-free passages, unlike the J-space paper's line-width result.

## The methodological finding
- The standard "probe survives J-removal" test is uninformative: every feature keeps ≥ 98% probe performance after removing r* or 4r*, *topic and language included*. Retention minus random-equal-rank removal is within ±0.02 for every feature.
- The J-space paper's POS/tense evidence is a diff-in-means probe with its own top-25 J atoms projected out, measured in z-scores, with no random-subspace control. On Qwen the same test also "passes" topic (AUROC 0.90 → 0.88). Under the calibrated gate, POS and tense are J-enriched.

## Questions
1. **Is one non-J feature enough for the headline?** col is a strong ground truth: causally used (49×), not in J by either criterion, not self-reportable, no leak. My proposal: a deep case study on col (dose-response, three readers, norm-preserving steering, null bands, templates frozen) plus backups if any pass. The backup agent is now gating count-like features (indentation, words since sentence start, list index) plus capitalisation and coding language. Is "one feature, done very carefully, plus a replication on any backup that passes" a credible headline for Neel? Or should the methodological finding become the co-headline?
2. **The calibration gap.** The in-J control for col was meant to be a count under explicit report, to rule out "readers just can't talk about counts". It came out non-J. So if the NLA and Oracle Lens both fail to track col, I can't distinguish "can't read non-J" from "can't verbalize counts". Options:
   - (a) find an in-J count feature, e.g. counts where number tokens *are* in J-Lens (digits present in context, "the 7th item");
   - (b) lean on language and tense as generic in-J calibration and state the count caveat;
   - (c) treat col's natural-arm scoring (readers vs TextNLA) as the count-verbalization check: if readers name approximate line length on natural items, they can talk about counts.

   Which, or something better?
3. **Framing of the methodological result.** Is "the standard non-J test is uninformative, and calibrating it against random subspaces flips POS and tense to J-enriched" worth a section? How do I write it without overclaiming against the J-space paper? Their model is Claude; ours is Qwen3.6-27B with a different lens.
4. **pred_pos is odd:** J-enriched (2.62 at r*) yet J-Lens doesn't read it (0.481, slightly below the band). How should I describe a feature that is geometrically in the J-space but invisible to the lens? Is that a third category, or a lens failure?
5. **Pivot logic.** The prereg says 1 non-J feature means "run with 1 plus a backup"; 0 fires P1. With col alone, do you see a scenario where we should pivot anyway?

## Other status
- **Running in parallel:**
  - the norm-preserving toy re-run (your anti-tracking test);
  - backup feature gates;
  - (C) planted-intermediates yield;
  - the re-solving decomposition on WorkspaceBench (TextNLA, claim-level text-verifiable precision with a second judge);
  - native template mining for each reader, frozen before steering.
- **A bug fixed:** Qwen's final norm uses (1 + weight); an early script used the weight alone. All gated numbers come from corrected reruns.
