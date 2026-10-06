# Handoff #7 to Fable — 2026-10-06: review of the confirmatory pre-registration (rev 4) before the freeze

Paste everything below the line into the Fable 5.1 chat.

---

All of your round-6 points are implemented in pre-registration revision 4, which freezes before Oct 18. This is a short review request: two open decisions, and three things I'd like you to sanity-check.

## What rev 4 says
- **Confirmatory order:**
  1. P1–P4 (Holm over 4);
  2. positive controls PC1/PC2 (raw language and tense at matched relative norm);
  3. pre-registered negatives N1–N5 (TOST, SESOI AUROC 0.55 = fraction of probe 0.10);
  4. template-free judge check T1 ("Does this text mention position within a line, line length, wrapping, or an upcoming line break? yes/no", Gemini 3.8 Flash then kimi-k3, 6,720 free readouts on fresh items, a 40-item judge-sensitivity set first);
  5. secondaries S1–S9;
  6. everything else exploratory.
- **Primaries.** All four use L42, centred steering, v = normalise((I − P_S1536)w), q90 dose, and per-item effect (LO(+Δ) − LO(−Δ))/2.
  - **P1, NLA col:** the single template "Mid-sentence continuation located near the end of a hard-wrapped line." vs "…at the beginning of a hard-wrapped line." Pilot 0.63, z +5.6 iso / +3.5 J-complement.
  - **P2, OL col:** the nl_cont pairs `"- \nthe"` / `"- the"` etc., mean of 3. Pilot 0.67, z ≈ +3.9.
  - **P3, OL tense 4r*:** f07_tense, mean of 3. Pilot z +12.5/+16.6.
  - **P4, NLA tense 4r*:** f07_tense, mean of 3. Pilot z +7.5/+9.2.
- **Claim criterion:**
  - z ≥ +3 vs BOTH bands, and the shift beats every null direction;
  - survives Holm;
  - the declared-secondary family mean agrees in sign;
  - the cell isn't void (leak or FVE).
  - **K = 100 null directions per band for the primaries** (with K=20 the band SD has ~16% sampling error).
- **Disclosure** (§1.4): ~514 centred template-level cells looked at in the pilot (~2,000 z-values). P1 was the best of 11 NLA col templates.
- **Power:** the band, not n, limits it. P1's family mean in the pilot is only z +2.5–2.9, so **winner's-curse shrinkage may fail P1's sign-agreement**, and the prereg says so. n: 240 fresh col passages, 160 tense, 120 language.
- **Headline keyed to P1/P2 only**, two branches with fixed wording; Parts A, D and the latent-generated reconstruction stand on either branch. Retractions and the cut list are applied. Analysis script frozen with its hash recorded; deviations log at `notes/deviations.md`.

## Decisions for you
1. **Leak rule for the primary col sets:**
   - (a) void a dose at the two-sided 95% J-complement band (as written; conservative; wrongly voids a valid cell ~14% of the time); or
   - (b) void only at |z| ≥ 3 for all sets.

   I lean towards (a) for the primaries, since a false void costs us a claim, never a false claim.
2. **The OL mechanism tests** (indentation steering in OL's carrier; the gain ratio across col, tense and two random directions; tense adapter-off) are running on the pilot split. Must they finish before the freeze? Default: fold them in if done by Oct 15. They're written into the prereg as secondary S6 with fixed predictions either way.

## Sanity checks
3. **Is "P1 = best of 11 templates" defensible as a primary,** given the disclosure, with the family mean as a declared secondary that must agree in sign? Or should P1 be the family mean, accepting the dilution you warned about?
4. **Is SESOI AUROC 0.55 the right equivalence bound** for the negatives (OL descriptive col templates flat)? With the null-band SD ≈ 0.012–0.03, 0.55 is about 2–4 null SDs.
5. **Anything that would make the confirmatory run uninformative** even if everything executes as planned?
