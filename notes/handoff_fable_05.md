# Handoff #5 to Fable — 2026-10-06: final pre-freeze review

Paste everything below the line into the Fable 5.1 chat.

---

All prep workstreams are back. This is the last review before the pre-registration freezes on Oct 18 (sprint Oct 19 – Nov 1). Every pilot number below is from a **declared pilot split, excluded from the main analysis**. Please be blunt about what is overclaimed, what's missing, and what the main run should change.

## 1. Results since handoff #4

**(a) col pilot** (60 items; L42; centred steering; 4r* J-removed direction; q90):
- No reader tracks col beyond K=20 isotropic + J-complement null bands: NLA 0.50–0.53, OL 0.49–0.51, all |z| < 1.5, every template family.
- Fraction of probe: NLA ≤ 0.12, OL ≤ 0.03.
- **Even the raw col direction isn't tracked** by either decoder, though it moves the J-Lens newline score (AUROC up to 0.84).
- 0 of 5,760 free readouts mention line length.
- **Leak detector with newline tokens:** r* = 384 leaks newline (z +2.6 to +3.0); 4r* = 1536 is clean at L42/L40.
- **Prereg flaw caught:** H2's "CI excludes 0.5" test passes on noise. NLA numerals at r*: 0.54 [0.53, 0.57], but random directions move it as much. The null band must be part of every primary test.

**(b) K3 matched-dose calibration PASSES** (60 language ru/pl + 60 tense items; same pipeline; raw direction; q90):

| Reader | Language | Tense | Raw col, same dose |
|---|---|---|---|
| NLA | 0.83 (z +25/+31) | 0.83 (z +12/+18) | 0.48 |
| OL | 0.76 (z +11/+21) | 0.55 (out of band, below the 0.65 bar) | 0.48 |

- At **equal relative norm** (0.11 and 0.14): NLA language 0.87/0.92, tense 0.85/0.89; OL language 0.80/0.85; **col 0.47–0.48 for both**.
- In-J features were steered at slightly *smaller* relative norm (0.095) than col (0.108).
- Band SD is similar across features (0.012–0.029). In-J per-item d_z +1.2 to +1.5 vs col −0.2 to −0.3.
- **Secondary (pilot):** the NLA tracks the **J-removed (4r*) tense direction leak-free** (0.60, z +8.7; clean under label and content leak sets) while OL is in band. So the NLA's col failure is feature-specific, not a general inability to read outside the top-1536 J-space. The language 4r* result is void: Cyrillic content leaked, z +3.7, caught only after adding a content token set to the leak check.

**(c) Backup features: none non-J.**
- Capitalisation and in-quote: in-J.
- Coding language and words-since-sentence-start: diffuse-but-J-readable (steers leak).
- Indentation: non-J by G2 but **unused** (G4 fails; it's probably copied from the previous line's whitespace).
- French gender: fails G1.
- **The in-J count control (list index with numerals, "Item N:") failed:** digits never enter the J-Lens, and steering has no effect. Numbers don't enter Qwen's J-space here, consistent with col_report.
- **Pattern:** on Qwen L42, the J-depleted features are positional bookkeeping (col, indentation, sentence position, bracket depth). Every non-count feature is J-enriched or readable and keeps only 12–23% of its causal effect after J-removal; col keeps 79%.
- **col nuance:** with the full-distribution newline mass (not the top-10), col reads weakly in J-Lens (AUROC 0.58). That matches your prediction that the count is non-J and the decision it feeds is J.

**(d) Planted intermediates (C):**
- K4 passes: 431/614 kept at L38 with the centred swap; 109 at alpha ≤ 1.5.
- **Doses are large:** median ‖Δ‖/‖h‖ 0.49, about 1.5–3× the natural X–Y class-mean distance.
- All readers report the planted bridge: OL 1.00, NLA 0.98, J-Lens 1.00, vs 0.01–0.03 unplanted. That's in-J calibration (the swap direction is 6.5× enriched).
- Confound: by L42 the model already encodes the new answer.
- **The NLA rewrites the clue to fit the plant** (12/12 free-form: "Eiffel Tower" → "Volvo… Sweden"). Its "prompt reconstruction" is generated from the latent.

**(e) Re-solving decomposition on WorkspaceBench** (regex families, L42, n=400):
- A sample-matched text-only Qwen (no activation) scores 0.46 vs NLA 0.47 (p=0.75) and OL 0.40.
- **0/187 NLA-only hits** (OL 4/159, all Chinese code-switching).
- A text model given only the NLA's readout scores 0.57–0.60, above the NLA's own 0.47.
- Text-verifiable precision: NLA 0.62, OL 0.52, text-only 0.92. κ between two judges is 0.72–0.77 on 40 claims; the human-labelled subset is still TODO.
- **WorkspaceBench's frozen prompt-only floors are ~0.14 too low** under the thinking-off rendering. Confirmed, cause unknown. Our conclusions use our own text arms.

**(f) Templates frozen** (sha256 0915672…), mined by natural-item discriminability only.
- Natural col: NLA 0.60 [0.51, 0.69], OL 0.55, vs language 1.00, tense 0.85/0.92.
- 0/1,280 free readouts mention line length.
- Fixed-template sensitivity is large: OL tense runs 0.39–0.87 across templates.

## 2. Where I think we are
**Tentative answer to Neel: "No, for this case."** On the one natural feature that is non-J under two lenses and that the model causally uses (col), neither the NLA nor OL tracks it under text-fixed steering. Yet both track workspace features at the same or smaller steering norm. On WorkspaceBench, the readers' successes are fully explained by text. The NLA does read *some* leak-free non-top-J content (tense 4r*), so the claim is not "NLAs only read the workspace". It's that what the NLA reads beyond J-space isn't the model's non-verbalizable positional bookkeeping, at least not col.

## 3. Questions for you
1. **Headline.** Given (a)–(e), is the write-up now: Part A (what counts as non-J; the survival test is uninformative; non-J on Qwen = positional bookkeeping) + Part B (col: no decoder reads it, K3 passes at matched dose) + panel 2 (WorkspaceBench success = text)? Is the tense-4r* NLA result strong enough to mention outside an appendix, or will it read as a fishing result? It is pilot-only, and it came from an analysis I added mid-pilot.
2. **The count-verbalisation gap is open again** (no in-J count exists on Qwen). Is "qualitative templates + language/tense calibration + 'readers never mention line length'" enough? Or should Part B's claim be scoped to "decoders don't read col *in any phrasing they use*"?
3. **Main-run design changes I propose:**
   - (i) primary tests require z beyond both null bands (not CI vs 0.5);
   - (ii) content token sets in every leak check;
   - (iii) mined templates plus OL continuation-style col candidates (newline vs another word);
   - (iv) add a binary short-vs-long class-mean col direction as a robustness arm (the current direction is regression-based);
   - (v) pre-register NLA-vs-OL on tense 4r* as a secondary cell;
   - (vi) planting primary = the alpha ≤ 1.5 subset (109 items), with larger doses reported as sensitivity;
   - (vii) n: with col's effect ≈ 0 and the band the limiter, is 240 col items still the right size, or should compute go to more in-J/secondary cells?
4. **Risk of overclaiming "No".** It rests on one feature. What's the most honest one-sentence answer to Neel's question we can defend? Should the title change from "Do NLAs read what the model can't say — or re-solve what it read?"?
5. **The WorkspaceBench floor issue.** It's confirmed but the cause is unknown. Mention it in the write-up (with the re-run table), file an issue with the authors first, or both?
6. Anything in this pile you think is wrong.
