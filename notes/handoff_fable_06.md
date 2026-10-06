# Handoff #6 to Fable — 2026-10-06: the col null partly overturned (pilot)

Paste everything below the line into the Fable 5.1 chat.

---

Your round-5 cells came back. Two results overturn the sentence we were about to freeze ("neither decoder reads col in any phrasing"). I've retracted it. Everything below is from the declared pilot split, excluded from the main analysis. Please be blunt, especially about forking paths.

## What came back
1. **Behavioural validation passes.** At L42 q90, the newline log-odds shift is +1.44 raw (z +86) and **+0.72 for the J-removed 4r* direction (z +55)**. Wrap-column shift in generation: −6.8 chars raw, −1.55 for 4r*. The 4r* edit changes behaviour, yet J-Lens can't see it (EOL 0.50; content sets clean, |z| ≤ 2.1).
2. **Two side checks resolved:**
   - **Natural col vs TextNLA:** text-only Gemini reads natural col at 0.97. Once that's regressed out, the NLA's natural 0.60 drops to 0.51, so natural col says nothing about activation reading.
   - **The (e) inconsistency was an instruction mismatch.** Same template, k=1: prompt-only 0.89 vs readout-only 0.60/0.56, and prompt + readout 0.88. The residual readout-only hits (4/2 of 400) are all regex artefacts. So Part D holds as "target information in readouts is recoverable from the prompt alone".
3. **The NLA tracks col with one mined template family (f07_count, frozen before any steering):**
   - raw 0.71 (z +5.0);
   - **4r* r11: 0.67 (z +3.7/+3.3)**;
   - 4r* q90: 0.63 (conservative z +3.0/+2.6, borderline).

   It rests mostly on **one template** ("near the end of a hard-wrapped line"). The fixed pilot templates and the "break coming" family are flat.
4. **OL's newline-continuation channel tracks col** (candidates `"- \nthe"` vs `"- the"`, fixed at 00:53, scored at 03:55):
   - raw 0.76 (z +6.6);
   - 4r* 0.67–0.72 (conservative z +3.9).

   The shift is linear in dose. J-removal keeps ~78% per unit dose, close to the model's own behavioural ~70%. **Adapter OFF** (base Qwen in OL's carrier prompt) drops it to 0.40–0.44, so it needs the LoRA. Every *descriptive* OL col template is flat or anti-tracking. My wording: "OL's continuation channel carries the newline decision", not "OL reports line position".
5. **Tense 4r* with mined templates:** OL 0.66–0.79 (z +12 to +17) ≥ NLA 0.59–0.74. So **"NLA > OL on tense 4r*" does not survive**. OL's tense candidates are continuations too; adapter-off for tense hasn't been run.
6. **The next-digit cell is unusable.** It's readable (J-Lens 0.99) but causally inert: 0 flips, and enrichment is no higher than shuffled labels. It probably encodes list length. The count-calibration gap stays open.

## My read and self-critique
- **D21's "content type, not location" story was built on fixed templates.** With native or mined phrasings, both decoders show **leak-free sensitivity to the J-removed col direction**: the NLA through one descriptive template, OL through its next-token continuation channel. OL's descriptive language stays flat.
- **Forking paths are now the main threat.** Across rounds the pilot tried many templates × directions × doses × readers. The positives came from *pre-specified* families, but the number of looks is large. An uncorrected z of +3.7 among ~100 cells is not strong.
- **The emerging claim** is "sensitivity to non-J content exists but is channel- and phrasing-specific: mostly not expressed in the decoders' descriptive language". That's more interesting than a clean null, and it bears directly on whether these tools can *report* non-workspace content. But it's exactly the kind of claim that dies on replication.

## Questions
1. **Is the retraction right?** Or does a single-template NLA effect at z +3.7, with q90 borderline, still leave "no robust reading" as the honest pilot summary?
2. **Is OL's continuation-channel effect "reading"?** It needs the LoRA (adapter-off kills it), but it tracks the model's own behavioural newline effect almost proportionally (78% vs 70%). Is that the decoder *reporting* the decision, or the LoRA *re-instating* the activation so Qwen's own next-token circuitry *acts on* it? Is there a cheap test that separates these? My idea: put the same vector into OL's carrier prompt but score a non-newline continuation that the model's behaviour wouldn't predict.
3. **Main-run design against forking paths:**
   - (a) one pre-registered primary template per reader per feature: NLA = the "hard-wrapped line" template, OL = nl_cont;
   - (b) Holm correction across a declared family of primary cells;
   - (c) replicate on fresh items;
   - (d) everything else exploratory.

   Is that enough? Should the primary for each reader instead be the *mean over the frozen template set*, so the template choice isn't post hoc?
4. **Headline.** If (3) replicates, the write-up becomes "decoders carry non-workspace content in narrow channels: one NLA phrasing, OL's continuation channel, but not in how they describe what they read". If it doesn't, the D21 story returns, with these results as a cautionary appendix. Is that the right fork, and is the title "Do NLAs read what the model can't say?" still apt?
5. Is there anything you'd now cut or add before the Oct 18 freeze?
