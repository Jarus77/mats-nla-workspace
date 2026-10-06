// Deck-wide metadata and the running "what we know so far" ledger.
// To update a status: change `status` ("established" | "pilot" | "tentative" | "pending") or the text.
// "pilot" = measured on the declared pilot split (excluded from the main analysis).
// `stage` = the first anchor slide on which the item appears (1..4).

module.exports = {
  meta: {
    title: "Do NLAs read what the model can't say?",
    subtitle: "Testing whether Natural Language Autoencoders capture non-workspace content in Qwen3.6-27B",
    author: "Suraj Prasad",
    affiliation: "MATS 12.0 · Neel Nanda stream",
    date: "Prep-phase update · 6 October 2026",
    file: "mats_nla_workspace",
  },

  knowledge: [
    { stage: 1, status: "established", text: "The instruments are sound: NLA layer convention pinned (block 42), FVE 0.761 reproduced, WorkspaceBench 12/12 replicated." },
    { stage: 1, status: "established", text: "At the same layer (42) the NLA ties or beats the Oracle Lens; OL's published lead comes from a best-of-11-layers protocol." },
    { stage: 2, status: "established", text: "The standard 'probe survives J-removal' test cannot tell in-J from non-J: topic and language pass it too." },
    { stage: 2, status: "established", text: "Under a calibrated gate only col (characters since newline) is cleanly non-J; POS, tense, capitalisation, quotes, code language are J-enriched or J-readable." },
    { stage: 2, status: "established", text: "Survey of ~13 features: non-J content the model uses on Qwen L42 looks like positional bookkeeping; numbers never enter the J-space." },
    { stage: 3, status: "pilot", text: "No robust descriptive reading of col: fixed templates stay inside the null band (NLA ≤ 0.53, OL ≤ 0.51), and no readout mentions line length." },
    { stage: 3, status: "pilot", text: "At the same dose the NLA tracks language and tense (0.83–0.92) and OL tracks language; col stays at 0.47–0.48. NLA's null is secure; OL's less so (tense 0.55, template-fragile)." },
    { stage: 3, status: "pilot", text: "Two candidate narrow channels, pre-registered for confirmation: OL's next-token continuation (J-removed 0.67–0.72, z ≈ +3.9; needs the adapter) and one NLA template (q90 0.63, borderline)." },
    { stage: 4, status: "pilot", text: "Beyond the top-1,536 J-space both decoders track tense's tail; with mined templates OL ≥ NLA (0.66–0.79 vs 0.59–0.74). A primary in the main run." },
    { stage: 4, status: "pilot", text: "Exp 09: OL's col channel moves one-for-one with the model's newline decision (gain 0.99 / 1.06) but also tracks tense the model doesn't act on: not a pure re-emission." },
    { stage: 4, status: "established", text: "The NLA generates its 'prompt' from the latent: under a planted bridge it rewrites the clue 12/12 times (Eiffel Tower → Volvo… Sweden)." },
    { stage: 4, status: "established", text: "On WorkspaceBench the readouts' target information is recoverable from the prompt: re-solver 0.89 from prompt vs 0.60 / 0.56 from readouts; prompt + readout 0.88." },
  ],
};
