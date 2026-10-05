// Deck-wide metadata and the running "what we know so far" ledger.
// To update a status: change `status` ("established" | "pilot" | "tentative" | "pending") or the text.
// "pilot" = measured on the declared pilot split (excluded from the main analysis).
// `stage` = the first anchor slide on which the item appears (1..4).

module.exports = {
  meta: {
    title: "Do NLAs read what the model can't say — or re-solve what it read?",
    subtitle: "Testing whether Natural Language Autoencoders capture non-workspace content in Qwen3.6-27B",
    author: "Suraj Prasad",
    affiliation: "MATS 12.0 · Neel Nanda stream",
    date: "Prep-phase update · 6 October 2026",
    file: "mats_nla_workspace",
    titleNote: "WORKING TITLE · PROVISIONAL UNTIL THE PRE-REGISTRATION FREEZES",
  },

  knowledge: [
    { stage: 1, status: "established", text: "The instruments are sound: NLA layer convention pinned (block 42), FVE 0.761 reproduced, WorkspaceBench 12/12 replicated." },
    { stage: 1, status: "established", text: "At the same layer (42) the NLA ties or beats the Oracle Lens; OL's published lead comes from a best-of-11-layers protocol." },
    { stage: 2, status: "established", text: "The standard 'probe survives J-removal' test cannot tell in-J from non-J: topic and language pass it too." },
    { stage: 2, status: "established", text: "Under a calibrated gate only col (characters since newline) is cleanly non-J; POS, tense, capitalisation, quotes, code language are J-enriched or J-readable." },
    { stage: 2, status: "established", text: "Survey of ~12 features: non-J content the model uses on Qwen L42 looks like positional bookkeeping; numbers never enter the J-space." },
    { stage: 3, status: "pilot", text: "Neither decoder tracks col beyond random directions at leak-free doses (NLA ≤ 0.53, OL ≤ 0.51 AUROC)." },
    { stage: 3, status: "pilot", text: "At the same dose the NLA tracks language and tense (0.83–0.92) and OL tracks language; col stays at 0.47–0.48. NLA's null is secure; OL's less so (tense 0.55, template-fragile)." },
    { stage: 3, status: "pilot", text: "The raw col direction moves the J-Lens newline readout (AUROC up to 0.84) yet no decoder tracks it: content type, not location." },
    { stage: 4, status: "pilot", text: "Beyond the top-1,536 J-space the NLA reads the tail of a verbalisable feature (tense 0.60, z +8.7) where OL doesn't; pre-registered secondary." },
    { stage: 4, status: "established", text: "The NLA generates its 'prompt' from the latent: under a planted bridge it rewrites the clue 12/12 times (Eiffel Tower → Volvo… Sweden)." },
    { stage: 4, status: "tentative", text: "WorkspaceBench successes are matched by a text-only model (0.46 vs 0.47; 0/187 reader-only), pending a re-solver consistency check." },
  ],
};
