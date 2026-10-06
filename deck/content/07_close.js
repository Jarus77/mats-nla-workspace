// Where we stand: ledger, answer, open checks, limitations, sprint plan, close.

module.exports = [
  {
    type: "anchor",
    section: "Where we stand",
    stage: 4,
    title: "What we know so far (4 / 4): the full ledger",
    notes:
      "The full ledger. Dark chips are established on full data, ochre chips are pilot-scale (60 items per feature, excluded from the main analysis), grey are tentative. " +
      "Read top to bottom: sound instruments; a gate that leaves one clean non-workspace feature, which is positional; decoders whose descriptions read workspace features but not col, with two candidate narrow channels; decoders that read the tail of verbalisable features, and an NLA that generates its story from the latent; and a benchmark whose targets are recoverable from the prompt alone.",
  },
  {
    type: "branches",
    section: "Where we stand",
    kicker: "Our answer to the title question, on both branches",
    title: "Do NLAs read what the model can't say? The main run decides between two pre-registered answers",
    branches: [
      { head: "If the candidate channels replicate", answer: "Barely, in narrow channels, and not in the language they use to describe what they read.", body: "Say 'channel-specific sensitivity', never 'reads'. OL's channel likely re-instates the activation for Qwen's own newline circuitry." },
      { head: "If they don't", answer: "No, on the one natural non-J feature the model uses.", body: "Both decoders read workspace features at the same dose, so the null is interpretable (with the template caveat)." },
    ],
    stands: [
      "Part A: the standard non-J test is uninformative; non-J content on Qwen L42 looks like positional bookkeeping (a ~12-feature survey).",
      "Part D: WorkspaceBench readouts' targets are recoverable from the prompt (0.89 vs 0.60 / 0.56).",
      "Part C: the NLA generates its 'prompt' from the latent (12/12 clue rewrites).",
    ],
    caveats: "Caveats on both: n = 1 clean non-J feature · pilot numbers with many looks · OL is template-fragile · count-verbalisation gap open.",
    notes:
      "So, the title question: do NLAs read what the model can't say? The honest answer today is a pre-registered fork. " +
      "If the two candidate channels replicate on fresh items after Holm correction, the answer is: barely, in narrow channels, and not in the language they use to describe what they read. We'd say 'channel-specific sensitivity', never 'reads'; and for the Oracle Lens the likely mechanism is re-instatement, the LoRA handing the activation back to Qwen's own newline circuitry. Oracle Lens is the model plus a LoRA: it reads what the model reads and says what the model can say. " +
      "If they don't replicate, the answer is: no, on the one natural non-J feature the model uses; and that null is interpretable because both decoders read workspace features at the same dose. " +
      "Three results stand on either branch: Part A's methodological finding and survey, Part D's benchmark decomposition, and the finding that the NLA generates its prompt reconstruction from the latent. " +
      "Caveats on both branches: one clean non-J feature; pilot numbers with many looks; the Oracle Lens's template fragility; and the open count-verbalisation gap.",
  },
  {
    type: "checks",
    section: "Where we stand",
    kicker: "Open checks before the pre-registration freezes",
    title: "Pre-freeze checks: three done, three still open",
    checks: [
      { tag: "e", head: "Re-solver consistency: DONE", body: "An instruction mismatch (D22). Same template: prompt 0.89 vs readout 0.60 / 0.56; prompt + readout 0.88.", done: true },
      { tag: "q", head: "q90 behavioural check: DONE", body: "J-removed col push: newline log-odds +0.72 (z +55), wrap −1.55 chars, J-Lens unmoved (D24).", done: true },
      { tag: "t", head: "Natural col vs text: DONE", body: "Text-only Gemini 0.97; the NLA's 0.60 falls to 0.51 after regressing out text (D23).", done: true },
      { tag: "m", head: "OL mechanism tests: RUNNING", body: "Indentation in OL's carrier; gain ratio (OL shift ÷ model shift) for col, tense, random directions; tense with the adapter off." },
      { tag: "d", head: "Class-mean col direction: OPEN", body: "A binary short-vs-long-line direction as robustness next to the regression-covariance direction." },
    ],
    changes: "Confirmatory design (D25): 4 primaries (NLA / OL col at q90 4r*, NLA / OL tense 4r*), Holm, |z| ≥ 3 against both bands + secondary family agrees in sign · fresh items · templates frozen literally · class-mean col direction as robustness.",
    notes:
      "Where the pre-freeze checks stand. Done: the re-solver consistency check, the behavioural validation at q90, and the natural-col text baseline.  Running: three tests of the Oracle Lens mechanism. If steering indentation, which the model does not act on, still shifts OL's continuation, OL reads beyond what the model acts on; if the ratio of OL's shift to the model's behavioural shift is constant across col, tense and random directions, that points to re-instatement; and the adapter-off check for tense. Still open too: a class-mean col direction as robustness. The bottom line is the confirmatory design.",
  },
  {
    type: "grid",
    section: "Where we stand",
    kicker: "Limitations",
    title: "Threats to validity we are tracking, and what addresses each",
    items: [
      { head: "One non-J feature", body: "No second clean feature in the ~12-feature survey; non-J content looks positional. Stated as a scope limit." },
      { head: "Count-verbalisation gap", body: "Numbers never enter Qwen's J-space (col_report, list index), so there is no in-J count control. The gap stays open; stated as a limitation." },
      { head: "Forking paths", body: "Many looks in the pilot; primaries chosen from it. Fixed by 4 named primaries, Holm, fresh items, templates frozen literally." },
      { head: "Templates and pilot scale", body: "OL tense 0.39–0.87 across templates; pilot numbers are 60 items per feature on a declared, excluded split." },
      { head: "Direction choice", body: "col probe R² 0.655 (a curved manifold); covariance vs class-mean direction. Adds the binary direction." },
      { head: "J-space definition", body: "No spectral elbow; two lenses differ (cosine 0.79). Claims say 'outside the top-r J-space', at r* and 4r*, under both lenses." },
      { head: "Judges", body: "Claim precision is judge-based (κ 0.72–0.77 judge vs judge). 300 human-labelled claims still to do." },
      { head: "Scope", body: "One model, one layer (42). OL is off-grid at 42 but no worse than at 40/44. Planted doses are large (0.49 of ‖h‖)." },
    ],
    notes:
      "Threats to validity, each with what addresses it. The big two are the first two. One feature: the survey explains why there's no second one, so it's a scope limit, not something the sprint can fix. And the count-verbalisation gap: numbers simply don't enter Qwen's J-space, so we have no in-workspace count control, and can't fully rule out that readers fail on col because they can't talk about counts. " +
      "The rest are standard: template sensitivity, pilot scale, direction choice, the J-space definition, judge validity, and scope.",
  },
  {
    type: "gantt",
    section: "Where we stand",
    kicker: "Sprint plan · 19 October – 1 November",
    title: "The sprint confirms four pre-registered primaries on fresh items, with kill checks on days 2, 3 and 7",
    days: ["19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "1"],
    rows: [
      { label: "Part B: col primaries, NLA template + OL continuation (q90, 4r*), fresh items", from: 1, to: 1, color: "nla" },
      { label: "Upstream (layer 38) arm, q90 behavioural check, leak test, FVE guard", from: 2, to: 2, color: "nla", check: "K2" },
      { label: "Positive controls at matched dose: language, tense", from: 3, to: 3, color: "nla", check: "K3" },
      { label: "Tense 4r* primaries (NLA, OL); OL mechanism tests; template-free judge", from: 4, to: 4, color: "text" },
      { label: "First figures; one-page interim for Fable / Neel", from: 5, to: 5, color: "muted" },
      { label: "Planted intermediates (α ≤ 1.5 primary); free-form rewrites", from: 6, to: 7, color: "jl", check: "K4" },
      { label: "Free-form generations, judges, human-label κ", from: 8, to: 8, color: "text" },
      { label: "Robustness: rank sweep, both lenses, OL@40/44, NLA iter 300, templates", from: 9, to: 10, color: "muted" },
      { label: "Optional: 20-item split-injection replication", from: 11, to: 11, color: "line" },
      { label: "Final figures, write-up, red-team with Fable", from: 12, to: 14, color: "ol" },
    ],
    kills: [
      { id: "K2", text: "doses valid: FVE within margin, no leak" },
      { id: "K3", text: "readers track ≥1 in-J feature at col's dose (passed in pilot)" },
      { id: "K4", text: "≥60 valid planted items (431 in prep; 109 at α ≤ 1.5)" },
    ],
    notes:
      "The sprint is 19 October to 1 November, framed as a confirmatory test of four pre-registered primaries on fresh items: the NLA's col template and the Oracle Lens's continuation channel at q90 on the J-removed direction, and J-removed tense for each decoder, with Holm correction. Day 1 the col primaries. Day 2 the upstream arm with the behavioural check and leak test: K2. Day 3 the language and tense positive controls at matched dose: K3. Day 4 the tense primaries, the Oracle Lens mechanism tests and the template-free judge check. Then an interim write-up, planted intermediates (K4), judges and human labels, robustness, an optional split-injection replication, and three days for figures, write-up and red-teaming." +
      "Pivot rule: we pivot only if K3 fails, or if col and every in-J control are flat for every reader.",
  },
  {
    type: "closing",
    section: "Where we stand",
    title: "Contributions so far, and what would change our minds",
    contributions: [
      "A calibrated, random-subspace-controlled gate for 'non-workspace' content (the standard survival test is uninformative), and a survey: on Qwen L42 the non-J content the model uses is positional.",
      "A text-fixed causal pipeline (centred steering, null bands, two-sided leak detector, matched-dose controls): decoders' descriptions read workspace features but not col; two narrow channels are candidates.",
      "Both decoders read the tail of verbalisable features (tense); the NLA generates rather than re-solves, rewriting the prompt to fit the latent.",
      "WorkspaceBench readouts' targets are recoverable from the prompt (0.89 vs 0.60 / 0.56); OL's lead is a best-of-11 effect.",
    ],
    open: [
      "The candidate channels surviving (or failing) Holm at |z| ≥ 3 on fresh items.",
      "A second non-J feature that is not a count, on another model or layer.",
      "Readouts carrying target information beyond the prompt on richer, non-regex benchmarks.",
    ],
    notes:
      "To close, the contributions so far. One: a calibrated gate for non-workspace content, and a survey showing the non-J content Qwen uses at layer 42 is positional. Two: a causal pipeline that separates reading from re-solving; on col the decoders' descriptions don't follow it, and two narrow channels are candidates. Three: evidence that the NLA generates a story from the latent rather than re-solving the text. Four: a benchmark decomposition and a protocol re-analysis. " +
      "What would change our minds: the candidate channels surviving or failing the pre-registered test on fresh items; a non-count non-J feature; or readouts beating the prompt on benchmarks beyond these four regex families. Questions welcome.",
  },
];
