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
      { tag: "m", head: "OL mechanism tests: RUNNING", body: "Experiment 09, pilot split: indentation in OL's carrier; gain ratio (OL shift ÷ model shift); tense with the adapter off. Results won't change the predictions." },
      { tag: "d", head: "Class-mean col direction: OPEN", body: "A binary short-vs-long-line direction as robustness next to the regression-covariance direction." },
    ],
    changes: "Confirmatory design (prereg rev 4): P1–P4 · K = 100 nulls per band · Holm over 4 · sign agreement · negatives N1–N5 (TOST) · T1 judge check · fresh items (240 col / 160 tense / 120 language) · freeze Oct 18 with a hash manifest.",
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
    title: "The sprint runs the frozen analysis once: primaries in the first three days, everything else after",
    days: ["19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "1"],
    rows: [
      { label: "Integrity check; capture fresh banks; leak, FVE and behaviour at P1–P4", from: 1, to: 1, color: "nla", check: "K5" },
      { label: "Score P1–P4 (real + 2 × 100 nulls) and PC1–PC2; upstream behaviour", from: 2, to: 2, color: "nla", check: "K2" },
      { label: "One run of the frozen analysis: primaries, controls, N1–N5; record the branch", from: 3, to: 3, color: "nla", check: "K3" },
      { label: "T1: 6,720 free readouts, two judges, blind user labels", from: 4, to: 4, color: "text" },
      { label: "Declared-secondary families → sign agreement; S1–S5", from: 5, to: 5, color: "muted" },
      { label: "OL mechanism tests on fresh items (S6)", from: 6, to: 6, color: "ol" },
      { label: "One-page interim to Fable / Neel", from: 7, to: 7, color: "muted" },
      { label: "Planted bridge calibration; reconstruction-rewrite rate (S8)", from: 8, to: 8, color: "jl" },
      { label: "Part A replications on held-out items, both lenses", from: 9, to: 9, color: "jl" },
      { label: "Exploratory cells (labelled)", from: 10, to: 10, color: "line" },
      { label: "Figures, statistics audit, deviations log, write-up, red-team", from: 11, to: 14, color: "ol" },
    ],
    kills: [
      { id: "K5", text: "integrity: hashes match, 0 item overlaps, K = 100 nulls exist" },
      { id: "K2", text: "a primary failing leak or FVE is void (Holm stays over 4)" },
      { id: "K3", text: "PC1 and PC2 both fail → pivot; one fails → proceed" },
    ],
    notes:
      "The sprint, 19 October to 1 November, after a freeze on 18 October with a manifest of hashes for the prereg, templates, item banks, analysis script, null seeds and dose constants. " +
      "Day 1 is the integrity check, K5: hashes match, zero overlap with pilot items, 100 null directions per band exist; then capture and the leak, FVE and behaviour checks at each primary cell, K2. Day 2 scores the four primaries against 200 null directions each and the positive controls. Day 3 runs the frozen analysis script exactly once and records which branch we're on; K3 checks the positive controls. Only then the template-free judge check, the secondary families that decide sign agreement, the Oracle Lens mechanism tests on fresh items, an interim write-up on day 7, the planted-bridge calibration, the Part A replications, labelled exploratory cells, and the last four days for figures, a statistics audit, the deviations log, the write-up and red-teaming with Fable.",
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
