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
    type: "table",
    section: "Where we stand",
    kicker: "Where we stand · results by experiment",
    title: "Results so far by experiment: established on full data, or pilot (excluded from the main run)",
    columns: ["Experiment", "What it tested", "Headline result", "Status"],
    colW: [2.0, 2.7, 6.1, 1.3],
    small: true,
    fontSize: 10.5,
    rows: [
      ["00 NLA gate (K0)", "NLA layer convention, FVE", "Reads block-42 output (cosine 0.9992 vs 0.974); FVE 0.761 reproduced", "Established"],
      ["01 WorkspaceBench replication", "12 published numbers, 3 readers", "12/12 in our 95% CIs; OL's lead = any-layer selection (+0.08–0.22)", "Established"],
      ["02 Non-J feature gates", "8 features, 4 gates, layer 42", "Survival test uninformative; col the only clean NON-J (causal 49× random)", "Established"],
      ["04 Backup features", "9 more candidates", "None qualify; indentation decodable-unused; numbers never enter J-space", "Established"],
      ["05 Planted intermediates", "In-J calibration, swap at layer 38", "431/614 pass K4; readers report the planted bridge 0.98–1.00; clue rewritten 12/12", "Established"],
      ["06 Re-solving decomposition", "Text-only baselines, WorkspaceBench", "Prompt 0.89 vs readouts 0.60 / 0.56; prompt + readout 0.88; precision 0.62 / 0.52 vs 0.92", "Established"],
      ["08 col pilot + K3", "Do readers follow col at matched dose?", "Fixed templates in band (≤ 0.53); in-J features tracked 0.83–0.92; two candidate channels", "Pilot"],
      ["09 OL mechanism", "Acting vs reading, OL's channel", "col gain 0.99 / 1.06, one-for-one with the model; tense 4r* tracked, model readout flat", "Pilot"],
      ["10 Operand gate", "Operand digits at '=' (after M. Kenney)", "J-depleted under two bases; probe 0.83 / 0.93; no leak-clean edit moves the answer (0/117)", "Pilot"],
    ],
    footnote: "Qwen3.6-27B, layer 42 throughout; NLA iter_000400, Oracle Lens olens_s3d_rl600, J-Lens n1000 (J++ second basis in exp 10). Pilot rows: declared pilot split, excluded from the main analysis. 03 (smoke) and 07 (templates) are plumbing.",
    notes: "One table for the whole prep phase, experiment by experiment. The top block is established on full data: the instruments, the gate, the backup survey, the planted-bridge calibration and the benchmark decomposition. The bottom block is pilot-scale and excluded from the main analysis: the col pilot with its two candidate channels, the Oracle Lens mechanism, and the externally prompted operand gate. The main run confirms or rejects the pilot rows.",
  },
  {
    type: "tiers",
    section: "Where we stand",
    kicker: "Strength of evidence",
    title: "What we can say strongly today, and what we can't",
    tiers: [
      { head: "STRONG", tone: "strong", share: 0.42, items: [
        "WorkspaceBench target information is recoverable from the prompt: 0.89 from prompt vs 0.60 / 0.56 from readouts; prompt + readout 0.88 (n = 400).",
        "OL's published lead is per-item any-layer selection (+0.08 to +0.22), reproduced exactly.",
        "The standard J-removal survival test can't tell features apart: all keep ≥ 98%, topic included; random equal-rank removal is identical.",
        "col is a clean non-J feature under two lenses: causal 49× random; 79% kept after J-removal vs 12–23% for non-count features.",
        "Method lessons: the dilution artefact and centred steering; a two-sided, calibrated leak detector; the null band, not n, limits power.",
      ] },
      { head: "MODERATE", tone: "moderate", share: 0.31, items: [
        "Non-J content the model uses on Qwen L42 looks like positional bookkeeping (a survey of ~16 gate runs on ~13 features, not a theorem).",
        "WorkspaceBench's frozen prompt-only floors look ~0.14 too low; cause unknown.",
        "The NLA confabulates clues that match a planted bridge (12 / 12).",
      ] },
      { head: "NOT YET", tone: "notyet", share: 0.27, items: [
        "Any claim that the readers read non-J content.",
        "The pilot candidates came after ~514 looks; confirming or rejecting them is the sprint's job.",
        "Four outcomes and their wording are written in advance.",
      ] },
    ],
    notes:
      "Before the answer, an honest sort of the evidence. Strong, on full data and robust to the checks we've run: the WorkspaceBench decomposition, 0.89 from the prompt against 0.60 and 0.56 from the readouts, with prompt plus readout at 0.88 on 400 items; the Oracle Lens's published lead being per-item any-layer selection, worth 0.08 to 0.22, reproduced exactly; the survival test failing to separate features, with topic keeping at least 98% and random-rank removal identical; col as a clean non-J feature under two lenses, causal at 49 times random and keeping 79% of its effect after J-removal against 12 to 23% for the non-count features; and the method lessons. " +
      "Moderate: that non-J content on this model is positional bookkeeping, which is a survey of about sixteen gate runs; the prompt-only floor looking about 0.14 too low, cause unknown; and the NLA confabulating matching clues, 12 of 12, which is a small sample. " +
      "Not yet: any claim that these readers read non-J content. The pilot candidates came after about 514 looks. Confirming or rejecting them is what the sprint is for, and the four possible outcomes and their wording are already written down.",
  },
  {
    type: "branches",
    section: "Where we stand",
    kicker: "Our answer to the title question: four pre-registered branches",
    title: "Do NLAs read what the model can't say? The main run decides between four pre-registered answers",
    branches: [
      { head: "If OL's channel replicates · 1b (likeliest) or 1a", answer: "1b: Not the NLA, and OL only through its next-token continuation channel.", body: "1a, if the NLA phrasing also replicates: 'Barely, in narrow channels, and not in the language they use to describe what they read.' Say 'channel-specific sensitivity', never 'reads'." },
      { head: "If OL's channel fails · 2 or 1c", answer: "2: No, on the one natural non-J feature the model uses.", body: "1c, if only the NLA phrasing replicates: 'Barely, and in one phrasing.' Both decoders read workspace features at the same dose, so a null is interpretable." },
    ],
    stands: [
      "Part A: the standard non-J test is uninformative; non-J content on Qwen L42 looks like positional bookkeeping (a ~13-feature survey).",
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
    title: "Pre-freeze checks: all five done",
    checks: [
      { tag: "e", head: "Re-solver consistency: DONE", body: "An instruction mismatch (D22). Same template: prompt 0.89 vs readout 0.60 / 0.56; prompt + readout 0.88.", done: true },
      { tag: "q", head: "q90 behavioural check: DONE", body: "J-removed col push: newline log-odds +0.72 (z +55), wrap −1.55 chars, J-Lens unmoved (D24).", done: true },
      { tag: "t", head: "Natural col vs text: DONE", body: "Text-only Gemini 0.97; the NLA's 0.60 falls to 0.51 after regressing out text (D23).", done: true },
      { tag: "m", head: "OL mechanism (exp 09): DONE", body: "col one-for-one with the model (0.99 / 1.06); tense 4r* tracked where the model's readout is flat. The binary verdict is demoted to 3 descriptive quantities (rev 4.2).", done: true },
      { tag: "d", head: "Class-mean col direction: DONE", body: "Round 5: the binary short-vs-long direction matches the regression direction (J-removed newline shift z +55 vs +54; reader cells unchanged).", done: true },
    ],
    changes: "Confirmatory design (prereg rev 4): P1–P4 · K = 100 nulls per band · Holm over 4 · sign agreement · negatives N1–N5 (TOST) · T1 judge check · fresh items (240 col / 160 tense / 120 language) · freeze Oct 18 with a hash manifest.",
    notes:
      "Where the pre-freeze checks stand. Done: the re-solver consistency check, the behavioural validation at q90, and the natural-col text baseline.  Also done: the Oracle Lens mechanism pilot, experiment 09, whose binary verdict we demoted to three descriptive quantities because the test is confounded by context. The class-mean col direction matches the regression direction, so that check is closed too. The bottom line is the confirmatory design.",
  },
  {
    type: "grid",
    section: "Where we stand",
    kicker: "Limitations",
    title: "Threats to validity we are tracking, and what addresses each",
    items: [
      { head: "One non-J feature", body: "No second clean feature in the ~13-feature survey; non-J content looks positional. Stated as a scope limit." },
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
