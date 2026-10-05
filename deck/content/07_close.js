// Where we stand: ledger, answer, open checks, limitations, sprint plan, close.

module.exports = [
  {
    type: "anchor",
    section: "Where we stand",
    stage: 4,
    title: "What we know so far (4 / 4): the full ledger",
    notes:
      "The full ledger. Dark chips are established on full data, ochre chips are pilot-scale (60 items per feature, excluded from the main analysis), grey are tentative. " +
      "Read top to bottom: sound instruments; a gate that leaves one clean non-workspace feature, which is positional; decoders that read workspace features but not col; an NLA that reads the tail of verbalisable features and generates its story from the latent; and a benchmark whose targets are recoverable from the prompt alone.",
  },
  {
    type: "answer",
    section: "Where we stand",
    kicker: "Our provisional answer to the title question",
    title: "Read what the model can't say, or re-solve what it read? Neither: the NLA generates",
    question: "Do NLAs read what the model can't say, or re-solve what it read?",
    answer: "Neither.",
    answerBody: "The non-J content Qwen uses is positional bookkeeping, and neither decoder reads it, even when its downstream decision is lens-readable. The NLA doesn't re-solve either: it writes a story consistent with the latent. What it reads beyond J-Lens is the tail of features the model can already verbalise.",
    confidence: "Confidence: moderate for col with the NLA; lower for OL; low for 'non-J content in general'. Check (e) resolved; title decided at the freeze.",
    caveats: [
      "n = 1 clean non-J feature; 'non-J = positional bookkeeping' is a survey of ~12 features, not a theorem.",
      "OL's col null is less secure than the NLA's: OL passes K3 on language only (tense 0.55) and is template-fragile (0.39–0.87).",
      "Templates: the col claim holds 'in any phrasing they use'; main run adds continuation-style candidates.",
      "col's newline decision is weakly J-visible (soft newline mass AUROC 0.58), so count and decision cells are separate.",
      "Pilot-scale numbers (60 items per feature, excluded split); the tense-tail result is a pre-registered secondary.",
    ],
    notes:
      "So, the title question: does the NLA read what the model can't say, or re-solve what it read? Our provisional answer is neither. " +
      "It doesn't read what the model can't say: the content outside the J-space that Qwen actually uses looks like positional bookkeeping, and neither decoder reads col, even when we make its downstream newline decision visible to the J-Lens, although both read workspace features at the same dose. " +
      "It doesn't re-solve: when we plant a different bridge inside the activation, it rewrites the prompt to fit. Its language model generates a story consistent with the latent. What it reads beyond the J-Lens is the distributed tail of features the model can already verbalise, like tense. And on WorkspaceBench the target information in its readouts is recoverable from the prompt text alone. " +
      "Confidence differs by claim: moderate for the NLA on col, lower for the Oracle Lens because it only passes calibration on language and is template-fragile, and low for any general statement about non-J content. The working sentence from D21 is: 'On Qwen3.6-27B at L42, the content outside the J-space that the model actually uses is positional bookkeeping, and neither the NLA nor Oracle Lens reads it in any phrasing, even when its downstream decision is lens-readable; what the NLA reads beyond J-Lens is the distributed tail of features the model can already verbalise, and on WorkspaceBench its successes are matched by a text-only model.' Check (e) is now resolved in its favour, so the WorkspaceBench clause stands in its careful form; the title is decided at the freeze.",
  },
  {
    type: "checks",
    section: "Where we stand",
    kicker: "Open checks before the pre-registration freezes",
    title: "Five checks before we freeze: one resolved, four still open",
    checks: [
      { tag: "e", head: "Re-solver consistency: DONE", body: "Resolved (D22): an instruction mismatch. Same template: prompt 0.89 vs readout 0.60 / 0.56; prompt + readout 0.88; residual 4 / 2 of 400 hits are regex-form artefacts.", done: true },
      { tag: "#", head: "Next-digit in-J number cell", body: "Next token in a numbered list ('next digit 5 vs 7'): maximally J. Tracked → the col null is about location; not tracked → Part B is scoped to content type." },
      { tag: "q", head: "q90 behavioural validation", body: "Report the realised newline log-odds shift at the exact dose and direction where the null is claimed." },
      { tag: "d", head: "Class-mean col direction", body: "Add a binary short-vs-long-line direction next to the regression-covariance direction (closer in kind to language/tense)." },
      { tag: "t", head: "Natural col vs TextNLA", body: "The NLA's natural col AUROC (0.60) needs a text-only decoder on the same items: long vs short lines are visible without counting." },
    ],
    changes: "Main-run changes (D21): |z| ≥ 3 against both bands · equivalence test for col (SESOI: fraction of probe 0.25) · content token sets in every leak check · raw-direction newline-continuation cell primary · tense 4r* NLA-vs-OL as secondary · planting α ≤ 1.5 primary.",
    notes:
      "Five checks before the freeze. The first, (e), is done: the apparent inconsistency between readout-only re-solving and the text arm was an instruction mismatch, and the same-template comparison shows prompt 0.89 against readout 0.60 and 0.56, with prompt plus readout at 0.88. So the WorkspaceBench sentence stands, in its careful form. Four remain open. " +
      "The next-digit cell settles location versus content type for the col null. The behavioural check confirms the push we claim the readers ignore actually moves the model's newline behaviour at that dose. The class-mean direction removes a difference in kind between col and the in-J features. And the natural col result needs a text-only baseline. " +
      "The main-run statistical changes are on the bottom line: a stricter threshold, an equivalence test so the col null is a positive claim of 'no effect larger than X', content sets in every leak check, and the newline-continuation cell as primary.",
  },
  {
    type: "grid",
    section: "Where we stand",
    kicker: "Limitations",
    title: "Threats to validity we are tracking, and what addresses each",
    items: [
      { head: "One non-J feature", body: "No second clean feature in the ~12-feature survey; non-J content looks positional. Stated as a scope limit." },
      { head: "Count-verbalisation gap", body: "Numbers never enter Qwen's J-space (col_report, list index), so there is no in-J count control. The next-digit cell is the planned fix." },
      { head: "Template sensitivity", body: "OL tense 0.39–0.87 across templates. Templates mined and frozen before steering; continuation-style col candidates added." },
      { head: "Pilot scale", body: "col, K3 and tense-tail numbers: 60 items per feature on a declared, excluded split. Main run: 240 col items, two directions." },
      { head: "Direction choice", body: "col probe R² 0.655 (a curved manifold); covariance vs class-mean direction. Adds the binary direction." },
      { head: "J-space definition", body: "No spectral elbow; two lenses differ (cosine 0.79). Claims say 'outside the top-r J-space', at r* and 4r*, under both lenses." },
      { head: "Judges", body: "Claim precision is judge-based (κ 0.72–0.77 judge vs judge). 300 human-labelled claims still to do." },
      { head: "Scope", body: "One model, one layer (42). OL is off-grid at 42 but no worse than at 40/44. Planted doses are large (0.49 of ‖h‖)." },
    ],
    notes:
      "Threats to validity, each with what addresses it. The big two are the first two. One feature: the survey explains why there's no second one, so it's a scope limit, not something the sprint can fix. And the count-verbalisation gap: numbers simply don't enter Qwen's J-space, so we have no in-workspace count control, and can't fully rule out that readers fail on col because they can't talk about counts. The next-digit cell is designed to close that. " +
      "The rest are standard: template sensitivity, pilot scale, direction choice, the J-space definition, judge validity, and scope.",
  },
  {
    type: "gantt",
    section: "Where we stand",
    kicker: "Sprint plan · 19 October – 1 November",
    title: "The sprint replicates the col null with positive controls, with kill checks on days 2, 3 and 7",
    days: ["19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "1"],
    rows: [
      { label: "Part B: col null replication (raw newline cell primary; 4r*; class-mean direction)", from: 1, to: 1, color: "nla" },
      { label: "Upstream (layer 38) arm, q90 behavioural check, leak test, FVE guard", from: 2, to: 2, color: "nla", check: "K2" },
      { label: "Positive controls at matched dose: language, tense, next-digit cell", from: 3, to: 3, color: "nla", check: "K3" },
      { label: "Tense 4r* NLA vs OL (secondary); text-only decoders; natural col vs TextNLA", from: 4, to: 4, color: "text" },
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
      "The sprint is 19 October to 1 November, framed as a confirmatory replication of a pilot null, with two positive controls and one secondary, in that order. Day 1 replicates the col null at full scale, with the raw-direction newline-continuation cell as primary and the class-mean direction added. Day 2 is the upstream arm with the q90 behavioural check: kill check K2. Day 3 the positive controls at matched dose, including the new next-digit cell: K3, which passed in the pilot. Day 4 the tense-tail secondary and the text-only baselines. Then an interim write-up, planted intermediates (K4), the re-solver consistency check and human labels, robustness, an optional split-injection replication, and three days for figures, write-up and red-teaming. " +
      "Pivot rule: we pivot only if K3 fails, or if col and every in-J control are flat for every reader.",
  },
  {
    type: "closing",
    section: "Where we stand",
    title: "Contributions so far, and what would change our minds",
    contributions: [
      "A calibrated, random-subspace-controlled gate for 'non-workspace' content (the standard survival test is uninformative), and a survey: on Qwen L42 the non-J content the model uses is positional.",
      "A text-fixed causal pipeline (centred steering, null bands, two-sided leak detector, matched-dose controls) and its first answer: decoders read workspace features but not col.",
      "Evidence that the NLA generates rather than re-solves: it reads the tail of verbalisable features and rewrites the prompt to fit the latent.",
      "WorkspaceBench readouts' targets are recoverable from the prompt (0.89 vs 0.60 / 0.56); OL's lead is a best-of-11 effect.",
    ],
    open: [
      "A reader tracking col, or the next digit, at a leak-free dose in the main run.",
      "A second non-J feature that is not a count, on another model or layer.",
      "Readouts carrying target information beyond the prompt on richer, non-regex benchmarks.",
    ],
    notes:
      "To close, the contributions so far. One: a calibrated gate for non-workspace content, and a survey showing the non-J content Qwen uses at layer 42 is positional. Two: a causal pipeline that separates reading from re-solving, and its first answer for col. Three: evidence that the NLA generates a story from the latent rather than re-solving the text. Four: a benchmark decomposition and a protocol re-analysis. " +
      "What would change our minds: a reader tracking col or the next digit at a leak-free dose in the main run; a non-count non-J feature; or readouts beating the prompt on benchmarks beyond these four regex families. Questions welcome.",
  },
];
