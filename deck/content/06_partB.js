// Part B: col, deep (experiments 02, 03, 05, 07, 08). Pilot numbers are flagged `pilot: true`
// and get a "PILOT · excluded from main analysis" chip.

const BAND = [{ name: "null band, upper (+1.96)", value: 1.96, color: "line" }, { name: "null band, lower (−1.96)", value: -1.96, color: "line" }];

module.exports = [
  {
    type: "section",
    section: "Part B",
    num: "02",
    title: "Part B: col has no robust descriptive reading",
    sub: "…although both decoders read workspace features at the same dose; two candidate narrow channels go to confirmation",
    notes: "Part B. First sentence, as in the write-up: this is one feature. Generalisation is open, and Part A told us why it's hard to get a second.",
  },
  {
    type: "chart",
    section: "Part B",
    kicker: "Part B · why col",
    title: "col is a doubly clean test: the model relies on it, and Qwen can't recompute it from the text",
    chart: {
      kind: "bar",
      categories: ["col", "bracket depth", "pred. POS", "tense", "language"],
      series: [{ name: "causal effect kept after removing the top-1,536 J-space", values: [0.79, 0.41, 0.33, 0.19, 0.05] }],
      pointColors: ["probe", "muted", "jl", "jl", "jl"],
      min: 0, max: 1, fmt: "0%", labels: true, legend: false,
      title: "Steering effect that survives J-removal (layer 42, q99)",
    },
    chartW: 6.6,
    side: {
      stats: [
        { value: "49×", label: "col's J-removed steering moves newline log-odds +0.99 vs ≤ 0.02 for random directions", color: "ink", size: 34 },
        { value: "7%", label: "Qwen's exact accuracy when asked to count characters without chain of thought", color: "ink", size: 34 },
        { value: "ρ 0.42", label: "free self-report vs truth; error no better than guessing the mean (MAE 16.4 vs 15.7)", color: "ink", size: 34 },
      ],
      note: "So a Qwen decoder can't re-solve col by counting. Nuance: the newline decision col feeds is weakly J-visible (soft newline mass AUROC 0.58).",
    },
    notes:
      "Why col is a good test. First, the model really uses it: steering along its J-removed direction shifts the log-odds of a newline by +0.99, against at most 0.02 for random directions, 49 times larger. And it keeps 79% of its full causal effect after J-removal, where tense keeps 19%. " +
      "Second, and this is the 'doubly clean' part, Qwen can't count characters without chain of thought: 7% exact. Self-report is poor too: Spearman 0.42, with error no better than guessing the mean. Since both decoders are built on Qwen, they can't solve col by reconstructing the text and counting. So if a reader follows col at a valid, leak-free dose, that is activation reading, not re-solving. " +
      "One nuance Fable flagged and experiment 04 confirmed: the count itself is non-J, but the newline decision it feeds is weakly J-visible. Under the strict 'full distribution' functional test, newline mass tracks col at AUROC 0.58, while the top-10 form is at chance, 0.501. So we keep separate templates for the count ('a long line') and the decision ('a line break is coming'), and include newline tokens in the leak set.",
  },
  {
    type: "chart",
    section: "Part B",
    kicker: "Part B · method lesson 1",
    title: "Method lesson: an apparent NLA-vs-Oracle Lens contrast was partly an artefact of how we steered",
    chart: {
      kind: "bar",
      categories: ["Oracle Lens", "NLA", "J-Lens"],
      series: [
        { name: "additive ±Δ (original)", values: [0.45, 0.6, 0.46], color: "line" },
        { name: "centred ±Δ (equal norm)", values: [0.49, 0.62, 0.46], color: "muted" },
      ],
      pointColorsBySeries: null,
      lines: [{ name: "chance", value: 0.5, color: "nla" }],
      min: 0.3, max: 0.7, fmt: "0.00", labels: true, legend: "b",
      title: "Toy: J-removed language direction (FR/DE, n = 40 pairs), q99, layer 42",
    },
    chartW: 6.4,
    side: {
      steps: [
        { head: "1 · The puzzle", body: "Oracle Lens anti-tracked the J-removed direction (0.45); the NLA tracked it (0.60)." },
        { head: "2 · Fable's test", body: "Norm-preserving steering changed nothing: both decoders renormalise their input anyway." },
        { head: "3 · Real cause", body: "Unequal dilution when the item already carries the feature. Fix: centred steering, equal norm on both sides." },
        { head: "4 · Result", body: "OL → 0.49, inside the null band. NLA 0.62 (+8.3 SD). J-Lens still anti-tracks: a real leak, so by our own rule the toy is not a non-J read." },
      ],
    },
    notes:
      "A method story worth telling, because it shows why the guards exist. In the pipeline smoke test on a toy feature, French vs German, the J-removed direction gave a beautiful contrast: the NLA tracked it, 0.60, and the Oracle Lens anti-tracked, 0.45. Exactly the NLA-beats-OL pattern Neel's question would predict. " +
      "Fable argued the ± contrast isn't immune to norm effects and suggested norm-preserving steering. We tested it: it's a no-op, because both decoders only see the direction of their input. The NLA injects at its own norm, the Oracle Lens at 16000 times the unit vector. " +
      "The real cause was dilution: an item that already carries the feature ends up with its other content weighted differently on the plus and minus sides. Centred steering, which removes the item's own component and gives both sides equal norm, fixed it. The Oracle Lens went to 0.49, inside the null band. The NLA stayed at 0.62, 8.3 SD above the band. " +
      "But the J-Lens still anti-tracked, 4 to 7 SD outside its band: the J-removed direction genuinely leaks into J-Lens-visible directions. So by our own leak rule even the NLA's tracking here couldn't be claimed as non-J reading. The detector is strict enough to matter.",
  },
  {
    type: "chart",
    section: "Part B",
    kicker: "Part B · method lesson 2",
    title: "Method lesson: removing the top-384 J-space isn't enough for col; removing 1,536 is leak-free at layer 42",
    pilot: true,
    chart: {
      kind: "bar",
      categories: ["layer 40", "layer 42", "layer 44"],
      series: [
        { name: "J-removed at r* (384; 256 at L44)", values: [2.7, 2.8, 3.0], color: "jl" },
        { name: "J-removed at 4r* (1,536; 1,024 at L44)", values: [-0.3, 0.1, 2.6], color: "jlPale" },
      ],
      lines: BAND,
      min: -3, max: 4, fmt: "0.0", labels: true, legend: "b",
      valTitle: "J-Lens newline score shift, z vs J-complement band (q90)",
    },
    chartW: 6.6,
    side: {
      bullets: [
        [{ text: "Leak: ", options: { bold: true } }, { text: "at r* the J-removed col direction still pushes the J-Lens towards newline (z +2.6 to +3.0). The main run steers at 4r* at layer 42." }],
        [{ text: "'CI excludes 0.5' is too weak: ", options: { bold: true } }, { text: "the NLA's numeral template at r* gave 0.54 [0.53, 0.57], but random directions move it as much (z +1.4 / +1.8). The null band must be part of every primary test." }],
        [{ text: "The FVE guard measures norm: ", options: { bold: true } }, { text: "at q99 random directions also fail it (−0.045 to −0.062). q90 is primary." }],
      ],
    },
    notes:
      "Three more lessons from the col pilot, 60 declared-pilot passages excluded from the main analysis. " +
      "First, leaks. Removing the top-384 J-space from col's direction is not enough: the steered direction still nudges the J-Lens newline score, z +2.6 to +3.0 at all three read layers, beyond the two-sided band. Removing the top 1,536 is clean at layers 40 and 42; at layer 44, where 4r* is only 1,024, it still leaks. So the main run steers at 4r* at layer 42. " +
      "Second, the prereg's test 'CI excludes 0.5' is too weak. The NLA's numeral template at r* gave 0.54 with an interval excluding 0.5, but random directions of the same size move it just as much. The null band has to be part of every primary test. " +
      "Third, the FVE guard, which checks the NLA's reconstruction doesn't degrade, fails for random directions at the q99 norm too. It's measuring norm, not col, so q90 is primary.",
  },
  {
    type: "chart",
    section: "Part B",
    kicker: "Part B · the col result",
    title: "With the fixed pilot templates, no reader follows col beyond what random directions do",
    pilot: true,
    chart: {
      kind: "bar",
      categories: ["'long line' (count, primary)", "'about 60 characters' (count, numeric)", "'line break coming' (decision)"],
      series: [
        { name: "NLA", values: [0.6, 1.0, 0.6], color: "nla" },
        { name: "Oracle Lens", values: [-0.6, 1.2, -0.3], color: "ol" },
        { name: "J-Lens (leak detector)", values: [-0.9, -0.5, 0.1], color: "jl" },
      ],
      lines: BAND,
      min: -3, max: 3, fmt: "0.0", labels: true, legend: "b",
      valTitle: "z vs J-complement null band",
      title: "Leak-free cell: layer 42, centred, top-1,536 J-space removed, q90",
    },
    chartW: 7.4,
    side: {
      stats: [
        { value: "0.50–0.53", label: "NLA AUROC across template families (Oracle Lens 0.49–0.51)", color: "nla", size: 30 },
        { value: "≤ 0.12", label: "NLA fraction of probe information (OL ≤ 0.03; pre-registered bar 0.25)", color: "nla", size: 30 },
        { value: "0 / 5,760", label: "free readouts mentioning line length (4,080 NLA, 1,680 OL)", color: "ink", size: 30 },
      ],
      note: "Power: the null band, not n, limits col. NLA z at n → ∞ would be only +0.6 to +1.4, while language and tense clear the same band by 10–30 SD. NLA − OL ΔAUROC +0.015 to +0.028: H3 (bar 0.05) will likely fail for col.",
    },
    notes:
      "Here's the col result, on the pilot split. The leak-free cell: layer 42, centred steering, the top-1,536 J-space removed, 90th-percentile dose. Three template families: the count in words, the count as a number, and the newline decision. " +
      "Every bar is a z-score against random directions of the same norm. All of them sit inside ±1.96, in fact within ±1.5 for every family against both null bands. In AUROC terms the NLA is at 0.50 to 0.53 and the Oracle Lens at 0.49 to 0.51. As a fraction of what the probe recovers, which is 1.0 by construction: NLA at most 0.12, Oracle Lens at most 0.03, against a pre-registered bar of 0.25. " +
      "Free text says the same: across 5,760 natural and steered readouts, neither decoder ever mentions line length. " +
      "Power, since people ask: for col the binding constraint is the null band, not the number of items. Random directions genuinely move the decoders, and that direction-to-direction spread doesn't shrink with n: even at infinite n the NLA's z on these cells would be +0.6 to +1.4. But it isn't a generic power failure: workspace features clear the same band by 10 to 30 SD (per-item effect sizes +1.3 to +1.5 vs about −0.2 to −0.3 for col). The NLA-minus-OL difference on col is +0.015 to +0.028, below the pre-registered 0.05 bar, so H3 will likely fail for col, and we say so at the freeze. " +
      "With these fixed templates, even the raw col direction, which strongly moves the J-Lens's newline readout (AUROC up to 0.84), doesn't move either decoder. But a null only means something if the readers would have noticed a workspace feature at the same dose. That's the next slide.",
  },
  {
    type: "chart",
    section: "Part B",
    kicker: "Part B · Figure 1 · matched-dose calibration (K3: PASS)",
    title: "The readers do track workspace features at the same steering strength; with fixed templates, col stays at chance",
    pilot: true,
    chart: {
      kind: "bar",
      categories: ["language (RU vs PL)", "tense (past vs present)", "col (raw direction)"],
      series: [
        { name: "NLA", values: [0.83, 0.83, 0.48], color: "nla" },
        { name: "Oracle Lens", values: [0.76, 0.55, 0.48], color: "ol" },
        { name: "J-Lens (label words)", values: [0.68, 0.79, 0.52], color: "jl" },
      ],
      lines: [{ name: "chance (0.5)", value: 0.5, color: "muted" }],
      min: 0.3, max: 1, fmt: "0.00", labels: true, legend: "b",
      valTitle: "FollowFeature AUROC",
      catTitle: "q90 dose, centred, raw direction, layer 42 (‖Δ‖/‖h‖: 0.095 · 0.096 · 0.108)",
    },
    chartW: 7.6,
    side: {
      rows: [
        { head: "NLA", body: "language z +25 / +31, tense +12 / +18 (isotropic / J-complement bands); col inside both", color: "nla" },
        { head: "Oracle Lens", body: "language z +11 / +21; tense +2.3 / +2.5, below the 0.65 bar; col inside both", color: "ol" },
        { head: "J-Lens", body: "language z +3.3 / +24, tense +8.3 / +21 (a competitor for in-J content)", color: "jl" },
      ],
      note: "Workspace features were steered slightly less (0.095 of ‖h‖) than col (0.108). K3 rule: OL or NLA ≥ 0.65 on ≥ 1 in-J feature at col's dose. Passed: NLA on both, OL on language.",
    },
    notes:
      "Figure 1, and the calibration that makes the col null interpretable: kill check K3, which passed. Same pipeline, same random null directions, same 90th-percentile dose rule, raw directions, centred steering. In fact the workspace features got a slightly smaller push, 9.5% of the activation norm, than col, 10.8%. " +
      "The NLA tracks language at 0.83 and tense at 0.83, 12 to 31 SD beyond both null bands. The Oracle Lens tracks language at 0.76. The J-Lens, a competitor here, tracks both. On col, at a larger push, both decoders sit at 0.48, inside both bands. " +
      "Now the asymmetry, which I want to be upfront about. The NLA's col null is secure: it reads both workspace features at this dose. The Oracle Lens's is less secure: on tense it only reaches 0.55, out of band but below the pre-registered 0.65 bar, and its tense tracking swings from 0.39 to 0.87 depending on the template. So for the Oracle Lens, 'doesn't read col' could partly be 'our templates don't speak its language'. " +
      "Pilot split, 60 items per feature, excluded from the main analysis.",
  },
  {
    type: "chart",
    section: "Part B",
    kicker: "Part B · equal relative norm",
    title: "At exactly equal steering norm, the decoders track language and tense; col stays at chance",
    pilot: true,
    chart: {
      kind: "bar",
      categories: ["language, 0.11", "language, 0.14", "tense, 0.11", "tense, 0.14", "col, 0.11", "col, 0.14"],
      series: [
        { name: "NLA", values: [0.87, 0.92, 0.85, 0.89, 0.48, 0.47], color: "nla" },
        { name: "Oracle Lens", values: [0.80, 0.85, 0.56, 0.57, 0.48, 0.47], color: "ol" },
      ],
      lines: [{ name: "chance (0.5)", value: 0.5, color: "muted" }],
      min: 0.3, max: 1, fmt: "0.00", labels: true, legend: "b",
      valTitle: "FollowFeature AUROC",
      catTitle: "feature, steering norm ‖Δ‖/‖h‖ (raw direction, centred, layer 42)",
    },
    chartW: 7.7,
    side: {
      stats: [
        { value: "z +18 to +31", label: "NLA on language and tense vs the same null directions; OL language z +20", color: "nla", size: 28 },
        { value: "0.56–0.57", label: "OL on tense: out of band but below the 0.65 bar, so OL's col null is less secure than the NLA's", color: "ol", size: 28 },
      ],
      note: "Caveat: templates matter (OL tense 0.39–0.87 across templates), so the col claim is 'with the frozen col templates'. The main run adds continuation-style col candidates.",
    },
    notes:
      "A stricter version of the same check: instead of each feature's own 90th-percentile dose, every item is pushed by exactly 11% or 14% of its activation norm, for language, tense and col alike, with the same null directions. " +
      "At 0.11 the NLA tracks language at 0.87 and tense at 0.85, the Oracle Lens language at 0.80; at 0.14 they rise to 0.92, 0.89 and 0.85. col sits at 0.47 to 0.48 for both decoders at both norms, z −0.7 and −1.0 against the J-complement band. So the col null is not a dose-scale artefact. " +
      "The Oracle Lens on tense stays at 0.56 to 0.57, and that is a template story: across templates its tense tracking ranges from 0.39 ('- happened in the past') to 0.87 ('- was' vs '- is'). That's why the col claim is explicitly 'with these frozen templates', and why the main run adds Oracle-Lens-style continuation candidates for col.",
  },
  {
    type: "stats",
    section: "Part B",
    kicker: "Part B · behavioural validation at the claimed dose",
    title: "The J-removed col push changes what the model does, yet the J-Lens can't see it",
    pilot: true,
    stats: [
      { value: "+0.72", label: "newline log-odds shift from the J-removed (4r*) col push at q90, layer 42: z +55 (raw direction: +1.44, z +86)", color: "ink" },
      { value: "−1.55 chars", label: "shift in where generation wraps the line under the same J-removed push (raw: −6.8 chars)", color: "ink" },
      { value: "0.50", label: "J-Lens newline readout under that push; label and content sets clean (|z| ≤ 2.1)", color: "jl" },
    ],
    families: [
      { head: "Why this matters", body: "the dose at which we test the readers really moves the model's behaviour, so a reader that ignores it is ignoring used content." },
      { head: "Invisible to the J-Lens", body: "this is exactly 'non-workspace' content: used by the model, outside what the J-Lens reads." },
      { head: "Dose fork closed (D25)", body: "q90 is the primary dose, behaviourally validated; r11 is sensitivity only." },
    ],
    caveat: "So the question for the readers is well posed: the content is present, used and J-invisible at the dose we test.",
    notes:
      "Before asking whether readers follow col, we checked the push itself does something. At layer 42, at the 90th-percentile dose, the J-removed col direction shifts the log-odds of a newline by +0.72, 55 SD beyond random directions; the raw direction by +1.44. In free generation, the line wraps 1.55 characters earlier under the J-removed push, 6.8 under the raw one. Yet under the J-removed push the J-Lens newline readout stays at 0.50 and our label and content leak sets are clean. " +
      "So this is content the model uses, at the dose we test, that the J-Lens cannot see. That is the target for Neel's question. Following Fable round 6, q90 is the primary dose and the r11 cell is only sensitivity.",
  },
  {
    type: "table",
    section: "Part B",
    kicker: "Part B · two candidate narrow channels (retracts D21)",
    title: "With native phrasings, two narrow channels do follow col: candidates, not findings",
    pilot: true,
    columns: ["Channel", "Raw direction", "J-removed (4r*)", "What else we know", "Status"],
    colW: [3.0, 1.6, 2.4, 3.6, 1.5],
    rows: [
      ["OL next-token continuation (newline vs another word)", "0.76 (z +6.6)", "0.67–0.72 (z ≈ +3.9)", "linear in dose; adapter off: 0.40–0.44, so it needs the LoRA; keeps ~78% under J-removal vs the model's own ~70%", { label: "CANDIDATE", tone: "fail" }],
      ["NLA, one template: 'near the end of a hard-wrapped line'", "0.71 (z +5.0)", "q90 0.63 (z +3.0, borderline); r11 0.67 (z +3.7)", "rests mostly on this one template; fixed templates and the decision family are flat", { label: "CANDIDATE", tone: "fail" }],
      ["OL descriptive templates (line position, length)", "flat", "flat or anti-tracking", "pre-registered as a negative, with an equivalence test", { label: "NEGATIVE", tone: "diffuse" }],
      ["Free text, both decoders", "—", "—", "0 of 5,760 readouts mention line length", { label: "NEGATIVE", tone: "diffuse" }],
    ],
    pending: "MECHANISM TEST (EXP 09) RUNNING · PILOT",
    footnote: "Oracle Lens is the model + a LoRA: it reads what the model reads and says what the model can say. The 78% vs 70% gain match favours re-instatement (the LoRA hands the activation back to Qwen's own newline circuitry) over reporting; indentation, gain-ratio and adapter-off tests are running as a pilot (experiment 09).",
    notes:
      "This slide retracts a sentence from round 5. We had said neither decoder reads col 'in any phrasing'. That rested on fixed templates. With native, mined phrasings, two narrow channels do follow the J-removed col direction, leak-free. " +
      "The Oracle Lens's next-token continuation channel, scoring 'a newline comes next' versus 'another word', tracks col: 0.76 on the raw direction at z +6.6, and 0.67 to 0.72 with the top-1,536 J-space removed, at a conservative z of about +3.9. The shift is linear in dose, and with the adapter switched off, base Qwen in the same carrier prompt slightly anti-tracks (0.40 to 0.44), so it needs the LoRA. But the Oracle Lens is the model plus a LoRA: it reads what the model reads and says what the model can say. J-removal keeps about 78% of its shift per unit dose, close to the model's own behavioural 70%. That favours re-instatement: the LoRA hands the activation back to Qwen's own next-token circuitry, which predicts a newline, rather than the Oracle Lens reporting line position. Every descriptive Oracle Lens col template is flat or anti-tracking. And a newline-probability shift is only useful if you already know to measure it, so it isn't an interpretability readout. " +
      "The NLA has one phrasing, 'near the end of a hard-wrapped line', at 0.63 on the primary q90 cell, borderline, and 0.67 at the r11 sensitivity dose. " +
      "These are candidates, not findings, for the reason on the next slide.",
  },
  {
    type: "grid",
    section: "Part B",
    kicker: "Part B · forking paths, disclosed",
    title: "Forking paths, disclosed: the pilot had many looks, so the main run confirms four named primaries",
    pilot: true,
    items: [
      { head: "Many looks", body: "≈ 514 template-level cells and ≈ 2,000 z-values inspected across three pilot rounds. ≈ 0.5 false z ≥ 3 col cells expected by chance." },
      { head: "Primaries from the pilot", body: "P1 = the best of 11 NLA col templates; P2 = the only OL col family with signal. Chosen after seeing pilot data; the prereg says so." },
      { head: "4 primaries, Holm", body: "NLA col (named template, q90, 4r*), OL col (continuation, q90, 4r*), OL tense 4r*, NLA tense 4r*. Holm over 4." },
      { head: "Claim criterion", body: "z ≥ 3 against both bands (K = 100 null directions each), Holm over 4, sign agreement with the declared-secondary family." },
      { head: "Fresh items", body: "240 col, 160 tense, 120 language passages, disjoint from the pilot and template mining. No pilot number enters a test." },
      { head: "Templates frozen literally", body: "No more mining, no new families." },
      { head: "Pre-registered negatives", body: "OL descriptive col families N1–N5 flat: TOST, SESOI AUROC 0.55 (fraction of probe 0.10)." },
      { head: "Template-free check (T1)", body: "Two judges, one pre-written yes/no question, 6,720 free readouts on 120 fresh col passages." },
    ],
    notes:
      "I want to be upfront about forking paths. Across three pilot rounds we scored about 514 template-level cells and looked at roughly 2,000 z-values. Under a normal band about half a false z-of-3 col cell is expected by chance, and we saw two clusters. P1 is the single best of 11 NLA col templates; P2 is the only Oracle Lens col family with any signal, fixed before its round was scored. So these are hypotheses, and we expect them to shrink on fresh items. " +
      "The main run therefore names four primaries in advance, uses 100 null directions per band, Holm over four, requires sign agreement with a declared family of neighbouring cells, and runs on fresh, passage-disjoint items: 240 col, 160 tense, 120 language. No pilot number enters any confirmatory test. Two pre-registered negatives use equivalence tests, and a template-free judge check reads free text.",
  },
  {
    type: "table",
    section: "Part B",
    kicker: "How we'll confirm it · prereg revision 4",
    title: "Four pre-registered primaries, tested once on fresh items; P1 may well fail",
    columns: ["ID", "Reader", "Cell (all L42, centred, top-1,536 J-space removed, q90)", "Frozen template(s), + vs −", "Pilot (excluded)", "Prior"],
    colW: [0.55, 1.15, 3.0, 4.35, 1.7, 1.35],
    small: true,
    rows: [
      [{ text: "P1", bold: true }, "NLA", "col, 02 covariance direction; Δ = 7.098 (≈ 0.079 of ‖h‖)", "'Mid-sentence continuation located near the end / at the beginning of a hard-wrapped line.'", "0.63, z +5.6 / +3.5", "0.35"],
      [{ text: "P2", bold: true }, "Oracle Lens", "same as P1", "'- ↵the' vs '- the'; '- ↵and' vs '- and'; '- ↵of' vs '- of' (↵ = newline; mean of 3)", "0.67, z +3.9 / +3.9", "0.55"],
      [{ text: "P3", bold: true }, "Oracle Lens", "tense (+ = past), 02 class-mean direction; Δ = 5.228 (≈ 0.059)", "'- They were / are'; '- The village had / has'; '- It was / is'", "0.66, z +12.5 / +16.6", "0.90"],
      [{ text: "P4", bold: true }, "NLA", "same as P3", "'Historical / Descriptive register'; '… factual'; '… tone'", "0.59, z +7.5 / +9.2", "0.85"],
    ],
    footnote: "Claim = z ≥ 3 vs both bands (K = 100 each) and beats all 100 nulls · Holm over 4 · sign agreement with the declared-secondary family · not void (leak, FVE). Also pre-registered: PC1/PC2 positive controls (language, tense, r11), negatives N1–N5 (TOST, SESOI AUROC 0.55), template-free judge check T1. Honest prediction: P1's family-mean z is only +2.5–2.9, so winner's-curse shrinkage may fail it.",
    notes:
      "This is how we'll confirm it, from prereg revision 4. Four primary cells, each fully specified now: reader, direction, dose and the literal template strings. P1, the NLA's col template; P2, the Oracle Lens's next-token continuation pairs; P3 and P4, J-removed tense for the Oracle Lens and the NLA. All at layer 42, centred steering, top-1,536 J-space removed, 90th-percentile dose. " +
      "A primary is claimed only if its z is at least 3 against both null bands, each built from 100 random directions, it beats all 100, it survives Holm over four, its declared family of neighbouring cells agrees in sign, and it isn't voided by the leak or FVE guard. Around them: two positive controls, language and tense at matched norm; five pre-registered negatives for the Oracle Lens's descriptive col templates, tested for equivalence with a smallest effect of interest of AUROC 0.55; and a template-free judge check over 6,720 free readouts. " +
      "Power comes from the band, not n: going from 60 to 240 items moves P1's binding z by about 0.1. And the honest prediction: P1 sits only 0.5 SD above the bar in the pilot and its family mean is z +2.5 to +2.9, so winner's-curse shrinkage may well fail it. Our priors say so: 0.35 for P1, 0.55 for P2, 0.85 to 0.90 for the tense cells.",
  },
  {
    type: "anchor",
    section: "Part B",
    stage: 3,
    title: "What we know so far (3 / 4): workspace features yes; col only through candidate narrow channels",
    next: "Next → Part C: so what do the decoders read beyond the workspace?",
    notes:
      "Third checkpoint. At doses where both decoders read workspace features, their descriptive language doesn't follow col, even though the same push measurably changes the model's newline behaviour and is invisible to the J-Lens. Two narrow channels, the Oracle Lens's continuation and one NLA phrasing, are candidates for the main run, with the forking-paths caveat. All pilot-scale.",
  },
];
