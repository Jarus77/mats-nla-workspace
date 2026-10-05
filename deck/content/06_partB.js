// Part B: col, deep (experiments 02, 03, 05, 07, 08). Pilot numbers are flagged `pilot: true`
// and get a "PILOT · excluded from main analysis" chip.

const BAND = [{ name: "null band, upper (+1.96)", value: 1.96, color: "line" }, { name: "null band, lower (−1.96)", value: -1.96, color: "line" }];

module.exports = [
  {
    type: "section",
    section: "Part B",
    num: "05",
    title: "Part B: no decoder reads col",
    sub: "…although both read workspace features at the same dose (n = 1 non-J feature, pilot scale)",
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
    title: "The col result: no reader follows col beyond what random directions of the same size do",
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
    },
    notes:
      "Here's the col result, on the pilot split. The leak-free cell: layer 42, centred steering, the top-1,536 J-space removed, 90th-percentile dose. Three template families: the count in words, the count as a number, and the newline decision. " +
      "Every bar is a z-score against random directions of the same norm. All of them sit inside ±1.96, in fact within ±1.5 for every family against both null bands. In AUROC terms the NLA is at 0.50 to 0.53 and the Oracle Lens at 0.49 to 0.51. As a fraction of what the probe recovers, which is 1.0 by construction: NLA at most 0.12, Oracle Lens at most 0.03, against a pre-registered bar of 0.25. " +
      "Free text says the same: across 5,760 natural and steered readouts, neither decoder ever mentions line length. " +
      "Even the raw col direction, which strongly moves the J-Lens's newline readout (AUROC up to 0.84), doesn't move either decoder. But a null only means something if the readers would have noticed a workspace feature at the same dose. That's the next slide.",
  },
  {
    type: "chart",
    section: "Part B",
    kicker: "Part B · Figure 1 · matched-dose calibration (K3: PASS)",
    title: "The readers do track workspace features at the same steering strength; they just don't track col",
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
    kicker: "Part B · content type, not location",
    title: "Even when col's downstream decision is visible to the J-Lens, neither decoder follows it",
    pilot: true,
    stats: [
      { value: "0.73–0.84", label: "J-Lens newline readout AUROC when we push the raw col direction (z +3.5 to +4.6 vs the isotropic band)", color: "jl" },
      { value: "≤ 0.54", label: "NLA on the same raw push, across count, numeric and newline-decision templates (q99)", color: "nla" },
      { value: "≤ 0.50", label: "Oracle Lens on the same push, same three template families (q99)", color: "ol" },
    ],
    families: [
      { head: "If location were the story", body: "the decoders should follow once the push makes the newline decision J-readable. They don't." },
      { head: "So the null is about content type", body: "positional bookkeeping isn't something these decoders report, in any phrasing they use, including newline-continuation phrasing." },
      { head: "The test that settles it", body: "a next-digit cell in a numbered list ('next digit 5 vs 7'): maximally J. Tracked → col null is about location; not → content type." },
    ],
    caveat: "Main run: the raw-direction cell with newline-continuation candidates becomes primary (separating the count from the decision), with OL continuation-style candidates and a class-mean col direction.",
    notes:
      "Here's the sharper reading Fable pushed us to in round 5. The raw col direction, the one that still contains its J-space component, moves the J-Lens's newline readout strongly: AUROC 0.73 to 0.84. So under that push, the downstream consequence, 'a line break is coming', is lens-readable. Yet neither decoder follows it: the NLA stays at or below 0.54 across all three template families, including the newline-decision templates, and the Oracle Lens at or below 0.50. " +
      "If the story were purely about location, inside versus outside the J-space, the decoders should follow once the consequence is in J. They don't. So the null looks like a matter of content type: positional bookkeeping is just not something these decoders talk about, in any phrasing we've tried, including newline-continuation phrasing. " +
      "The clean test is a new cheap cell: the next digit in a numbered list, 5 versus 7. That's maximally in-J and should pass the causal gate. If the readers track it, the col null is about location; if not, Part B is explicitly scoped to content type.",
  },
  {
    type: "table",
    section: "Part B",
    kicker: "Part B · power",
    title: "For col, more items won't help: its effect is about zero, while workspace features clear the same band by 10–30 SD",
    pilot: true,
    columns: ["NLA, q90, J-complement band", "Real shift", "Null SD (direction ⊕ item)", "z at n = 60", "z at n → ∞"],
    colW: [3.6, 1.9, 2.9, 1.8, 1.9],
    rows: [
      ["col 'long line', 4r*", "+0.051", "0.141", "+0.56", "+0.61"],
      ["col 'about 60 characters', 4r*", "+0.118", "0.109", "+0.99", "+1.41"],
      ["col 'line break coming', 4r*", "+0.050", "0.079", "+0.60", "+0.62"],
      ["language, raw (calibration)", "d_z +1.5", "band SD 0.013 (AUROC)", "≈ +30", "—"],
      ["tense, raw (calibration)", "d_z +1.3", "band SD 0.021 (AUROC)", "≈ +18", "—"],
    ],
    footnote: "Random directions genuinely move the decoders, and that spread doesn't shrink with n. NLA − OL ΔAUROC on col is +0.015 to +0.028, below the pre-registered H3 bar of 0.05: H3 will likely fail for col, stated at the freeze.",
    notes:
      "A power point the audience will ask about. For col, the binding constraint is the null band, not the number of items: random directions genuinely move the decoders' log-odds, and that direction-to-direction spread doesn't shrink as you add items. Even with infinite items, the NLA's z on the leak-free col cells would be +0.6 to +1.4. " +
      "But the calibration shows this isn't a generic power failure. The same band is narrow relative to in-J effects: per-item effect sizes of +1.3 to +1.5 for language and tense, z around +18 to +30, against roughly minus 0.2 to 0.3 for col. The band limits col only because col's effect is about zero. " +
      "And one honest consequence: the NLA-minus-Oracle-Lens difference on col is +0.015 to +0.028, below our pre-registered 0.05 bar, so the H3 contrast will likely fail for col. We'll state that at the freeze rather than discover it later.",
  },
  {
    type: "chart",
    section: "Part B",
    kicker: "Part B · natural items (no steering)",
    title: "On natural text the readers describe in-J features well but barely discriminate col",
    chart: {
      kind: "bar",
      categories: ["col (count)", "col (line break)", "list index 'Item N'", "tense", "language"],
      series: [
        { name: "NLA", values: [0.6, 0.54, 0.84, 0.85, 1.0], color: "nla" },
        { name: "Oracle Lens", values: [0.55, 0.55, 0.99, 0.92, 1.0], color: "ol" },
      ],
      lines: [{ name: "chance (0.5)", value: 0.5, color: "muted" }],
      min: 0.3, max: 1, fmt: "0.00", labels: true, legend: "b",
      valTitle: "natural-item AUROC (mean of 3 frozen templates)",
    },
    chartW: 7.4,
    side: {
      bullets: [
        "Readers name a visible list index well (NLA .84, OL .99), but numbers never enter Qwen's J-space, so this is likely text re-description, not J-space reading.",
        "0 of 320 NLA and 0 of 960 OL mining readouts ever mention line length, column or wrapping.",
        "A probe correlates 0.88 with col on the same items; NLA / OL |ρ| ≤ 0.16.",
        "Open check: the NLA's natural 0.60 needs a text-only decoder on the same items (long vs short lines are visible without counting).",
      ],
    },
    notes:
      "The natural arm, without steering. Templates were mined from each reader's own outputs and frozen before any steering run. On natural items, both readers are excellent on language, good on tense, and on a visible list index, 'Item 7:', the NLA gets 0.84 and the Oracle Lens 0.99. On col they barely discriminate: 0.60 at best for the NLA, with an interval touching 0.5. " +
      "I originally hoped the list index would answer 'maybe readers just can't talk about counts'. But experiment 04 showed numbers never enter Qwen's J-space, so the list-index success is probably text re-description: the numeral is right there in the prompt. The count-verbalisation gap stays open. " +
      "Base rate: across 1,280 mining readouts no reader ever mentions line length. And a simple probe correlates 0.88 with col on the same items, so the information is in the activation; the readers just don't surface it.",
  },
  {
    type: "anchor",
    section: "Part B",
    stage: 3,
    title: "What we know so far (3 / 4): the decoders read workspace features, not col",
    next: "Next → Part C: so what does the NLA read beyond the workspace?",
    notes:
      "Third checkpoint. At doses where both decoders demonstrably read workspace features, neither reads col; for the NLA this null is secure, for the Oracle Lens less so because it only passes calibration on language and is template-fragile. And because the raw col direction makes the J-Lens's newline readout move while the decoders stay flat, the null looks like a matter of content type rather than location. All pilot-scale.",
  },
];
