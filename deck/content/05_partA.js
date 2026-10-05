// Part A: what counts as non-J? (experiments 02 and 04)

const FEATS = ["col", "col_report", "bracket depth", "pred. POS", "tense", "language", "topic"];

module.exports = [
  {
    type: "section",
    section: "Part A",
    num: "04",
    title: "Part A: what even counts as non-workspace?",
    sub: "The standard test is uninformative; a calibrated gate leaves one clean feature",
    notes: "Part A started as plumbing, finding features to steer, and became a co-headline: the standard way of certifying 'non-workspace' content doesn't work.",
  },
  {
    type: "chart",
    section: "Part A",
    kicker: "Part A · the standard test",
    title: "The standard test, 'does a probe survive removing the J-space?', passes every feature, topic included",
    chart: {
      kind: "bar",
      categories: FEATS,
      series: [
        { name: "remove top-384 J-space", values: [0.005, 0.001, 0.0, 0.005, 0.005, 0.003, -0.009], color: "jl" },
        { name: "remove top-1,536 J-space", values: [0.016, 0.003, 0.0, -0.013, -0.013, 0.0, -0.019], color: "jlPale" },
      ],
      min: -0.1, max: 0.1, fmt: "0.000", labels: false, legend: "b",
      valTitle: "retention after J-removal minus retention after random removal",
      title: "Every feature keeps ≥ 98% of probe performance; J-removal ≈ random removal (±0.02)",
    },
    side: {
      stats: [
        { value: "0.849 → 0.834", label: "topic probe accuracy after removing the top-1,536 J-space (language: 0.966 → 0.966)", color: "ink", size: 28 },
        { value: "0.900 → 0.880", label: "topic under the J-space paper's own recipe (25 J atoms projected out); tense .853 → .848, POS .908 → .904", color: "ink", size: 28 },
      ],
      note: "Why: strong features are redundantly encoded. Removing any 384 or 1,536 of 5,120 dimensions, random ones included, leaves them decodable.",
    },
    notes:
      "The standard evidence for 'outside the J-space' is: train a probe, project out the J-space, and show the probe still works. The J-space paper's evidence for POS and tense is of this kind. " +
      "We ran it on seven features, including topic and language, which are obviously in the J-space. Every feature keeps at least 98% of its probe performance, at both ranks. The chart shows retention after J-removal minus retention after removing a random subspace of the same rank: all within ±0.02 of zero. Topic goes from 0.849 to 0.834. Under the paper's own recipe, projecting out 25 J atoms from the probe, topic goes 0.900 to 0.880, as 'non-J' as tense or POS. " +
      "The reason is redundancy: a strong feature is linearly decodable from almost any large subspace of a 5,120-dimensional vector. " +
      "Framing matters here. This is a property of the test, not a claim that their conclusion is wrong. There are three differences from the paper: the model (Claude vs Qwen), the lens, and 25 atoms vs rank 384 or 1,536. Our claim is 'the test as run can't support the conclusion', and we offer the random-subspace control as a drop-in.",
  },
  {
    type: "chart",
    section: "Part A",
    kicker: "Part A · a calibrated gate",
    title: "A calibrated gate does separate them: the count features are J-depleted, the rest J-enriched",
    chart: {
      kind: "bar",
      categories: FEATS,
      series: [
        { name: "enrichment at r* = 384", values: [0.67, 0.59, 0.57, 2.62, 5.6, 3.32, 5.98], color: "jl" },
        { name: "enrichment at 4r* = 1,536", values: [1.1, 0.86, 1.16, 1.56, 2.31, 1.73, 2.34], color: "jlPale" },
      ],
      lines: [{ name: "non-J threshold (1.5)", value: 1.5, color: "nla" }],
      min: 0, max: 6.5, fmt: "0.0", labels: true, legend: "b",
      valTitle: "diff-in-means energy in top-r J-space ÷ random (1.0 = random)",
    },
    side: {
      bullets: [
        [{ text: "Geometric: ", options: { bold: true } }, { text: "how concentrated the feature's direction is in the top-r J-space (≤ 1.5 at both ranks = non-J). Generic activation directions are 2.2× at r*." }],
        [{ text: "Functional: ", options: { bold: true } }, { text: "does the full J-Lens readout track the value, in the form the lens emits, beyond a shuffled-label null band?" }],
        [{ text: "J-Lens readout AUROC: ", options: { bold: true } }, { text: "col 0.500 [0.499, 0.501] · depth 0.5035 (just above) · tense 0.52 · language 0.62 · topic 0.83 · POS (token classes) 0.664." }],
      ],
    },
    notes:
      "So we built a symmetric, calibrated gate with Fable. Geometric criterion: take the feature's difference-in-means direction and ask how much of its energy lies in the top-r J-space, relative to a random direction. 1.0 is random, and for reference a generic activation direction is already 2.2 at r*. We call a feature non-J if it's at most 1.5 at both ranks. " +
      "Functional criterion: does the J-Lens readout actually track the feature's value, in the form the lens emits, compared to a shuffled-label null band? " +
      "Under this gate the picture separates cleanly. col, col_report and bracket depth are J-depleted. POS, tense, language and topic are J-enriched, tense at 5.6 and topic at 6.0. Bracket depth narrowly fails the functional test (0.5035 against a band top of 0.501), so it's labelled diffuse-but-J-readable and excluded. " +
      "POS initially looked unreadable because we searched for the word 'adjective'; the lens emits words, not labels. Classifying its top-10 tokens by part of speech shows it reads POS at 0.664. Fable predicted that.",
  },
  {
    type: "table",
    section: "Part A",
    kicker: "Part A · the gate table",
    title: "Only one feature is cleanly non-J, and the label survives a change of J-Lens",
    columns: ["Feature", "What it is", "Enrichment r* / 4r* (n1000 lens)", "(25-prompt lens)", "J-Lens readout", "Causal effect kept after J-removal", "Label"],
    colW: [1.45, 2.65, 1.75, 1.35, 1.65, 1.6, 1.68],
    rows: [
      ["col", "characters since the last newline", "0.67 / 1.10", "0.82 / 1.41", "0.500", "79%", { label: "NON-J", tone: "nonj" }],
      ["col_report", "same count, asked explicitly", "0.59 / 0.86", "0.50 / 0.93", "0.500", "91%", { label: "NON-J*", tone: "nonj" }],
      ["bracket depth", "nesting level of brackets", "0.57 / 1.16", "—", "0.5035", "41%", { label: "DIFFUSE", tone: "diffuse" }],
      ["pred. POS", "predicted next-word part of speech", "2.62 / 1.56", "1.41 / 1.61", "0.664 (token classes)", "33%", { label: "IN-J", tone: "inj" }],
      ["tense", "past vs present at verb slots", "5.60 / 2.31", "4.36 / 2.37", "0.52", "19%", { label: "IN-J", tone: "inj" }],
      ["language", "Russian vs Polish", "3.32 / 1.73", "2.74 / 1.83", "0.62", "5%", { label: "IN-J", tone: "inj" }],
      ["topic", "reference", "5.98 / 2.34", "—", "0.83", "—", { label: "IN-J", tone: "inj" }],
    ],
    footnote: "* col_report was meant to be the in-J count calibration; it stayed non-J (Qwen doesn't pull the count into J on request). r* = 384, 4r* = 1,536 at layer 42.",
    notes:
      "The full gate table at layer 42. col is non-J under both J-Lenses: the neuronpedia lens fit on 1,000 prompts, our primary, and the 25-prompt camilablank lens, which differs a lot (vector cosine 0.79, top-25 overlap only 19%). That robustness line matters for Part B. " +
      "col_report is the same count but asked as an explicit question. It was supposed to be our in-workspace count control, mirroring the paper's result that numbers enter the J-space when the model is asked to report them. On Qwen it stays out, which is itself a small model-difference finding, and it means we lost the planned calibration. " +
      "The last numeric column previews Part B: when we steer along each feature's direction with the top-1,536 J-space removed, how much of the full causal effect survives? col keeps 79%; tense keeps 19%; language 5%. For in-J features the causal route runs through the J-space. " +
      "Language here is Russian vs Polish, the direction used throughout experiment 02.",
  },
  {
    type: "chart",
    section: "Part A",
    kicker: "Part A · a survey of ~12 features (experiments 02 + 04)",
    title: "Survey: no natural non-count feature is outside J; the non-J content Qwen uses looks like positional bookkeeping",
    chart: {
      kind: "bar",
      categories: ["col (02)", "code language", "capitalisation", "in-quote state", "words since sentence start", "French gender agreement"],
      series: [{ name: "causal effect kept after removing the top-1,536 J-space", values: [0.79, 0.23, 0.21, 0.2, 0.18, 0.12] }],
      pointColors: ["probe", "muted", "muted", "muted", "muted", "muted"],
      min: 0, max: 1, fmt: "0%", labels: true, legend: false,
      title: "Share of the steering effect that survives J-removal",
    },
    chartW: 6.9,
    side: {
      rows: [
        { head: "Capitalisation · in-quote", body: "IN-J (enrichment 2.71 · 2.22)", tone: "inj" },
        { head: "Coding language", body: "DIFFUSE-BUT-J-READABLE; its steering leaks into J (z 2.4–5.8)", tone: "diffuse" },
        { head: "French gender", body: "fails decodability (AUROC 0.60)", tone: "fail" },
        { head: "Indentation", body: "non-J (1.11 / 1.14) but unused: steering doesn't move it", tone: "nonj" },
        { head: "Words since sentence start", body: "J-depleted (1.03 / 1.09) but J-Lens-readable", tone: "diffuse" },
        { head: "List index with numerals", body: "the in-J count control FAILED: digits never reach the J-Lens", tone: "fail" },
      ],
    },
    notes:
      "Part B rests on one feature, so we gated nine backups with the same procedure, 260 items each, about $80 of compute. None qualifies. " +
      "The pattern is the finding. The count and positional features, col, indentation, words since sentence start and bracket depth, are geometrically J-depleted. Every non-count feature we tried, capitalisation, coding language, quote state and gender agreement, is J-enriched or J-Lens-readable, and keeps only 12 to 23% of its causal effect after J-removal, against col's 79%. (The backup percentages were measured at layer 38 against the raw direction, col's at layer 42, so compare the pattern, not the decimals.) " +
      "So on Qwen at layer 42, across this survey of about a dozen features, natural non-workspace content looks like positional bookkeeping. That is a survey result, not a theorem: a different model, layer or feature family could break it. That means we can't defeat the objection 'col is special because it's a count' with a non-count feature. We state it as a scope limit. " +
      "Second result: the list index with numerals in context, our planned in-J count control, failed. Digits never reach the J-Lens top-10. Numbers just don't enter Qwen's J-space in these settings, consistent with col_report. So the count-verbalisation calibration gap stays open; I come back to it in limitations.",
  },
  {
    type: "anchor",
    section: "Part A",
    stage: 2,
    title: "What we know so far (2 / 4): one clean non-workspace feature, and it is a count",
    next: "Next → Part B: does any reader follow col when we push it inside the activation?",
    notes:
      "Third checkpoint. The usual test for non-workspace content is uninformative; the calibrated gate works; and it leaves exactly one clean feature, col. Everything non-J on this model at this layer looks like positional bookkeeping, and numbers don't enter the J-space at all. Now: does any reader read col?",
  },
];
