// Gates: are the instruments sound? Chart specs are rendered natively by lib/charts.js.
// Colour keys: nla, ol, jl, text, probe, muted, line, nlaPale, olPale, jlPale, textPale, or a hex string.

const LAYERS = ["20", "24", "28", "32", "36", "40", "42", "44", "48", "52", "56", "60"];

module.exports = [
  {
    type: "chart",
    section: "Gates",
    kicker: "Gate K0 · NLA layer convention",
    title: "Gate K0: the NLA reads block-42 output, with 1.7% error against 20% for the neighbouring layer",
    chart: {
      kind: "bar",
      categories: ["block 40", "block 41", "block 42", "block 43"],
      series: [{ name: "relative L2 error (mean, 64 examples)", values: [0.304, 0.222, 0.026, 0.232] }],
      pointColors: ["muted", "muted", "nla", "muted"],
      min: 0, max: 0.35, fmt: "0.000", labels: true, legend: false,
      title: "Our recomputed activation vs the NLA's own examples (lower = match)",
    },
    side: {
      stats: [
        { value: "0.9999", label: "median cosine at block 42 (block 41: 0.979); best match on 64/64 examples", color: "nla" },
        { value: "0.761", label: "NLA FVE reproduced (model card 0.766; iter 300: 0.759 vs 0.756)", color: "nla" },
        { value: "0.760 vs 0.695", label: "functional test: FVE on block-42 vs block-41 captures", color: "ink", size: 26 },
      ],
      note: "Correction logged: an early handoff called the layer 'ambiguous, 0.974 vs 0.95'. The 0.974 was the wrong convention.",
    },
    notes:
      "Kill check K0: which layer was the NLA trained on? An off-by-one would quietly corrupt every NLA number. 'Layer 42' could mean the output of block 41 or of block 42. " +
      "We recomputed activations for the NLA's 64 shipped example texts and compared. Block-42 output matches with median cosine 0.9999 and mean relative L2 error 0.026 (median 0.017); block 41 is at 0.22 error and cosine 0.974. Cosine alone is a weak discriminator because adjacent layers are all about 0.97 similar, so relative L2 is the clean signal, and block 42 wins on all 64 examples. " +
      "The functional test agrees: the NLA's FVE is 0.760 on block-42 captures and 0.695 on block-41 captures. And we reproduced its headline FVE, 0.761 against the card's 0.766. " +
      "One honesty note: in an early handoff to Fable I reported the layer as ambiguous, 0.974 vs 0.95, from a partial agent output. That 0.974 was the wrong convention, and we sent the correction.",
  },
  {
    type: "chart",
    section: "Gates",
    kicker: "Gate · WorkspaceBench replication",
    title: "We reproduce all 12 published WorkspaceBench numbers inside our 95% confidence intervals",
    chart: {
      kind: "bar",
      categories: ["typo", "multilingual", "multihop", "basic readout"],
      series: [
        { name: "Oracle Lens (ours)", values: [0.86, 0.65, 0.34, 0.73], color: "ol" },
        { name: "Oracle Lens (published)", values: [0.83, 0.62, 0.32, 0.71], color: "olPale" },
        { name: "NLA (ours)", values: [0.73, 0.42, 0.15, 0.57], color: "nla" },
        { name: "NLA (published)", values: [0.75, 0.44, 0.22, 0.53], color: "nlaPale" },
        { name: "J-Lens (ours)", values: [0.15, 0.08, 0.10, 0.24], color: "jl" },
        { name: "J-Lens (published)", values: [0.17, 0.06, 0.08, 0.24], color: "jlPale" },
      ],
      min: 0, max: 1, fmt: "0.00", labels: false, legend: "b", valTitle: "pass rate (regex families, n = 100)",
    },
    side: {
      stats: [
        { value: "12 / 12", label: "published values inside our 95% bootstrap CIs (largest gap: NLA multihop 0.15 vs 0.22, CI edge)", color: "ink" },
        { value: "$2.80", label: "OpenRouter + ~1.5 GPU-h on Modal for the full replication", color: "ink", size: 30 },
      ],
      bullets: ["Protocol as published: Oracle Lens and J-Lens best-of-11 layers; NLA at layer 42 only."],
    },
    notes:
      "Second gate: reproduce the published benchmark. WorkspaceBench is the September 2026 benchmark for exactly these readers on Qwen3.6-27B. We ran the four regex-scored multi-token families, 100 items each, with the bench's own code, for all three readers. " +
      "All 12 published numbers fall inside our 95% bootstrap intervals. The largest gap is the NLA on multihop, 0.15 against 0.22, at the edge of our interval, consistent with k=1 sampling noise. Directions are mixed, so no systematic setup difference. " +
      "Solid bars are ours, pale bars published. Note the protocol: the Oracle Lens and J-Lens are scored best-of-11 layers, the NLA at one layer. Hold that thought. " +
      "Cost: $2.80 of API calls for the J-Lens summariser and about 1.5 GPU-hours.",
  },
  {
    type: "chart",
    section: "Gates",
    kicker: "Gate · a protocol asymmetry",
    title: "The Oracle Lens's published lead comes from scoring the best of 11 layers; at layer 42 the NLA ties or wins",
    charts: [
      {
        kind: "line",
        categories: LAYERS,
        series: [
          { name: "OL, per layer", values: [0.02, 0.03, 0.09, 0.10, 0.18, 0.26, 0.28, 0.26, 0.30, 0.39, 0.43, 0.38], color: "ol" },
          { name: "OL, any-of-11 (published rule)", values: Array(12).fill(0.65), color: "olPale", dash: "dash" },
          { name: "NLA, layer 42 only", values: Array(12).fill(0.42), color: "nla", dash: "dash" },
        ],
        min: 0, max: 0.8, fmt: "0.00", legend: "b", title: "multilingual", catTitle: "Oracle Lens layer",
      },
      {
        kind: "line",
        categories: LAYERS,
        series: [
          { name: "OL, per layer", values: [0.22, 0.23, 0.27, 0.30, 0.43, 0.42, 0.45, 0.48, 0.46, 0.58, 0.62, 0.61], color: "ol" },
          { name: "OL, any-of-11 (published rule)", values: Array(12).fill(0.73), color: "olPale", dash: "dash" },
          { name: "NLA, layer 42 only", values: Array(12).fill(0.57), color: "nla", dash: "dash" },
        ],
        min: 0, max: 0.8, fmt: "0.00", legend: "b", title: "basic readout", catTitle: "Oracle Lens layer",
      },
    ],
    strip: [
      "Published rule = per-item any-layer: an item passes if any of 11 layers hits. That is layer selection plus a best-of-11-samples bonus.",
      "At OL's best single layer (mostly L56) it slightly exceeds the NLA, within noise: typo .78 vs .73, multilingual .43 vs .42, basic .62 vs .57.",
      "Decision (D7): layer 42 is the primary layer for every reader; OL at 40/44 is sensitivity.",
    ],
    notes:
      "While replicating we found something worth a line in the write-up. The published rule for the multi-layer readers is per-item any-layer: an item counts as a pass if any of 11 layers gets it. Applying that rule reproduces the published values exactly. But it's both layer selection and, since each layer is an independent sample, a best-of-11 sampling bonus. " +
      "The violet curve is the Oracle Lens per layer. At layer 42 it scores 0.28 on multilingual against the NLA's 0.42, and 0.45 against 0.57 on basic readout. Its peak is late, around layer 56, at 0.43 and 0.62, which only slightly exceeds the NLA, within noise. The any-of-11 rule lifts it to 0.65 and 0.73. " +
      "So this is a protocol asymmetry, not an error: OL's layer flexibility is a real capability. But a matched comparison should report both. Note also that layer 42 is a weak layer for the Oracle Lens on these families, so the NLA's same-layer tie is less impressive than it looks. " +
      "Decision D7: layer 42 is primary for all three readers.",
  },
  {
    type: "anchor",
    section: "Gates",
    stage: 1,
    title: "What we know so far (1 / 4): the instruments work",
    next: "Next → Part A: what even counts as non-workspace content?",
    notes:
      "First checkpoint. We know the NLA's layer convention, we reproduce its reconstruction quality, and we reproduce the benchmark for all three readers. And we've learned that at a matched layer the NLA is at least as good as the Oracle Lens. Nothing yet about where the NLA's content comes from.",
  },
];
