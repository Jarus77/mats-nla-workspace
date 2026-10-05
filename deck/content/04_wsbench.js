// Finding 1: WorkspaceBench success is explained by text (experiment 06).

module.exports = [
  {
    type: "section",
    section: "Part D",
    num: "07",
    title: "Part D: on WorkspaceBench, a text-only model matches the readers",
    sub: "Re-solving decomposition at layer 42 · 4 regex families · n = 400 items",
    notes: "Part D, from tool D: how much of each reader's benchmark success could a model that never sees the activation match? One consistency check on this part is still open, so these slides carry a flag.",
  },
  {
    type: "chart",
    section: "Part D",
    flag: "CONSISTENCY CHECK PENDING",
    kicker: "Part D · text-only baselines",
    title: "A text-only Qwen that never sees the activation scores as well as the NLA: 0.46 vs 0.47",
    chart: {
      kind: "bar",
      dir: "bar",
      categories: ["NLA @42", "Oracle Lens @42", "J-Lens @42", "Text-only Qwen (NLA style, k=1)", "Text-only Qwen (bench template)", "Text-only Gemini 3.8 Flash", "Re-solve from NLA readout only (Gemini)"],
      series: [{ name: "pooled pass rate (n = 400)", values: [0.47, 0.40, 0.02, 0.46, 0.76, 0.75, 0.60] }],
      pointColors: ["nla", "ol", "jl", "text", "text", "text", "textPale"],
      min: 0, max: 1, fmt: "0.00", labels: true, legend: false,
    },
    chartW: 7.9,
    side: {
      stats: [
        { value: "p = 0.75", label: "NLA vs text-only Qwen in the NLA's own style, sample-matched (McNemar); OL vs text: p = 0.09", color: "ink", size: 30 },
        { value: "187 vs 159", label: "same-layer hits, NLA vs Oracle Lens (p = 0.002)", color: "nla", size: 30 },
      ],
      bullets: ["The template matters a lot (typo: 0.45 vs 0.85 for the same model), so 'text explains it' is judged against a union of templates."],
    },
    notes:
      "Here's the core comparison, pooled over 400 items at layer 42. The NLA passes 47%. The Oracle Lens at the same layer 40%. The J-Lens 2%. " +
      "Now the ochre bars: models that never see the activation. Base Qwen, writing in the NLA's own style from the prompt alone, one sample, gets 46%. That's statistically indistinguishable from the NLA: McNemar p = 0.75. With the benchmark's own prompt-only template the same Qwen gets 76%, and Gemini 3.8 Flash gets 75%. " +
      "And the pale bar: give a text model only the NLA's readout, never the prompt, and ask it to re-solve. It scores 0.60 with Gemini (0.57 with Qwen), more than the NLA's own 0.47. The readouts carry more re-solvable content than they get credit for. " +
      "Open consistency check (e): readout-only re-solving (0.57 to 0.60) beats the sample-matched text arm (0.46) but stays below the prompt-based arms (0.75 to 0.76). Before we freeze 'fully explained by text' we check item by item whether readout-only re-solving ever succeeds where every prompt-based arm fails. " +
      "Two caveats: the template choice moves things a lot, so we judge 'text explains it' against a union of templates; and among the readers, the NLA does beat the Oracle Lens at the same layer, 187 hits vs 159.",
  },
  {
    type: "stats",
    section: "Part D",
    flag: "CONSISTENCY CHECK PENDING",
    kicker: "Part D · item level",
    title: "Nothing the readers get right is out of text's reach: 0 of 187 NLA hits are reader-only",
    stats: [
      { value: "0 / 187", label: "NLA hits that no text-only sample reaches (8 samples: 2 models × 2 templates × k)", color: "nla" },
      { value: "4 / 159", label: "Oracle Lens hits no text sample reaches; all 4 are Chinese code-switching on multilingual items (a scoring-form artefact)", color: "ol" },
      { value: "0.57–0.60", label: "pass rate when a text model re-solves from the NLA's readout alone, above the NLA's own 0.47", color: "text" },
    ],
    families: [
      { head: "typo", body: "the only family where readers beat text at k = 1 (NLA 73 vs 45), a register/length effect: the text arm quotes the misspelling without correcting it" },
      { head: "multihop", body: "text crushes the readers: NLA 15 vs text-only Qwen 44 vs Gemini 75" },
      { head: "multilingual / basic", body: "NLA ≈ text-only Qwen (p = 0.87 / 0.63)" },
    ],
    caveat: "This does not show the readers don't read. The layer-42 activation encodes the prompt, so reading-then-re-solving looks identical here. It shows WorkspaceBench cannot separate the two, which is why the causal test matters.",
    notes:
      "Item by item: of the NLA's 187 correct items, every single one is also reached by at least one of 8 text-only samples. Zero reader-only hits. For the Oracle Lens it's 4 of 159, and all four are multilingual items where OL answered in Chinese, matching the item's Chinese form, while the text arms named the concept in a form the regex doesn't credit. That's a scoring artefact, not latent content. " +
      "Fair-comparison caveat: the union of 8 text samples is best-of-8; the sample-matched comparison is the 0.46 vs 0.47 from the previous slide. " +
      "Per family: typo is the only family where readers beat text at k=1, and that's a register effect. On multihop, where hidden intermediates should matter most, text crushes the readers. " +
      "And the crucial framing: this is not evidence that readers don't read. The activation encodes the prompt; a reader that reads and a reader that re-solves look identical here. What it shows is that this benchmark can't tell them apart. That motivates the causal test.",
  },
  {
    type: "chart",
    section: "Part D",
    flag: "CONSISTENCY CHECK PENDING",
    kicker: "Part D · claim precision",
    title: "Readers' claims are text-true less often than a text-only model's: precision 0.62 and 0.52 vs 0.92",
    chart: {
      kind: "stacked",
      dir: "bar",
      categories: ["NLA @42", "Oracle Lens @42", "Text-only Qwen"],
      series: [
        { name: "contradicted", values: [0.36, 0.35, 0.08], color: "7F96A3" },
        { name: "restates text", values: [0.09, 0.06, 0.23], color: "A4B6C0" },
        { name: "matches gold target", values: [0.11, 0.18, 0.18], color: "C3D1D8" },
        { name: "inferable from text", values: [0.38, 0.14, 0.49], color: "DCE4E8" },
        { name: "unsupported", values: [0.06, 0.28, 0.02], color: "F0F4F6" },
      ],
      min: 0, max: 1, fmt: "0%", labels: true, legend: "b", labelColor: "15222C",
    },
    chartW: 7.6,
    side: {
      stats: [
        { value: "0.62 · 0.52 · 0.92", label: "text-verifiable precision (NLA · OL · text-only), 100 readouts each", color: "ink", size: 28 },
      ],
      bullets: [
        "NLA errors are mostly misquotes of the prompt (246 of 301 contradicted claims look like quotes).",
        "OL errors are wrong entities and facts; 28% of its claims are unsupported specifics.",
        "Judge: Gemini 3.8 Flash; second judge kimi-k3 agrees at κ 0.72 (5-way) / 0.77 (3-way) on 40 claims. Human labels still to do.",
      ],
    },
    notes:
      "Claim-level view. A judge splits each readout into atomic claims and labels each against the text and the gold answer. Text-verifiable precision is the share of checkable claims that are true; 'unsupported' is not counted as false. " +
      "NLA 0.62, Oracle Lens 0.52, and the text-only Qwen 0.92. The error profiles differ: the NLA's contradictions are mostly near-miss quotations of the prompt, consistent with a reader that reconstructs the prompt imperfectly. The Oracle Lens's are wrong entities and facts, plus 28% unsupported specifics. " +
      "And look at the 'matches gold target' slice: 11% NLA, 18% OL, and 18% for the text-only model. A model with no activation makes as many gold-consistent claims as either reader. " +
      "Judge validity: a second judge from a different family agrees at kappa 0.72 to 0.77 on 40 claims. The pre-registered human-labelled subset is still to do, so treat these as judge-based.",
  },
  {
    type: "chart",
    section: "Part D",
    kicker: "Part D · side finding",
    title: "Side finding: the benchmark's frozen prompt-only floors appear about 0.14 too low",
    chart: {
      kind: "bar",
      categories: ["typo", "multilingual", "multihop", "basic readout"],
      series: [
        { name: "frozen published floor", values: [0.67, 0.42, 0.69, 0.59], color: "textPale" },
        { name: "our re-run (same template, thinking off)", values: [0.83, 0.47, 0.84, 0.80], color: "text" },
      ],
      min: 0, max: 1, fmt: "0.00", labels: true, legend: "b", valTitle: "prompt-only pass rate",
    },
    side: {
      bullets: [
        "Significant on 3 of 4 families (z ≈ +2.2 to +3.3); an earlier independent draw shows the same pattern.",
        "Cause inconclusive: no rendering reproduces the frozen values; the frozen run has empty cells; the generator isn't public.",
        "Implication: published reader-vs-floor gaps flatter the readers. Our conclusions use our own text arms, not the frozen floor.",
        "Plan (D21): file a factual issue with the re-run table first, then one paragraph in the write-up. Outward-facing, so it needs Suraj's approval.",
      ],
    },
    notes:
      "A side finding we were careful with. WorkspaceBench publishes a 'prompt-only floor': stock Qwen sees only the text. Re-running its exact template and sampling under the thinking-off rendering every reader uses, we get 0.83, 0.47, 0.84 and 0.80, against frozen values of 0.67, 0.42, 0.69 and 0.59. Significant on three of four families, and an earlier independent draw agreed. " +
      "We could not find the cause: no rendering we tried reproduces the frozen numbers exactly, the frozen run has unexplained empty cells, and the generator code isn't public. So we say 'appear too low', treat the frozen floors as a lower bound, and none of our conclusions depend on them. " +
      "If confirmed, the published gaps between readers and the floor flatter the readers. Contacting the authors is an outward-facing decision we've left for after the sprint.",
  },
];
