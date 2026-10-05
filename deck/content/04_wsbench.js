// Part D: WorkspaceBench target information is recoverable from the prompt text (experiment 06, D14, D22).

module.exports = [
  {
    type: "section",
    section: "Part D",
    num: "04",
    title: "Part D: WorkspaceBench targets are recoverable from the prompt",
    sub: "Re-solving decomposition at layer 42 · 4 regex families · n = 400 items",
    notes: "Part D, from tool D: is the target information in the readers' readouts something a text model could get from the prompt alone? A consistency check (e) raised during the Fable round is now resolved, and the wording here is the defensible version from D22.",
  },
  {
    type: "chart",
    section: "Part D",
    kicker: "Part D · text-only baselines",
    title: "A text-only Qwen that never sees the activation ties the NLA at k = 1: 0.46 vs 0.47",
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
      bullets: ["Caveat: the 0.46 arm used the NLA's own instruction, which never asks for the answer. Same model, bench template: 0.76. The fair same-template comparison is on the next slide."],
    },
    notes:
      "Here's the core comparison, pooled over 400 items at layer 42. The NLA passes 47%. The Oracle Lens at the same layer 40%. The J-Lens 2%. " +
      "Now the ochre bars: models that never see the activation. Base Qwen, writing in the NLA's own style from the prompt alone, one sample, gets 46%. That's statistically indistinguishable from the NLA: McNemar p = 0.75. With the benchmark's own prompt-only template the same Qwen gets 76%, and Gemini 3.8 Flash gets 75%. " +
      "And the pale bar: give a text model only the NLA's readout, never the prompt, and ask it to re-solve. It scores 0.60 with Gemini (0.57 with Qwen), more than the NLA's own 0.47. The readouts carry more re-solvable content than they get credit for. " +
      "An important caveat on that tie: the 0.46 arm used the NLA's own instruction, which never asks for the answer, so it understates what the text makes available; the same Qwen with the bench template scores 0.76. And among the readers, the NLA does beat the Oracle Lens at the same layer, 187 hits vs 159.",
  },
  {
    type: "chart",
    section: "Part D",
    kicker: "Part D · same-template re-solver comparison (check e, resolved)",
    title: "Given the prompt, a text model names the target more often than given the readout, and the readout adds nothing",
    chart: {
      kind: "bar",
      categories: ["Prompt only", "NLA readout only", "OL readout only", "Prompt + NLA readout", "Prompt + OL readout"],
      series: [{ name: "pooled pass rate, Gemini re-solver, same template, k = 1 (n = 400)", values: [0.89, 0.6, 0.56, 0.88, 0.88] }],
      pointColors: ["text", "nla", "ol", "textPale", "textPale"],
      min: 0, max: 1, fmt: "0.00", labels: true, legend: false,
      title: "Same re-solver, same instruction; only the input changes",
    },
    chartW: 7.2,
    side: {
      stats: [
        { value: "0.88 vs 0.89", label: "adding the readout to the prompt doesn't help (p = 0.38 NLA / 0.47 OL)", color: "ink", size: 30 },
        { value: "4 · 2 of 400", label: "readout-only hits every prompt arm misses (NLA · OL), all regex-form artefacts", color: "ink", size: 30 },
        { value: "0 / 187", label: "NLA hits that no prompt-only text sample reaches (OL: 4 of 159, Chinese code-switching)", color: "nla", size: 30 },
      ],
    },
    notes:
      "This resolves the consistency check Fable raised. Earlier, a re-solver reading only the NLA's readout scored 0.57 to 0.60, above the 0.46 text arm, which looked like the readouts carried something the text didn't. It was an instruction mismatch: the 0.46 arm used the NLA's own instruction, which never asks for the answer, while the re-solver instruction asks for the entity and its language, which is exactly what the regex rewards. " +
      "So we ran the fair comparison: the same Gemini re-solver, same template, one sample, with only the input changing. Given the prompt alone it names the target 89% of the time. Given only the NLA's readout, 60%; the Oracle Lens's readout, 56%. Given prompt plus readout, 88%: adding the readout doesn't help (p = 0.38 and 0.47). The residual readout-only hits, 4 for the NLA and 2 for the Oracle Lens out of 400, are all regex-form artefacts: uncredited short forms, the Oracle Lens's Chinese code-switching copied through, one name-order variant. " +
      "The defensible sentence, from D22: on WorkspaceBench's four regex multi-token families at layer 42, the target information in NLA and Oracle Lens readouts is recoverable from the prompt text alone. " +
      "Claims to avoid: 'the readers re-solve from text', because this data can't show the mechanism; and 'readouts contain nothing beyond text', because only these families' targets were tested. Per family, text crushes the readers on multihop (NLA 15 vs text-only Qwen 44 vs Gemini 75); typo is the only family where readers beat the NLA-style text arm, a register effect.",
  },
  {
    type: "chart",
    section: "Part D",
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
