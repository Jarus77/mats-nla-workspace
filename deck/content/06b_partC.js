// Part C: what the NLA does read beyond the workspace (experiments 05 and 08-calibration).

module.exports = [
  {
    type: "section",
    section: "Part C",
    num: "03",
    title: "Part C: what the decoders do read beyond the workspace",
    sub: "The tail of verbalisable features, and a story generated from the latent",
    notes: "Part C. If the NLA doesn't read col, what is its extra? Two pieces of evidence: a tense component outside the top-1,536 J-space, and how it describes planted activations.",
  },
  {
    type: "chart",
    section: "Part C",
    kicker: "Part C · the tense tail (a main-run primary)",
    title: "Both decoders read tense with the top-1,536 J-space removed; with mined templates OL ≥ NLA",
    pilot: true,
    chart: {
      kind: "bar",
      categories: ["NLA", "Oracle Lens", "J-Lens (label words)"],
      series: [{ name: "AUROC, J-removed (4r*) tense direction, q90, layer 42, fixed templates", values: [0.6, 0.51, 0.5] }],
      pointColors: ["nla", "ol", "jl"],
      lines: [{ name: "chance (0.5)", value: 0.5, color: "muted" }],
      min: 0.3, max: 0.8, fmt: "0.00", labels: true, legend: false,
      title: "Fixed templates (K3 pilot): only the NLA moves",
    },
    chartW: 6.6,
    side: {
      stats: [
        { value: "0.66–0.79", label: "Oracle Lens with mined (continuation) templates, z +12 to +17", color: "ol", size: 32 },
        { value: "0.59–0.74", label: "NLA with mined templates. The earlier 'NLA > OL on tense 4r*' does not survive", color: "nla", size: 32 },
      ],
      note: "Leak-free at layer 42 (J-Lens label words and a 66-verb content set don't move). A geometric claim: tense is self-reported at 87%, so this is the tail of a verbalisable feature, not 'what the model can't say'. OL adapter-off check for tense still to run.",
    },
    notes:
      "What do the decoders read beyond the workspace? Tense, with the top-1,536 J-space projected out. With the fixed calibration templates only the NLA moved, 0.60 at z +8.7, and we briefly read that as an NLA-over-Oracle-Lens contrast. With mined templates that contrast does not survive: the Oracle Lens's continuation-style tense templates reach 0.66 to 0.79, z +12 to +17, at or above the NLA's 0.59 to 0.74. Same template lesson as col. " +
      "It is leak-free at layer 42 under both label and content sets. It's a geometric claim: tense is a feature the model can verbalise, self-reported 87% of the time, so this is the decoders reading the distributed tail of a verbalisable feature beyond the top-r J-space. Both tense cells are main-run primaries. The adapter-off check for the Oracle Lens on tense, which would tell us whether this is also re-instatement, hasn't been run yet.",
  },
  {
    type: "chart",
    section: "Part C",
    kicker: "Part C · planted intermediates = in-J calibration",
    title: "In-J calibration: when we plant a different bridge entity, all three readers report the planted one",
    chart: {
      kind: "bar",
      categories: ["unplanted", "planted (prompt unchanged)"],
      series: [
        { name: "Oracle Lens @42", values: [0.01, 1.0], color: "ol" },
        { name: "NLA @42", values: [0.03, 0.98], color: "nla" },
        { name: "J-Lens @42", values: [0.01, 1.0], color: "jl" },
      ],
      min: 0, max: 1.1, fmt: "0.00", labels: true, legend: "b",
      valTitle: "P(reader names the planted bridge Y), 200 items",
    },
    chartW: 6.6,
    side: {
      stats: [
        { value: "431 / 614", label: "items kept at layer 38: the model's answer flips, unrelated prompts barely change (KL ≤ 0.1), the prompt is still decodable", color: "ink", size: 30 },
        { value: "z ≥ +84", label: "against random plants of the same size (OL +166, NLA +112, J-Lens +168 vs the isotropic band)", color: "ink", size: 30 },
      ],
      note: "This is in-J calibration, not non-J evidence: the swap direction is 6.5× J-enriched, and by layer 42 the model already encodes Y's answer.",
    },
    notes:
      "Planted intermediates, which after Fable round 5 we relabel as in-J calibration rather than a re-solving test. Prompts like 'The capital of the country famous for the Eiffel Tower is'. At layer 38 we swap the bridge's class-mean projection from France to a partner country, here Sweden, keeping the prompt identical. We keep an item only if the model's own answer flips, unrelated prompts barely change, and the original prompt is still decodable at layer 42. 431 of 614 items pass at layer 38, so kill check K4 passes. " +
      "All three readers report the planted bridge: Oracle Lens 1.00, NLA 0.98, J-Lens 1.00, up from 1 to 3% unplanted, more than 80 SD beyond random plants. " +
      "That's the reading pattern, but it's calibration, not non-workspace evidence: the swap direction is strongly J-enriched (6.5×) and the J-Lens reads it too. And by layer 42 the model already encodes Sweden's answer, so a reader could back-infer the bridge from the answer.",
  },
  {
    type: "quote",
    section: "Part C",
    kicker: "Part C · the NLA generates",
    title: "The NLA doesn't re-solve the text: it rewrites the clue to fit the planted bridge",
    prompt: "Fact: The capital of the country famous for the Eiffel Tower is",
    planted: "Bridge planted inside the activation: Sweden (the text still says Eiffel Tower)",
    output: "\"…the country whose CEO designed adopted Volvo… Sweden\"",
    outputSource: "NLA free-form explanation of the planted activation (12 of 12 such explanations name the planted country and none the original; other rewrites include \"…maple leaves (Canada)\")",
    lessons: [
      { head: "What it means", body: "The NLA's 'prompt reconstruction' is generated from the latent. That rules out 'decode the text, then compute' as its mechanism: its language model writes a story consistent with the latent." },
      { head: "Dose caveat", body: "Answer flips needed ‖Δ‖/‖h‖ ≈ 0.49 (q10–q90 0.33–0.71), 1.5–3× the natural France–Sweden distance. The main run makes α ≤ 1.5 (109 items) primary." },
    ],
    notes:
      "My favourite qualitative result. In free-form explanations of planted activations, the NLA names the planted country in 12 of 12 cases. But it doesn't say 'a question about the Eiffel Tower, answer Sweden'. It rewrites the clue: 'the country whose CEO designed adopted Volvo… Sweden', or 'maple leaves (Canada)'. " +
      "So the NLA's apparent prompt reconstruction is itself generated from the latent. It confabulates a prompt consistent with what the activation now encodes. So whatever branch the col result lands on, the NLA is not reading the text and then solving: its language model generates a consistent story from the latent. That finding stands either way. " +
      "Dose caveat: to flip the answer we needed large pushes, median 49% of the activation norm, 1.5 to 3 times the natural distance between countries; the main run makes the 109 items at alpha ≤ 1.5 primary. Twelve free-form samples is small; the main run scales this up.",
  },
];
