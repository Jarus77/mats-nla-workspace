// Part C: what the NLA does read beyond the workspace (experiments 05 and 08-calibration).

module.exports = [
  {
    type: "section",
    section: "Part C",
    num: "03",
    title: "Part C: what the NLA does read beyond the workspace",
    sub: "The distributed tail of verbalisable features, and a story generated from the latent",
    notes: "Part C. If the NLA doesn't read col, what is its extra? Two pieces of evidence: a tense component outside the top-1,536 J-space, and how it describes planted activations.",
  },
  {
    type: "chart",
    section: "Part C",
    kicker: "Part C · the tense tail (pre-registered secondary)",
    title: "The NLA reads tense even with the top-1,536 J-space removed, leak-free; the Oracle Lens doesn't",
    pilot: true,
    chart: {
      kind: "bar",
      categories: ["NLA", "Oracle Lens", "J-Lens (label words)"],
      series: [{ name: "AUROC, J-removed (4r*) tense direction, q90, layer 42", values: [0.6, 0.51, 0.5] }],
      pointColors: ["nla", "ol", "jl"],
      lines: [{ name: "chance (0.5)", value: 0.5, color: "muted" }],
      min: 0.3, max: 0.8, fmt: "0.00", labels: true, legend: false,
      title: "Tense direction with the top-1,536 J-space projected out",
    },
    chartW: 6.6,
    side: {
      stats: [
        { value: "z +8.7", label: "NLA vs the J-complement band (+5.7 isotropic); paired win 0.90; Oracle Lens inside the band (z +1.5 / +1.7)", color: "nla", size: 32 },
        { value: "leak-free", label: "J-Lens label words and a 66-verb content set don't move at layers 42 / 40 (z −0.3 / +0.1)", color: "jl", size: 32 },
      ],
      note: "Provenance: found in the K3 calibration pilot (60 items, excluded), then pre-registered as a secondary main-run cell. A geometric claim, not 'reads what the model can't say': tense is self-reported at 87%. The language version is void (Cyrillic content leak, z +3.7).",
    },
    notes:
      "Here's what the NLA does read beyond the workspace. In the calibration pilot we also steered tense along its direction with the top-1,536 J-space projected out. The NLA still follows it: 0.60 AUROC, 8.7 SD beyond the J-complement band, paired win 0.90. The Oracle Lens stays inside the band. And it's leak-free at layer 42: neither the J-Lens label words nor a set of 66 past and present verb tokens move. " +
      "Two honesty points. Provenance: we found this inside a pilot, so it is pre-registered as a secondary cell for the main run, and if it doesn't replicate it moves to the appendix. And it's a geometric claim: tense is a feature the model can verbalise, self-reported 87% of the time. So this is the NLA reading the distributed tail of a verbalisable feature beyond the top-r J-space, not reading what the model can't say. " +
      "The same test on language is void: a Cyrillic content set shows that direction leaks into the J-Lens. That's why every leak check now includes a content set, not just label words.",
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
      "So the NLA's apparent prompt reconstruction is itself generated from the latent. It confabulates a prompt consistent with what the activation now encodes. That is why our answer to the title's dichotomy is 'neither': the NLA is not reading the text and then solving, and it is not reading what the model can't say; its language model generates a consistent story from the latent. " +
      "Dose caveat: to flip the answer we needed large pushes, median 49% of the activation norm, 1.5 to 3 times the natural distance between countries; the main run makes the 109 items at alpha ≤ 1.5 primary. Twelve free-form samples is small; the main run scales this up.",
  },
];
