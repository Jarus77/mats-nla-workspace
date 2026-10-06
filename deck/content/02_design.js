// Design and research process.

module.exports = [
  {
    type: "twoByTwo",
    section: "Design",
    kicker: "The frame",
    title: "We sort reader behaviour into a 2×2: where the content lives × what the reader tracks",
    colHeads: ["Tracks the activation", "Tracks the text"],
    colSubs: ["output follows a feature moved with the text fixed", "output follows a text change with the feature clamped"],
    rowHeads: ["Content in the J-space", "Content outside the J-space"],
    cells: [
      { head: "Workspace reading", body: "Expected for all three readers: the calibration." },
      { head: "Re-solving workspace-type content", body: "Rebuild the prompt, then describe or answer it." },
      { head: "Non-workspace reading", body: "Neel's hypothesis for the NLA.", highlight: true },
      { head: "Re-solving non-J-type content", body: "e.g. counting characters in decoded text." },
    ],
    footnote: "Plus 'unsupported': output that follows neither manipulation (confabulation). Measured at population level by intervention sensitivity.",
    notes:
      "This 2×2 came out of the first critique round. Rows: does the content live in the J-space or outside it. Columns: does the reader's output follow the activation, or follow the text. " +
      "Top-left is workspace reading, which we expect from everyone and use as calibration. Right column is re-solving: the reader is really responding to the text. " +
      "Bottom-left is the cell Neel's question is about: a reader that reports content outside the J-space by actually reading the activation. " +
      "We measure the columns by interventions: move the feature with the text fixed (left column) or move the text with the feature clamped (right column). " +
      "Our original design had four 'buckets', prompt / workspace / non-workspace / unsupported, but Fable pointed out they weren't a partition: character count is non-J and also computable from the text. The 2×2 fixes that.",
  },
  {
    type: "tools",
    section: "Design",
    kicker: "Three tools",
    title: "Three tools attack the 2×2 from different sides",
    tools: [
      { tag: "A", head: "Natural non-J features + text-fixed steering", body: "Gate natural features for 'outside the top-r J-space' (Part A), then push the best one (col) up and down inside the activation with the text fixed (Part B).", cell: "Targets: bottom-left cell" },
      { tag: "C", head: "Planted intermediates", body: "Swap the bridge entity of a two-hop fact inside the activation (Eiffel Tower → France becomes Sweden) with the prompt fixed, at a dose that flips the model's own answer.", cell: "Role: in-J calibration, and a window on how the NLA builds its story" },
      { tag: "D", head: "Re-solving decomposition", body: "On WorkspaceBench, compare readers with text-only models that never see the activation, and with a model re-solving from the reader's own readout.", cell: "Targets: right column (one consistency check open)" },
    ],
    currency: "Common currency: paired log-odds of two candidate strings in each reader's native format, compared against null bands of K = 20 random directions of the same norm (never subtracted).",
    notes:
      "Three tools. Tool A is the headline: find natural features that live outside the J-space, then steer them with the text fixed. That's split into Part A, 'what even counts as non-J', which turned into a finding in itself, and Part B, the deep dive on col. " +
      "Tool C plants an intermediate (we now treat it as in-J calibration, because the planted direction is strongly J-enriched): we move the activation along the direction separating, say, France from Sweden, so the model's own answer flips while the prompt is unchanged. Does the reader report the planted bridge or the one implied by the text? " +
      "Tool D is a decomposition on the published benchmark: how much of each reader's success can a text-only model match? " +
      "We cut a fourth tool, split-injection, after critique: Viswanath already has the 70B result and it's the most out-of-distribution manipulation. " +
      "The results come in four parts: A, what counts as non-J; B, the col null; C, what the NLA does read beyond the workspace; D, the benchmark decomposition. The scoring currency is the same everywhere: paired log-odds between two candidate strings, written in the reader's own style, compared against random directions of the same size.",
  },
  {
    type: "followFeature",
    section: "Design",
    kicker: "The decisive measurement",
    title: "Push a feature up or down inside the activation, keep the text, and ask whether the reader's words follow",
    steps: [
      { head: "Same text, ± push", body: "h± = h − (v·h)v ± Δv\n(centred: equal norm on both sides)" },
      { head: "Reader scores", body: "log p(\"near the end of a long line\") − log p(\"at the start of a line\")" },
      { head: "Does it follow?", body: "paired win rate and mean shift over items" },
      { head: "Against chance", body: "z vs K = 20 random directions of the same norm (isotropic and J-complement)" },
    ],
    guards: [
      { head: "Leak detector", body: "The J-Lens must not move (two-sided, vs the J-complement band). If it moves, the content leaked into J and the dose is void for non-J claims." },
      { head: "Dose in range", body: "q90 of the natural projection: ‖Δ‖/‖h‖ ≈ 0.08–0.11 for col. FVE guard: the NLA's reconstruction must not degrade by > 0.05." },
      { head: "Direction", body: "Steer along the J-removed direction (top-1,536 J-space projected out) so the push itself carries no J content." },
    ],
    notes:
      "This is the measurement every later result uses, so let me be concrete. Take a passage and the activation h at the read token. Build two versions: push the col direction up by Δ, and down by Δ. We use a 'centred' construction, removing the item's own component first, so both versions have exactly the same norm; I'll show later why that mattered. " +
      "Feed both to the reader and score two candidate phrases in its native style, say 'near the end of a long line' versus 'at the start of a line'. If the reader reads col, the log-odds should be higher for the pushed-up version on most items. " +
      "We never compare against 0.5 alone. We repeat the whole thing with 20 random directions of the same norm and report a z-score against that null band. " +
      "Three guards. The leak detector: the J-Lens must not move, or the content has leaked into the J-space. The dose must be in the natural range: we use the 90th percentile of the natural projection, about 8 to 11% of the activation norm. And we steer along the direction with the top-1,536 J-space removed.",
  },
  {
    type: "loop",
    section: "Design",
    kicker: "How we worked",
    title: "An adversarial research guide changed the design four times before any headline run",
    // a fifth round (D21) reframed the story into the four parts used in this deck
    loop: ["Plan + pre-register", "Agents run gates", "Decision log D1–D21", "Hand off to Fable", "Critique: accept or push back"],
    changes: [
      { round: "Round 1", what: "Four attribution buckets weren't a partition (a character count is both non-J and text-computable).", fix: "→ the 2×2 frame (D2)" },
      { round: "Round 2", what: "Steering at layer 38 lets non-J content rotate into J before the read layers.", fix: "→ J-removed directions, leak detector, read-layer steering (D4)" },
      { round: "Round 3", what: "'Probe survives J-removal' passes topic and language too: it can't label anything.", fix: "→ calibrated gate: enrichment + J-Lens readout (D8–D9)" },
      { round: "Round 3–4", what: "The ± contrast isn't dilution-immune; OL's 'anti-tracking' might be an artefact.", fix: "→ centred steering: OL 0.45 → 0.49 = chance (D9, D11)" },
    ],
    pushback: "We pushed back too: a linear probe is a reference, not a ceiling; the FVE out-of-distribution check is partly circular (the reconstructor was co-trained).",
    notes:
      "A word on process, because it shaped the results. I ran the engineering with parallel coding agents doing gates and experiments, logged every decision (D1 to D20 so far), and after each block handed a write-up to 'Fable', a separate model instance acting as an adversarial research guide, asked to be blunt. " +
      "Four critiques changed the design. Round 1: my four attribution buckets weren't a partition, which gave us the 2×2. Round 2: steering upstream at layer 38 can let the model move non-J content into the J-space before the read layer, a hole I'd missed; fix: J-removed directions, a leak detector, and steering at the read layer. Round 3: once we noticed the standard non-J test passes topic, Fable helped design the calibrated gate. And the ± contrast isn't immune to dilution, which led to centred steering and made an apparent Oracle Lens effect disappear. " +
      "It wasn't one-way: I pushed back that a linear probe is a reference, not a ceiling, and that using the NLA's own reconstructor as an out-of-distribution detector is partly circular. Both were accepted.",
  },
];
