// Opening + background. Every slide: type, section, title (the claim), notes (speaker narration).

module.exports = [
  {
    type: "title",
    section: "Opening",
    notes:
      "Hi, I'm Suraj. This is my MATS 12.0 project in Neel Nanda's stream, at the end of the prep phase (6 October). " +
      "The title is the whole question: when a Natural Language Autoencoder (NLA) describes an activation, is it reading content the model never says out loud, " +
      "or is it reconstructing the prompt and re-solving it, which a text-only model could also do? " +
      "Everything is on Qwen3.6-27B at layer 42 of 64, with three public 'readers': the NLA, the Oracle Lens and the J-Lens. " +
      "The plan: the question and our answer so far, then enough background to follow it, the design, the gates, three findings, what's still pending, and the sprint plan (19 Oct to 1 Nov).",
  },
  {
    type: "question",
    section: "Opening",
    kicker: "The question, and the answer up front",
    title: "Provisional answer: neither. The NLA doesn't read what the model can't say, and doesn't re-solve: it generates",
    quote: "What can NLAs capture that Oracle Lens / J Lens don't? Is it really the stuff not in the workspace?",
    quoteSource: "Neel Nanda, MATS 12.0 central doc (\"Basic science\" project list)",
    answers: [
      { lead: "Not the non-workspace stuff:", text: "the non-J content Qwen uses looks like positional bookkeeping, and neither decoder reads col, though both read workspace features at the same dose." },
      { lead: "What it reads beyond J:", text: "the distributed tail of features the model can already verbalise (tense, pilot), and it writes a story consistent with the latent rather than re-solving the text." },
      { lead: "On the benchmark:", text: "the target information in its WorkspaceBench readouts is recoverable from the prompt text alone (0.89 from prompt vs 0.60 from readout)." },
    ],
    notes:
      "This is Neel's question, verbatim from the MATS 12.0 doc: 'What can NLAs capture that Oracle Lens / J Lens don't? Is it really the stuff not in the workspace?' And our title poses a dichotomy: does the NLA read what the model can't say, or re-solve what it read? Here is the provisional answer first: neither. " +
      "One: we surveyed about a dozen natural features. The content outside the J-space that the model actually uses looks like positional bookkeeping, things like characters since the last newline. On that feature, col, neither the NLA nor the Oracle Lens moves when we push it inside the activation, although at exactly the same push the NLA tracks language and tense at 0.83 to 0.92 AUROC. " +
      "Two: what the NLA does read beyond the J-space is the distributed tail of ordinary, verbalisable features: it still tracks tense with the top-1,536 J-space removed, where the Oracle Lens doesn't. And it doesn't re-solve the text: when we plant a different answer inside the activation, it rewrites the prompt to fit. It generates a consistent story from the latent. " +
      "Three: on WorkspaceBench, a text model given the prompt names the target more often than the same model given the NLA's readout, 0.89 against 0.60, and adding the readout to the prompt doesn't help. " +
      "Caveats up front: one clean non-J feature, pilot-scale numbers on a declared, excluded split, and the pre-registration hasn't frozen yet, so the title is provisional.",
  },
  {
    type: "why",
    section: "Opening",
    kicker: "Why it matters",
    title: "If readers only re-solve the prompt, they can't watch a model that stops writing its thoughts down",
    steps: [
      { head: "Today", body: "Much of a model's reasoning is visible in its chain of thought, so we can read it." },
      { head: "Tomorrow", body: "Reasoning may go silent: no CoT, latent reasoning, or CoT that is not faithful." },
      { head: "Then we need", body: "Readers that report what the model computed but never said. Neel's doc calls this preparing for a world without monitorable CoT." },
    ],
    test: "The bar for usefulness: a reader must report content that a text-only model, given the same prompt, could not reconstruct.",
    notes:
      "Why care? Neel's doc frames a whole agenda as 'preparing for a world where we don't have monitorable CoT'. " +
      "Today a lot of reasoning is written out. If that goes away, through latent reasoning or unfaithful CoT, our main window is reading activations directly. " +
      "Activation verbalizers like NLAs and the Oracle Lens are promising because they produce fluent descriptions. But fluency is cheap: if a reader works by reconstructing the prompt from the activation and then thinking about it, it adds nothing that a text model with the transcript couldn't do. " +
      "So the bar is: does the reader report content that isn't recoverable from the text? 'Non-workspace' content, information the model uses but can't verbalise, is the sharpest version of that test.",
  },
  {
    type: "residual",
    section: "Background",
    kicker: "Background 1 / 4",
    title: "A model's working memory: one 5,120-number vector per token, rewritten by each of 64 layers",
    prompt: ["The", "capital", "of", "the", "country", "famous", "for", "the", "Eiffel", "Tower", "is"],
    side: [
      "Each token carries a 5,120-dimensional vector (the residual stream).",
      "Each layer reads it and adds its update; what the model 'knows' at a point is spread through it.",
      "Every reader in this talk gets one vector: layer 42 (about two-thirds deep), last token.",
    ],
    notes:
      "Quick mental model. Qwen3.6-27B has 64 layers. Every token position carries a vector of 5,120 numbers, the residual stream, and every layer reads it and writes an update back. " +
      "By the last token of 'The capital of the country famous for the Eiffel Tower is', the vector somewhere encodes France, the relation 'capital', and also mundane things like where we are in the line. " +
      "All three readers in this talk get exactly one such vector: the output of decoder block 42 (we checked this convention carefully, more later), at the last token. " +
      "Layer 42 matters because the NLA we use was only trained there.",
  },
  {
    type: "jspace",
    section: "Background",
    kicker: "Background 2 / 4",
    title: "The 'workspace' idea: a small set of directions holds what the model is poised to say",
    points: [
      [{ text: "J-Lens: ", options: { bold: true } }, { text: "average the Jacobian from layer ℓ to the output, then unembed. It shows which tokens a direction would push the model to say, in general, not just here." }],
      [{ text: "J-space: ", options: { bold: true } }, { text: "the directions the J-Lens can read. We use the top-r singular directions, r* = 384 (and 4r* = 1,536) of 5,120." }],
      [{ text: "Most content is elsewhere: ", options: { bold: true } }, { text: "the J-space paper reports its excess variance explained 'never exceed[s] 10%'." }],
    ],
    inner: "J-space\ntop-384 of 5,120",
    innerSub: "verbalizable: poised to be said",
    outerSub: "everything else: syntax, position, counts, …",
    notes:
      "The 'global workspace' paper from Anthropic (July 2026) argues there's a privileged subspace holding what the model is poised to verbalise. They find it with the J-Lens: take the average Jacobian from a layer to the final layer, push a vector through it and the unembedding, and read off the tokens it would make the model say. Averaging over contexts is the point: it finds directions that are verbalizable in general. " +
      "The J-space is the span of those readable directions. We define it as the top-r singular directions of the unembedding-times-Jacobian matrix. Our rank rule gives r* = 384 out of 5,120, and we always check 4r* = 1,536 as well. The spectrum has no elbow, so we never say 'outside the workspace', only 'outside the top-r J-space'. " +
      "Crucially, most of the residual stream is not J-space; the paper says J-space excess variance never exceeds 10%. Content outside is what Neel's question is about.",
  },
  {
    type: "readers",
    section: "Background",
    kicker: "Background 3 / 4",
    title: "Three readers turn that vector into words, with very different machinery",
    readers: [
      {
        key: "jl",
        name: "J-Lens",
        tag: "linear token readout",
        lines: ["Vector × average Jacobian × unembedding → top tokens", "No training on explanations; neuronpedia n1000 lens", "Output: a list of tokens"],
        role: "Competitor for in-J content; leak detector for non-J",
      },
      {
        key: "ol",
        name: "Oracle Lens",
        tag: "LoRA verbalizer",
        lines: ["Vector injected as a token embedding (16000·unit(h))", "RL-trained on layers 20–60, step 4 (42 is off-grid)", "Output: '- ' concept bullets, often continuations"],
        role: "Example: \"- Le chat dort paisiblement dans son panier…\"",
      },
      {
        key: "nla",
        name: "NLA",
        tag: "verbalizer + reconstructor",
        lines: ["Verbalizer writes an explanation; reconstructor rebuilds the vector", "RL reward = reconstruction; FVE 0.761 (card 0.766)", "Trained at layer 42 only"],
        role: "Example: \"French descriptive sentence…\"",
      },
    ],
    notes:
      "Three readers, colour-coded for the rest of the talk: teal J-Lens, violet Oracle Lens, vermilion NLA. " +
      "The J-Lens is linear: no learned decoder, it just lists tokens. We use the neuronpedia n1000 lens, fit on 1,000 prompts, which is also WorkspaceBench's artifact. " +
      "The Oracle Lens is a LoRA on Qwen that takes the vector as a special token embedding, scaled to norm 16000, and writes concept bullets. It was RL-trained at layers 20 to 60 in steps of 4, so 42 is off its grid; we checked that this doesn't hurt it. In practice its outputs are often continuations of the text, like the French sentence shown. " +
      "The NLA is a pair of Qwen copies: a verbalizer that writes an explanation, and a reconstructor that tries to rebuild the vector from the explanation. RL rewards good reconstruction. We reproduced its fraction of variance explained, 0.761 against the card's 0.766. It exists only at layer 42. " +
      "Neel's observation is that the NLA seems to capture more than the other two. Our question is where that 'more' comes from.",
  },
  {
    type: "nonj",
    section: "Background",
    kicker: "Background 4a / 4",
    title: "'Non-workspace' content: the model uses it, a probe decodes it, but it sits outside the top-r J-space",
    criteria: [
      { head: "Decodable", body: "A linear probe recovers it at layer 42 (G1)." },
      { head: "Outside the top-r J-space", body: "Its direction is J-depleted (enrichment ≤ 1.5 at r* and 4r*) and the J-Lens readout does not track it (G2)." },
      { head: "Causally used", body: "Steering along its J-removed direction changes behaviour far more than random directions do (G4)." },
      { head: "Self-report (reported, not a gate)", body: "Self-report failures are often elicitation failures, so they can't define non-J." },
    ],
    example: {
      head: "Our running example: col",
      body: "In hard-wrapped text, col = characters since the last newline. The model tracks it to decide when to break the line. It is used, decodable, and invisible to the J-Lens.",
      sample: "The quick brown fox jumps over the\nlazy dog and keeps running across the▌",
    },
    notes:
      "What do we mean by non-workspace? Four properties. It must be decodable by a linear probe at layer 42. It must be outside the top-r J-space by two criteria: geometrically, its direction is not concentrated in the J-space (enrichment at most 1.5 at both ranks; 1.0 is random), and functionally, the J-Lens readout does not track it. And it must be causally used: steering along the J-removed direction must move the model's behaviour far more than random directions. " +
      "Self-report is recorded but not a gate: in my Phase 1 project colours were clearly in the J-space yet never named, so failure to self-report often just means bad elicitation. " +
      "The example we'll live with: col, the number of characters since the last newline in hard-wrapped text. The model uses it to decide when to break the line, and as you'll see it passes every gate.",
  },
  {
    type: "confound",
    section: "Background",
    kicker: "Background 4b / 4: the core confound",
    title: "A reader can be right without reading the latent: it can rebuild the prompt and re-solve it",
    readPath: { head: "Reading", body: "Reports the feature straight from the activation." },
    solvePath: { head: "Re-solving", body: "Rebuilds the prompt from the activation, then works out the answer itself, like a text model would." },
    natural: "On natural data both paths give the same, correct answer, so accuracy alone cannot tell them apart.",
    fix: "The fix: change the feature in the activation while the text stays fixed. A reader follows; a re-solver doesn't.",
    aside: "Why worry: the NLA's warm-start explanations were written by Sonnet from the text prefix alone, with no activation.",
    notes:
      "This is the key idea of the project. The activation at layer 42 encodes the prompt very well; the NLA's reconstructor proves that. So a reader has two ways to be right. Path one, reading: it reports a feature the model computed. Path two, re-solving: it effectively decodes the prompt, then does the reasoning itself, the way any capable text model would. " +
      "On natural benchmarks both paths give the same answer, so accuracy can't separate them. The way out is an intervention: change the feature inside the activation and keep the text exactly the same. A reader that reads will follow the change; a re-solver, which only sees the reconstructed text, won't. " +
      "There's a reason to worry about re-solving for NLAs specifically: their warm-start training explanations were written by Sonnet from the text prefix alone, without activations. In the concurrent pilot's breakdown about 60% of NLA multihop concepts were prompt wording or format.",
  },
  {
    type: "prior",
    section: "Background",
    kicker: "Prior work",
    title: "Prior work showed pieces; reading vs re-solving on natural features was still open",
    cards: [
      { head: "Concurrent MATS pilot", src: "LittleMeHere/nla-vs-lenses · Oct 2026", body: "Same model and readers. Multihop bridge named: J-Lens 39/100, OL 80, NLA 75. Readers mostly re-solve; 'on no task does the NLA find a hidden step the oracle lens misses'. Lists 'content known to be outside J-space' as not tested." },
      { head: "Viswanath", src: "LessWrong · Jul 2026 · Llama-3.3-70B", body: "Injected split vectors: the model names the J part 80%, the non-J part 0%; the NLA names both 100%. Follow-up: natural safety contrasts are mostly non-J (enrichment 1.45). Open caveat: 'thematic residue' vs decoding. No Oracle Lens." },
      { head: "Hong", src: "LessWrong · Jul 2026 · Qwen3.6-27B", body: "About 95% of the NLA's RL reconstruction gain lies outside the J-space, but about 90% of its specific claims are unsupported." },
      { head: "Global workspace paper", src: "Anthropic · Jul 2026", body: "Reports features used but not in J: part of speech, tense, and the character count during line wrapping. A source of natural ground truth." },
    ],
    open: "Still open: (1) natural features with known values, (2) a causal test that separates reading from re-solving, (3) all three readers, including the Oracle Lens, on Qwen3.6-27B at a matched layer.",
    notes:
      "We stand on four pieces of work, and I want to be precise and fair about each. " +
      "A fellow MATS 12 applicant posted a pilot on exactly this question on 4 October: same model, same readers. They found the prose readers mostly re-solve multihop questions, and wrote that the NLA never finds a hidden step the Oracle Lens misses. They explicitly list content known to be outside the J-space as not tested. I re-analysed their raw files; everything reproduces (appendix). " +
      "Viswanath showed on Llama-3.3-70B that the NLA names injected non-J vector components that the model itself cannot name. His unposted follow-up suggests natural safety content is mostly non-J, but he flags himself that the NLA might infer from thematic residue rather than decode. " +
      "Hong, on our exact model, found the NLA's RL gains are almost all outside the J-space, while most of its specific claims are unsupported. " +
      "And the workspace paper gives candidate ground truth: features the model uses that don't show up in the J-space. " +
      "What's open is the combination: natural features with known values, a causal test that separates reading from re-solving, and all three readers on Qwen at a matched layer.",
  },
];
