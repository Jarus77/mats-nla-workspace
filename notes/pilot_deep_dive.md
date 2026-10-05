# Pilot deep dive: `LittleMeHere/nla-vs-lenses`, plus the Viswanath and Hong posts in full

Compiled 2026-10-05. Sources:
- **The pilot.** Cloned to `scratchpad/nla-vs-lenses` at commit `58d34a8`. I read the README, STATUS, every script and the `runs/` raw files.
- **Viswanath.** The LW post (via GraphQL) **plus his GitHub repo** `PranavViswanath/nla-introspection` (`b4f744c`, 2026-07-13). The repo includes an **unposted follow-up** (`jspace-followup/`).
- **Hong.** The LW post (via GraphQL).

Every pilot number marked "recomputed" was rederived from the raw JSONL with my own scorer (`scratchpad/analysis/{common,mh,abl}.py`). All the headline counts reproduce exactly.

---

## Part A. The pilot

### A1. Exact methods

| Aspect | What they did | Source |
|---|---|---|
| Model | Qwen/Qwen3.6-27B, bf16. Loaded through `wsbench` `Producer` (transformers 5.x, `Qwen3_5ForCausalLM`). RunPod, one 80 GB GPU. | `smoke_pass_a.py`, `pod_bootstrap.sh` |
| DeltaNet kernels | `causal_conv1d` missing (torch fallback). `flash-linear-attention` was installed on later pods only, so numerics may differ across pods. The abl50 residuals were **recaptured** on a new pod ("the L42 residuals from the first pod were not copied off"). | logs, `ablate_chain.sh` |
| Capture | Residual after decoder block `model.layers[L]` (wsbench forward hook on block output, fp32 to CPU). Saved to `.pt`, then fed to the readers. **Capture-once / two-pass**: base model plus lenses plus OL in pass A; NLA alone in pass B. | `smoke_pass_a.py`, `smoke_pass_b.py` |
| Read position | **multihop:** the bank's "last word" token. That is offset −2 for the 33 prompts ending in a bare space, else −1. Plain render, no chat template. **chain/brew:** the last token of the chat render (assistant onset after the empty think block; brew prefilled "Answer" so the read is at ":"). | items header, `chain_pass.py` |
| J-Lens variant | **The wsbench default:** `neuronpedia/jacobian-lens` `Qwen3.6-27B_jacobian_lens_n1000.pt` (wikitext, n=1000). Readout is the "magnitude-free cosine" `(W_U J_L h)/‖J_Lᵀ W_U[t]‖`, top-10. A memory patch keeps J on CPU and W_U in bf16, which costs score precision. **Not** `camilablank/workspace-lenses` (that repo is downloaded but serves the R-lens). Our handoff says J-Lens = `camilablank/workspace-lenses`. **[VERIFY which J we use; they may differ.]** | `patch_jlens_mem.py`, wsbench `methods.py` |
| Oracle Lens | `agu18dec/local-workspace` `s3d.ddp600.s0/iter_000600` (gated dataset; byte-identical to `agu18dec/olens_and_ar/olens_s3d_rl600`). Input: unit-normalise h, ×16000, **replace** the marker embedding (㈜). The prompt contains `{layer}`. Sampling: T=1, top_p .95, top_k 64, 256 tokens, **k=1** (one string with 4 bullets). | wsbench, raw files |
| NLA | `ceselder/qwen3.6-27b-nla-rl`, `av_base` plus `av_rl_adapters/iter_000400` (the model card suggests 300). Input: a norm-matched **add** at the reader's block-1 output at the marker. Same sampling, **k=1**, 256 tokens; 3/100 readouts truncated mid-sentence. | `smoke_pass_b.py` |
| Layers | multihop: 42 (main) and 16/28/36 (sweep); **OL also at 44**; J at 20..60 step 4 plus 42 plus 44. chain: readers at 42 and at 56. brew: 42. | runs |
| Deletion (abl50) | For each bridge string, take the token ids of 6 case/space variants. d_t = Jᵀ W_U[t] (**no final-norm gain g**). QR-orthonormalise, then h ← h − QQᵀh. 2–6 directions (45/50 items have 4). Control: as many directions from random vocab tokens (a J-space-random-like control). | `ablate_pass.py` |
| Strong deletion (abl3) | Bridge directions plus every vocab token whose direction has **cos > 0.8** with a bridge direction, plus up to 3 rounds of "J top-50 still shows a bridge-matching token → add its neighbourhood". Mean **31.2** directions, median 19. (abl2 used cos > 0.5: mean **1,152** directions, which damaged the control too, so it was superseded.) | `ablate2_pass.py`, recomputed |
| Swap | Strong-delete **both** own and partner bridge sets, then add `k·|s|·unit(d_partner_main)`, where s = unit(d_own_main)·h. **k escalates through 1, 2, 4, 8, 16 until J top-10 shows the partner.** Recomputed: **34/50 items hit the k=16 cap**; the rest stopped at k = 2/4/8 (5/4/7). | `ablate3_chain.sh`, recomputed |
| Scoring | multihop: **exact word match** of the bank intermediate (no scorer script is committed; my word-boundary regex reproduces 39/80/75 exactly). chain: digits / English words / Chinese numerals, then set intersection with the intermediates (`score_chain.py`). brew: prompt-blind role judge (Claude `claude -p` and Gemini 3.8 Flash on Vertex, T=0). | `score_chain.py`, `judge_roles.py` |
| Chance | Each readout scored against **every other item's** targets, averaged. | `score_chain.py` |
| Claim audit | Stage A: `claude -p` splits the readout into unique concepts. Buckets assigned in order: intermed → jlens (J top-10 overlap) → past_ctx (any content-word stem overlap with the prompt) → next_tok → format_meta (≥50% meta words) → other. The "other" concepts are then judged by `claude -p` as contradicted / restates / form / unstated. **The J-lens reference file is L44 while the NLA read is L42** (`nla_claims_vs_jlensL44.jsonl`). | `classify_claims.py`, `judge_other.py` |

### A2. Every number (recomputed where possible)

**multihop, n=100, L42** (recomputed, exact match to STATUS)

| | J-Lens | OL | NLA |
|---|---|---|---|
| Names bridge | 39 | 80 (L44: 72) | 75 |
| Chance (other items' bridges) | 0.73 | 1.67 | 1.21 |
| Names the **answer** (target) | – | 89 | 75 |

- J any layer (my lenient prefix matcher): 81; STATUS says 76.
- J per layer: L20 3, L24 12, L28 8, L32 8, L36 25, L40 35, L44 39, L48 42, L52 41, L56 59, L60 54.
- Bridge never appears in the prompt: 0/100. Model top-1 next token equals the target on 74/100 (17 have a space as top-1).
- **Per-item overlap (new):**
  - Venn (J, OL, NLA): all three 36; OL and NLA only 36; none 16; OL only 6; NLA only **3**; J and OL only 2; J only 1.
  - **NLA hits where OL misses: 3/100; OL hits where NLA misses: 8.** McNemar p = 0.23. There is no evidence the NLA finds anything OL misses.
  - phi(OL, NLA) = **0.69**; phi(J, OL) = 0.35; phi(J, NLA) = 0.32. The two prose readers succeed on the same items, which fits both decoding the same prompt-recoverable information.
  - When J shows the bridge: OL 38/39, NLA 36/39. When J misses: OL 42/61, NLA 39/61.
- **Bridge co-mentioned with the answer:** NLA 67 of 75 bridge hits (8 bridge-only); OL 79 of 80 (1 bridge-only). This is the signature of stating the solved fact ("iron (Fe)"), not reading a mid-computation variable.
- **By source:**
  - jlens_source (41 items): J 6, OL 23, NLA 21.
  - staging (24): J 14, OL 23, NLA 23.
  - staging_b3 (35): J 19, OL 34, NLA 31.
- **Verbosity and prompt echo:** NLA 127 words vs OL 73. Prompt's last 4 words quoted verbatim: NLA 11/100, OL 8/100.

**Layer sweep, multihop n=50** (recomputed)

| Layer | J | OL | NLA |
|---|---|---|---|
| L16 | 0 | 9 (OL off-contract, below 20) | 11 (NLA off-layer) |
| L28 | 1 | 26 | 27 (NLA off-layer) |
| L36 | 9 | 33 | 28 (NLA off-layer) |

**Deletion abl50** (2–6 dirs, n=50; recomputed)

| | Original | Random control | Deleted | ctrl→deleted transitions (hit→hit / hit→miss / miss→hit) |
|---|---|---|---|---|
| J | 14 | 15 | **0** | 0 / 15 / 0 |
| OL | 33 | 30 | 17 | 14 / 16 / 3 |
| NLA | 33 | 36 | 25 | 23 / 13 / 2 |

**Strong deletion and swap, abl3** (cos > 0.8, mean 31 dirs; recomputed)

| | Random | Strong | Swap: keeps own | Swap: planted | Planted in random arm (null) | Planted when J shows planted |
|---|---|---|---|---|---|---|
| J | 15 | 0 | 0 | 17 (STATUS: 14) | 0 | – |
| OL | 33 | 20 | 20 | 7 | 0 | **6/17** |
| NLA | 34 | 23 | 20 | 6 | 0 | **4/17** |

- Survival among items hit under the random control: OL 17/33 = 52%; NLA 20/34 = 59%.
- abl2 (cos > 0.5, ~1,152 dirs): OL random 19 → strong 10; NLA 20 → 11; planted: OL 4 (null 3), NLA 1 (null 3). **Uninterpretable.**

**Single-sample test-retest noise (new; this matters a lot)**

The random-direction control barely changes h, so comparing original vs control readouts measures sampling noise.

| Comparison | Items that flip | Cohen's κ |
|---|---|---|
| OL original vs abl50 control | 7/50 | 0.70 |
| OL original vs abl3 random | 10/50 | 0.55 |
| NLA original vs abl50 control | 7/50 | 0.67 |
| NLA original vs abl3 random | 7/50 | 0.68 |
| OL L42 vs L44 (n=100) | 16/100 | 0.57 |

**So any single-sample difference under about 10/50 between conditions is within retest noise.** The 33→20 deletion effects pass this bar; most swap and brew differences do not.

**chain, n=60** (recomputed with their scorer)

| Read | J | OL | NLA |
|---|---|---|---|
| L42 | 6 (chance 3.6) | 27 (chance 21.0) | 9 (chance 12.4) |
| L56 | 25 (chance 6.8) | **36** (chance 12.1) | 13 (chance 8.9) |

- J per layer: 0 through L36; L40 7, L42 6, L44 6, L48 8, L52 7, L56 25, L60 6.
- Distinct numbers per readout: OL 9.3 (L42) / 6.2 (L56); NLA 5.3 / 4.2. This is why chance rates are high.
- At L56, OL names the **answer** in 39/60, more often than it names an intermediate (36). NLA names the answer 26/60.
- Venn at L56 (NLA, OL): OL only 28, both 8, NLA only 5, neither 19.
- OL states the start number in 16 readouts at L42. STATUS reports 31/36 L56 hits without the start number.
- **Asymmetry:** at L56 OL is on-contract and the NLA is off-layer. At L42 OL is off-contract and the NLA is on-layer. No cell is fair to both.

**brew, n=50, L42 at ":"** (recomputed)
- Claude and Gemini judges agree on the intermediate label in 34/50 for each reader.
- Role-correct intermediate: NLA 5 (Claude) / 3 (Gemini); OL 2 / 5. Gemini says "none" for the intermediate in 32/50 (NLA) and 24/50 (OL).
- Readers name the *answer* correctly in only 3 (NLA) and 5 (OL). So at this cell they are not even reading the output.
- J top-10 has the intermediate at L42 in 0/50, L56 19, L60 46.
- Deletion (z in any role, Gemini): OL 7 → 0, NLA 12 → 2.

**NLA claim audit (multihop, n=100 readouts)**
- 1,696 concepts:
  - other 461 (27%)
  - past_ctx 379 (22%)
  - format_meta 378 (22%)
  - jlens 254 (15%)
  - intermed 200 (12%)
  - next_tok 24 (1%)
- The 461 "other" concepts, judged: form 45%, unstated 27%, contradicted 22%, restates 6%.
- 66/98 readouts have at least one contradicted concept.

### A3. Methodological weaknesses (ranked by how much they threaten the conclusions)

1. **"Re-solving" is confounded with "non-workspace reading".** This is the central one.
   - The deletion removes **J-lens readout directions only**. OL and NLA naming surviving that deletion (52–59%) is what you would see if they re-solve from the prompt.
   - It is **equally** what you would see if the bridge is carried in **non-J directions** the readers can decode. That is exactly Neel's hypothesis.
   - The pilot reads survival as re-solving. Without a text-identical counterfactual, or a probe showing the bridge is gone after deletion, the two are indistinguishable. The pilot's own "not ruled out (a)" admits this.
2. **Reader-side edits only, with no behavioural check.**
   - Deletion and swap edit the vector handed to the readers. The model never runs on the edited residual.
   - Nobody checks that the bridge is causally used by the model, or that deleting or planting it would change the model's answer.
   - The planted vector is one token-readout direction at up to 16× the own-bridge coefficient. That is off-manifold, so readers may ignore it for uninteresting reasons.
3. **Weak and selected swap.**
   - The scale escalates *until J-Lens shows the planted token* (selection on one reader's outcome).
   - **34/50 items still hit the 16× cap**, and J shows the planted token in only 14–17/50.
   - Prose readers follow the planted token in 4–7/50, within the noise band above.
4. **Sample size and k=1.** n=50–100 dev items, one sample per cell. Measured retest κ is 0.55–0.70, about 14–20% of items flip. Brew and swap differences (2–7 items) are noise.
5. **Mention scoring, not role-aware scoring.** Word match counts mentions. Bridge mentions co-occur with the answer in 67/75 (NLA) and 79/80 (OL). Brew's role-aware judge shows mentions overstate claims.
6. **Off-contract and off-layer cells.**
   - OL at 42 is off-contract, and `{layer}=42` appears in its prompt. OL at 16 is outside the 20–60 band.
   - The NLA at 16/28/36/56 is off its only training layer.
   - The "L28: readers 26–27 vs J 1" argument is made with an **off-layer NLA**.
   - The chain L56 comparison pits an on-contract OL against an off-layer NLA.
7. **Possible NLA layer off-by-one is unverified.**
   - wsbench L = output of block 42 = `hidden_states[43]`.
   - EasyNLA's `layer_index: 42` may mean `hidden_states[42]` (output of block 41).
   - **New supporting evidence:** Viswanath's Anthropic Llama NLA "L53" is explicitly `hidden_states[53]` = block-52 output, so this convention is in use in the ecosystem.
8. **No probe-based presence ground truth**, and no reader-free label for whether the model uses the hidden step.
9. **No text-only baseline.** WorkspaceBench prompt-only scores 0.90 on multihop, higher than every reader at L42. No TextNLA, so "re-solving" is never directly measured.
10. **Judges unvalidated.**
    - The two judges disagree on about 1/3 of brew labels.
    - `claude -p` is the splitter and the "other" judge.
    - The bucket rules are lenient: one shared word stem means "prompt words".
    - The J reference for buckets is at L44 against an L42 NLA read.
11. **Read position and answer leakage.** At the multihop read token the model's top-1 is already the answer (74/100). A reader can read the imminent answer and back-infer the bridge, which inflates "bridge" hits for both prose readers.
12. **Hygiene.**
    - Different pods and kernels; recaptured residuals.
    - bf16 J scores.
    - The L56 run was added post hoc after seeing where J peaks (they disclose this).
    - Items share bridges (75 distinct first bridges / 100; Italy ×5), which slightly inflates chance.
    - NLA iter 400 rather than the card's 300.

### A4. What to ADOPT

- **Capture-once to disk, then feed identical vectors to every reader.** Also two-pass memory: base plus lenses plus OL, then the NLA alone. This guarantees identical inputs, which matters given the DeltaNet kernel issue.
- **An item-permuted null:** score each readout against other items' targets. Extend it to forced-choice foils and to AUROC.
- **A matched-count random-direction control** for every directional edit (their random vocab-token directions are a J-space-random control).
- **A per-layer J sweep to find where the content is linearly present** before testing readers there (chain: L56). Then test the readers at matched, on-contract layers.
- **"Does the readout contain the inputs needed to recompute?"** (their start-number check) as a re-solving covariate. Generalise to a per-item **prompt-recovery score** (their untested explanation; cheap; CPU).
- **A prompt-blind role judge** with candidate lists (brew), and **two judges with agreement reported**.
- **The claim taxonomy** (intermediate / J-overlap / prompt / next-token / format / other → contradicted / restates / form / unstated) as a starting codebook for our claim audit, with validated judges.
- **Raw readouts plus logs committed, and every number recomputed from raw files.**
- **Infrastructure:**
  - the J-lens memory patch for one H100;
  - the open OL model-repo copy;
  - the observation that `causal_conv1d` and fla must be pinned.

### A5. What NOT to copy

- Word-match mention rates as the primary metric. Use forced-choice teacher-forced log-prob AUROC against matched nulls, plus role-aware judged claims.
- k=1 sampling. Use k≥5 for free-form readouts, and report retest κ.
- Off-contract (OL L42) or off-layer (NLA ≠ L42) cells as the primary comparison.
- Reader-side-only edits presented as causal evidence about the model. Interventions go into the **model's** forward pass upstream, with a behavioural flip check.
- Escalating the intervention until one of the compared readers responds.
- Interpreting "survives J-direction deletion" as "re-solved from the prompt".
- Unvalidated `claude -p` judges, and comparing claims against a J reference at a different layer.

### A6. What the pilot actually establishes (my read)
1. At L42 on multihop, OL ≥ NLA ≫ J-top-10. The NLA's unique hits (3/100) are no more than noise.
2. Prose-reader bridge mentions are mostly "solved-fact" statements co-occurring with the answer.
3. Part of what prose readers say depends on J-lens directions: deletion drops OL 30→17 and NLA 36→25, beyond retest noise.
4. The rest is **unattributed**: re-solving *or* non-J reading.
5. On chain at L56, OL reads numbers that J top-10 often misses. But OL also names the answer more often than the intermediate, and L56 is late (circular-zone risk).
6. Nothing in the pilot tests content known to be outside the J-space.

---

## Part B. Viswanath (Llama-3.3-70B split injection): exact methods

Read in full: the post, plus `RESULTS.md`, `PROGRESS.md`, `docs/EXPERIMENT.md`, `exp/expA_boundary/p1c3_mvp.py`, `exp/expB_linecount/p1d_linecount.py` and `artifacts/p1d_results.jsonl`, and the follow-up `jspace-followup/README.md`.

**Setup**
- Target: `unsloth/Llama-3.3-70B-Instruct`, bf16.
- NLA: Anthropic's `kitft/Llama-3.3-70B-NLA-L53-av` via SGLang TP=2, `injection_scale=30`. 4×H200.
- **Layer convention:** the NLA reads `hidden_states[53]` = block-52 output.

**J-lens**
- Fitted with `anthropics/jacobian-lens` on **128 wikitext prompts** at block 52 (~1.7 GPU-h). The paper used about 1,000.

**How the J-space is defined**
- **Top-r right singular vectors of J_52** (`torch.linalg.svd(J)`; V = Vhᴴ), d=8192. This is **not** Anthropic's per-activation sparse NNLS over J-lens vectors (k≈25).
- Spectrum: 50% of J energy at rank 1,076, 90% at 4,450. No elbow.
- Concept-vector J-fraction: 3.4% @ r=100, 6.1% @ r=200, 10.2% @ r=400.
- **A random vector already puts 2.4% inside r=200**, so concepts are only about 2.5× chance (follow-up: enrichment 1.90×).

**Concept vectors and the split**
- Concept vectors: diff-in-means of the last-token activation for "Tell me about {word}" minus the mean of 100 baseline words.
  - 16 concrete concepts, e.g. Elephants, Otters, Cinnamon, Granite.
  - 8 emotional/abstract concepts: fear, betrayal, secrecy, loneliness, justice, danger, love, celebration.
- Split: v_J = V_r V_rᵀ v and v_nonJ = v − v_J. **Each component is renormalised to ‖v‖.**

**Injection**
- Add s·component at the **block-52 output at all positions** (prompt and generation).
- s ∈ {1, 2, 4, 8}, multiplying the raw concept-vector norm. Main s=4; at s=4 the full and nonJ conditions are often "incoherent".
- **The NLA reads the last-position activation at the same layer the vector was added to.** It is decoding h + s·v directly (an at-site capability test). Upstream propagation is not tested.

**Report channels**
- Macar two-turn introspection prompt.
- P(yes) from logits: **abandoned.** randJ gave 0.058–0.095, above real J components (0.038).
- "Named" = case-insensitive substring of the concept in greedy 64-token text, with a coherence filter (unique-word ratio > 0.3).
- NLA = substring in k=2 samples (T=0.7, 200 tokens).

**Rank sweep (s=4).** Columns: model naming (coherent only), NLA naming, and J variance share.

| r | J: model named | J: NLA | nonJ: model named | nonJ: NLA | J var share |
|---|---|---|---|---|---|
| 200 | 0.05 (n=19) | 0.74 | 0.25 (n=4) | 0.98 | ~6% |
| 800 | 0.33 (9) | 0.91 | 0.29 (7) | 1.00 | ~15% |
| **2000** | **0.80 (5)** | 1.00 | **0.00 (11)** | **1.00** | ~31% |
| 4000 | 0.60 (5) | 1.00 | 0.00 (7) | 0.80 | ~53% |

**Other numbers**
- **r=200, n=16 MVP:**
  - P(yes): J 0.038 [0.014, 0.069] vs nonJ 0.084; paired diff −0.046, sign-flip p = 0.077. **This reversed the n=6 pilot.**
  - NLA: full 0.97, J 0.61, nonJ 0.94. Control 0.00.
- **Controls** (norm-matched random full / J / nonJ): NLA 0.00 on all three. P(yes) 0.003–0.095.
- **Swaps, r=2000, s=4, 6 pairs** (A's nonJ + B's J, norm-matched to A):
  - The model produced a name in 3/12 trials, every one the J-side concept; the nonJ side was named 0/12.
  - The NLA named both sides at about 1.0. Config controls (n=10): every channel at 0.
- **Liveness:** concept-token logprob uplift on neutral carriers (last 32 positions):
  - nonJ r=200: +13.7 nats (s=2) and +21.1 (s=4), 32/32 positive.
  - nonJ r=2000, s=4: **+17.1 nats**, 20/20 positive.
- **Phase 0 Result 3:** the J-lens readout of v_nonJ at r=200 **names the concept for 8/8**. So "nonJ" is not lens-dark. The J-lens readout of nonJ at r=2000 is not reported.
- **Line counting (p1d, 12 passages, widths around 40, my tabulation of the raw rows):** NLA states a number within ±3 of the true count
  - at newline tokens: **5/155 (3%)**;
  - pre-newline: 4/155;
  - mid-line: 3/72;
  - unwrapped control: 0/72;
  - **report-condition last token: 0/12**.
- Counting-lexicon mentions occur at 62/155 newlines vs 41/72 unwrapped mid-line positions, so they are not specific. **A natural non-J feature, and the NLA does not read it.**

**Unposted follow-up (jspace-followup, Jul 13), directly relevant to our "claim attribution"**
- **Enrichment** = J-energy / (r/d):
  - random 1.00; concept vectors 1.90;
  - safety contrasts 1.45 on average;
  - grader gaming 1.00 at every r from 200 to 6000.
- **NLA reads of norm-matched components of natural safety activations** (keyword grading; floors 0.00–0.03 on neutral prompts, 0.10–0.22 for wrong-scenario keywords):

  | Component | r=200 | r=2000 | r=4000 |
  |---|---|---|---|
  | full | 0.55 | 0.55 | 0.55 |
  | J-only | 0.18 | 0.38 | 0.38 |
  | nonJ-only | 0.53 | 0.55 | 0.57 |

- Banked injections, read the same way: full 0.99, J 1.00, nonJ 0.99. So at-site injections cannot discriminate the components.
- **Equal-norm ablation dose-response** (P(even) under grader gaming): J and nonJ components are comparably causal. An earlier "nonJ uniquely causal" result was a norm artifact.
- Random J directions are only **2–2.6× more output-influential (next-token KL)** than random non-J directions, against the 6–11× the spectrum predicts.
- His caveat: component reads may reflect "thematic residue". The splice control (p1h2) is queued, not run.

**Limitations (stated and mine)**
- **At-site injection.** It tests whether the NLA can decode a direction in its input vector, not whether non-J content arises or is used naturally.
- **Large doses.** s=4 × the concept norm, at all positions.
- **Tiny cells.** 5–15 per cell and 6 swap pairs; substring grading; one trial prompt.
- **The rank was chosen post hoc** at the naming crossover. The J-lens was fit on 128 prompts.
- **The nonJ part still carries the concept's unembedding** (+17 nats), so the NLA could be reading token identity.
- Same-base confound; no Oracle Lens; no text-only baseline.
- **Natural line count is negative for the NLA.**
- **Overlap:** his follow-up already does reader-side J/nonJ component reads on *natural* activations (Llama). Our claim attribution must go beyond reader-side decomposition: model-side interventions, OL, Qwen, TextNLA.

## Part C. Hong ("95% of NLA RL gain outside J-space"): exact methods

**Model and NLA**
- Qwen3.6-27B, but **his own NLA training run**: FVE 0.185 → 0.413, far weaker than ceselder's 0.766 at iter 400.
- Anthropic's public checkpoints score FVE 0.712 on his held-out split; the model is unspecified.

**Confabulation**
- An LLM judge (unnamed) finds about 90% of specific claims (names, numbers, quotes) unsupported by the input.
- 10 LLM judges prefer post-RL over pre-RL only 54.5% of the time [45.2, 63.5].

**Swap test** (frozen AR, 400 samples; ΔFVE)

| | Entity swap (same-category) | Abstract swap (topics/styles) |
|---|---|---|
| His NLA | **−0.219** | −0.072 |
| Anthropic checkpoint | −0.394 | −0.103 |

Example: "Bing Maps/Redmond" → "Stripe/Austin" gives −0.89 on one sample.

**Fact-check reward**
- g = fraction of the description's regex-extracted named entities and numbers found in the source tokens.
- Over 800 steps: g 0.124 → 0.268 → **0.417**; FVE 0.4126 → 0.4172 → 0.4194.
- Judges prefer the fact-reward model 83.6% [75.6, 89.4].

**J-space split**
- Reconstruction is "split exactly" into the in-J and out-of-J parts.
- In-J is "nearly saturated from the start"; out-of-J "almost tripled"; "about 95% of training gains live outside the J-space".
- The split is unchanged (diff < 0.1%) under the fact reward.
- **No rank, J-lens source, layer or exact formula is given.** The only definition offered is "a small verbalizable subspace carrying 6–7% of activation energy".

**Critical reading**
- **If the J-space holds about 6–7% of activation variance, a reconstruction gain spread in proportion to variance would put about 93% outside J by default.** So "95% outside" is close to the chance-level allocation and is **not evidence of special non-J content** (cf. Viswanath's enrichment: random vectors put 2.4% inside r=200).
- He tests no text-level content.
- **The lit review's "unlocated" post is resolved.** Hong's "an LLM that never sees the activation reconstructs about as well" links to LW `LQXWiF8PyJ5ojNsEv`, which is **Zhang's robustness post**.

**Limitations:** one model; his own weak NLA (probably not ceselder's); exact-string fact check; unnamed judges; no definition of the J-space.

## Part D. Implications for our design (feeds `prereg_v0.md`, revision 1)

Revision 1 (lead/Fable, approved) **cuts split-injection** and makes natural-feature steering the headline. The points below are updated to match.

1. **A J/non-J split by SVD rank is arbitrary without calibration.** The r=200 vs r=2000 conclusions flipped on Llama. Report curves over r, fix r* by an a-priori rule that does not use the readers under test, and check against Anthropic's k=25 NNLS definition. **Non-J = the probe survives J-space removal, plus absence from the J-Lens. Self-report is reported, not gated.**
2. **Viswanath's at-site injection only shows decoding capability, so we do not repeat it** (optional 20-item replication at most). The test is natural features moved **along their probe direction** with the text fixed, at percentile doses, with a reconstructor-FVE OOD gate.
3. **(C) planted intermediates must move the feature (probe direction).** Not J-Lens token directions (the pilot's swap favours the J-Lens by construction). Not full-residual patches (these move the decodable prompt, so a re-solver would follow them). Keep only items where the model's answer flips **and** the prompt stays decodable.
4. Use enrichment over chance (J-energy / (r/d)) everywhere a "fraction in J" is reported. This also defuses Hong's "95% outside", which sits near chance allocation.
5. Single-sample noise is large (κ ≈ 0.6). Primary scoring is teacher-forced paired log-odds (no sampling); free-form readouts use k ≥ 5.
6. **The natural line count is already negative for the NLA on Llama (3%).** Our natural-feature arm includes in-J calibration in the same family (explicit-report line count) so that a null is interpretable.
7. **The pilot's deletion-survival is the exact ambiguity we resolve.** "Survives J-direction deletion" is predicted both by re-solving and by non-J reading. The FollowFeature (text fixed) and FollowText (feature clamped) contrasts separate the two.
