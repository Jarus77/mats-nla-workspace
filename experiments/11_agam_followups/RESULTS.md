# 11 Mentor follow-ups (Agam, 13 Oct 2026) — PILOT / EXPLORATORY

**Status:** Phase A (existing data) complete; Phase B (one Modal job) and Phase C (judge calls) in progress.
Everything here re-reads the declared pilot splits of exp 08/09 (excluded from the main analysis) and the
WorkspaceBench readouts of exp 01/06, plus one new pilot job on the same pilot items. Nothing changes a
primary, direction, dose or template of the pre-registration (rev 4.2); the one proposed change (T1 k=4 → 8
with a consensus rule) goes into rev 4.3 before the 18 Oct freeze. Disclosure: all analyses were decided after
seeing the pilot results.

Agam's six points and where each is answered:

| # | Point | Answer | Section |
|---|---|---|---|
| 1 | OLens can't really read L42; check 44 and other layers | OL@40/44 look like OL@42 on every pilot cell; trained-grid sweep in Phase B | A1, B1 |
| 2 | Sample OL/NLA several times, LLM consensus | Nothing to consense on for col (regex 0–2%); judge-based consensus on k=8 in Phase C | A2, C1 |
| 3 | Full stops etc. where OL and NLA diverge | OL's continuation channel is not punctuation-driven; OL's tense-token channel is partly boundary-driven, the model's own readout is not | A3, B1 |
| 4 | Side-by-side OL / NLA / J-Lens readouts | `results/sidebyside.md` (exp 01, 08, 05, 10); OL text for 05/10 and steered J-Lens lists from Phase B | A4, B2 |
| 5 | Probe accuracy vs when readers show the concept, by position/layer | 40/42/44 table from exp 08; full layer × position map in Phase B | A5, B1 |
| 6 | More examples of NLA/OL hallucinating the same way | `results/catalogue.md`: 75 WorkspaceBench items where both readers are contradicted; 12/12 planted rewrites; 588/600 misquoted arithmetic expressions; same-wrong-content judge in C2 | A6, C2 |

Files: `a11_lib.py` (pure helpers, `test_a11_lib.py`), `a11_analyze.py` → `results/phaseA.{json,md}`,
`a11_sidebyside.py` → `results/sidebyside.md`, `results/catalogue.{md,json}`, `a11_modal_rfm.py` (+ `env.py`,
`a11_modal_lib.py`, `test_a11_modal_lib.py`, `spawn_driver.py`, `watch.py`) → `rfm-results:/11_agam_followups/*`
→ `results/phaseB/`, `a11_phaseB_analyze.py` → `results/phaseB.{json,md}`, `a11_consensus.py` → `results/c1_consensus.json`,
`results/c2_shared_wrong.json`.

---

## A1. Oracle Lens at layers 40 / 42 / 44 (point 1)

**Setup.** Exp 08 col pilot (60 items, pilot split): the same steered vectors were scored by OL at 40, 42 and 44
(each layer's own J-space and directions), descriptive templates (count_qual, count_num, decision), centred
construction, doses q90/q99. At 42 every cell has its matched null band (K=20 iso + K=20 J-complement). At
40/44 only the jrem1 null blocks were scored, so the jrem4 and raw cells there are compared against those
blocks (equal or larger norm → conservative z, marked "proxy").

**Result.** OL@40 and OL@44 are indistinguishable from OL@42 on the descriptive channel: every cell at every
layer has AUROC 0.46–0.53 and |z| ≤ 1.3 (count_qual jrem4 q90: 0.49 / 0.49 / 0.51 at 40 / 42 / 44; decision
raw q90: 0.53 / 0.50 / 0.51). Full table in `results/phaseA.md`. Calibration (exp 08 calib, raw directions):
OL reads language at 40 / 42 / 44 at AUROC 0.77 / 0.76 / 0.77 and tense (label template) at 0.55 / 0.55 / 0.52,
so layer 42 is not a weak spot for OL relative to its trained neighbours. On WorkspaceBench (exp 06), OL@40 /
42 / 44 pass 0.37 / 0.40 / 0.39 pooled; the exp 01 per-layer curve rises from 20 to 56 on every family
(basic_readout 0.33 → 0.80), which is why best-of-11 inflates OL's published number.

**What is still missing** (Phase B): OL's *continuation* channel (nl_cont, the P2 cell) and the tense f07 cell
were only ever scored at 42 (exp 09, round 5). B1 scores both on OL's trained grid 36–52 with per-layer
J-removed directions and K=10+10 nulls.

## A2. Consensus over k samples (point 2)

**Existing data.** Exp 08 stored k=4 free readouts per item per condition (NLA 17 conditions, OL 7). A
"line-position mention" (strict regex: characters / line length / wrap / line break / newline in the
count-or-line sense) occurs in 0.0–0.8% of samples in every condition for both readers, at most 1 of 60 items
per condition, and never with ≥ 2 of 4 samples except one NLA item at jrem1 q90 (+). Exp 07's k=8 unsteered
readouts give self-consistency 0.99–1.00 for the same property, i.e. the readers consistently *do not* mention
line position. Consensus cannot recover a signal that no sample carries.

**Therefore** the consensus analysis Agam asks for has to be claim-based, not regex-based: Phase B generates
k=8 readouts at the P1/P2 cell (zero, ±Δ) for NLA@42, OL@42 and OL@44, and C1 asks a judge to list each
sample's line-position claims and count how many of the 8 samples support each claim (Kenney's ≥ 2/8 rule).
Proposal for rev 4.3: T1 k=4 → k=8 with the ≥ 2/8 consensus rate as a declared analysis beside the per-sample
rate (judge instruction unchanged).

## A3. Punctuation context at the read token (point 3)

**Setup.** Each pilot item's text ends at the read token. We classified the context (ends with a sentence-final
stop / other punctuation / a word; a sentence boundary within the last 12 characters) and recomputed the
paired shift (LO(+) − LO(−)) per stratum for the two divergent OL channels.

**col (round 5, L42, 60 items; strata 4 punct / 56 word, 4 sentence-start).** The OL continuation shift
(nl_cont, J-removed q90) is 1.72 nats overall, 1.72 in both "ends" strata and 2.02 / 1.70 by sentence start;
NLA m2's family (f07_count) is 0.35 overall, 0.29 / 0.35. Neither channel's response to col is punctuation-
driven; the strata are too small (n = 4) to say more.

**tense (exp 09, L42, 80 items; 11 stop / 3 punct / 66 word; 12 sentence-start).**

| cell (J-removed, q90) | all | ends=stop | ends=word | sentence-start | not |
|---|---|---|---|---|---|
| OL tense_tok ("- was" vs "- is") | 0.34 | 0.58 | 0.28 | 0.68 | 0.28 |
| OL f07_tense ("- They were" vs "- They are") | 2.02 | 2.09 | 1.99 | 2.24 | 1.98 |
| model next-token tense readout | −0.02 | −0.02 | −0.02 | −0.02 | −0.02 |

At r108 the pattern is the same (tense_tok 1.06 vs 0.51; f07 3.74 vs 3.69; model −0.04). So: OL's bare
tense-token channel is about twice as strong when the read token sits at a sentence boundary, where the
model's next token is likely punctuation rather than a verb, while its sentence-template channel (f07, the
P3 cell) is uniform, and the model's own next-token tense readout is flat in every stratum. The divergence
between OL and the model on tense 4r\* is therefore not explained by punctuation; part of the *tense_tok*
channel is. Phase B adds the model's actual next-token punctuation mass per item (B1 captures logits) and
re-splits the P3 cell by it.

## A4. Side-by-side readouts (point 4)

`results/sidebyside.md`: six seeded WorkspaceBench items (NLA@42, OL@42/40/44, J-Lens@42 top-10), six col
items at zero / +Δ / −Δ (NLA 2 of 4 samples, OL 2 of 4, J-Lens natural top-10), all 12 planted-bridge NLA
pairs, six arithmetic prompts (Kenney's NLA readouts vs the probe). Phase B adds OL bullets for the planted
and operand vectors (at 42 and 44) and steered J-Lens token lists; `a11_sidebyside.py` is reusable on Guiv's
section-2 items (NLA / OL / J-Lens lists keyed by item id).

## A5. Probe accuracy vs reader detection by layer (point 5)

**Caveat first.** On steered pairs the ridge probe scores AUROC 1.00 for col at every layer by construction
(the steering direction is the probe's own covariance direction), so "fraction of probe" is uninformative
there. The meaningful probe number is natural decodability per layer:

| feature (exp 02, L40 / 42 / 44) | probe | J-Lens reads it? | OL / NLA track it (exp 08 calib, raw)? |
|---|---|---|---|
| col (R²) | 0.58 / 0.66 / 0.65 | no at any layer (EOL AUROC 0.50–0.53 on J-removed; 0.73–0.84 on raw = the J part) | no (0.46–0.53) |
| tense (acc) | 0.84 / 0.84 / 0.85 | raw 0.79–0.88; J-removed 0.50 | OL label 0.55 / 0.55 / 0.52; f07 at 42 ≥ 0.9 |
| language (acc) | 0.96 / 0.97 / 0.97 | raw 0.68–0.76 | OL 0.77 / 0.76 / 0.77 |

Exp 04 features at 38–44 (`phaseA.json: natural_decodability_by_layer`) are flat across layers too (e.g.
capitalisation 0.85 → 0.87). Within 40–44 nothing changes: what a probe can decode is constant, and which
readers report it is constant. The rank correlation between probe AUROC and reader AUROC across the exp 08
cells is undefined for the NLA (probe constant) and ≈ 0 for OL's descriptive families, +0.77 for OL's label
family on the calibration features (n = 12). The layer × position map that actually answers the question
(layers 24–56, four positions, cross-validated probes on natural items, J-Lens, OL on its grid, NLA at 42) is
Phase B.

## A6. Hallucination catalogue (point 6)

`results/catalogue.md`. From the exp 06 claim-level judge (100 WorkspaceBench items): 97/100 NLA readouts and
77/100 OL readouts contain at least one claim contradicted by the text; on 75 items both do. Patterns:
re-imagined prompt (the readout quotes a text that differs from the real one: NLA "The text is 'The built up
area covering north-east England…'"; Kenney's arithmetic bank, where 588 of 600 NLA readouts quote a
different expression than the prompt, e.g. prompt `51 − 49 =` → "64 − 49 ="), confabulated specifics (names,
sources, numbers), format-driven claims (quiz / trivia framing that the text lacks), and shared wrong answers
(both readers name the same non-gold entity, e.g. "hypothermia" for hyperthermia, "absorption" for
adsorption). Whether the shared-wrong rate exceeds a mismatched-item baseline is C2.

---

## B. Phase B (Modal job `agam_main` + `agam_nla`, A100-80GB)

**Setup.** The 60 col and 120 tense pilot items (same items as exp 08 / round 5) captured at layers
24–56 (step 4, plus 42) and at the read token and the three tokens before it; per-layer J-space bases
(03's at 40/42/44; computed here for the others, same SVD definition); class-mean directions per
(layer, position) (long vs short line by median split; Past vs Pres), J-removed at 1536; dose q90 − median
of the items' own projection; K = 10 isotropic + 10 J-complement nulls at that dose; the model's next-token
distribution at the read token (punctuation and newline mass). Probes are 5-fold grouped cross-validated
logistic probes **on these 60 / 120 items only**, so they are far noisier than exp 02's 40k-row probes
(col 0.42–0.77 across cells, tense 0.65–0.83); treat them as a weak decodability signal, not as G1.

### B1. Where col leaves the J-space (map, `results/phaseB.md`)

| layer | col direction enrichment r\*=384 / 4r\*=1536 (read token) | J-Lens leak z of the J-removed col push | tense enrichment |
|---|---|---|---|
| 24 | 3.67 / 1.79 | 0.0 | 3.38 / 1.62 |
| 32 | 2.78 / 1.71 | −0.2 | 3.87 / 1.82 |
| 40 | 1.76 / 1.46 | 0.2 | 3.33 / 1.72 |
| 42 | 1.54 / 1.29 | 0.0 | 3.49 / 1.76 |
| 44 | 1.46 / 1.17 | 0.4 | 3.34 / 1.68 |
| 48 | 1.45 / 1.00 | 0.2 | 3.14 / 1.57 |
| 56 | 1.42 / 0.96 | **19.4** | 2.20 / 1.18 |

The class-mean col direction is strongly J-enriched at early layers and falls to about the ≤ 1.5 gate only at
layers 42–48 (42 is 1.54 at r\* with this direction; 52 rises to 1.74, and its push leaks); tense stays 3–4× enriched until 52 and drops at 56. The J-removed col push is leak-free at
every layer up to 48 and leaks massively into the J-Lens at 56 (close to the unembedding, where the
top-1536 directions no longer contain the newline readout). So "col is non-J" is a property of the band
around the readers' layer, not of the model as a whole; at 24–36 the same feature sits inside the J-space.
The positions −2…−4 give the same enrichment picture (`phaseB.md`). The model's next token at the col read
token is punctuation with mean mass 0.12 and a newline with 0.14; at the tense read token punctuation 0.14,
newline 0.02.

**Direction caveat for the sweep.** The per-(layer, position) directions are class means over 60 (col) or
120 (tense) items. At 40 / 42 / 44 their cosine with the bank-based directions exp 08 used (02's regression
direction on 40k rows) is only 0.38–0.41 for col (raw and J-removed alike) and 0.50 / 0.36 for tense. The
sweep therefore steers a noisier version of each feature and understates every reader's response; its value
is the *relative* trend across layers at a fixed construction, not the absolute z. For the P2 channel itself a
separate pass (`agam_ol_bank`) rescores OL at 40 / 42 / 44 with the bank directions, the bank q90 doses and
the 08 null vectors, which is the apples-to-apples answer to point 1.

### B2. Point 1 answered on the P2 cell: OL's continuation channel at the trained layers 40 / 44 (bank directions)

`agam_ol_bank`: exp 08's spec (02's regression direction per layer, J-removed at 4r\*(L), bank q90 dose, the
08 null vectors, K = 20 + 20), OL scored at 40, 42 and 44 with the P2 templates (nl_cont) and the fixed
descriptive template. Same 60 pilot items.

| OL layer | nl_cont, J-removed 4r\* q90: AUROC / win / shift / z | nl_cont, raw: AUROC / shift / z | fixed descriptive, J-removed: AUROC / z |
|---|---|---|---|
| 40 (trained) | 0.65 / 0.98 / +1.48 / **+4.8** | 0.76 / +3.02 / +9.7 | 0.50 / −0.0 |
| 42 (off-grid, round-5 reference 0.67–0.72, z +3.9) | 0.67 / 0.98 / +1.69 / **+5.3** | 0.76 / +2.90 / +9.2 | 0.50 / −0.2 |
| 44 (trained) | 0.72 / 0.97 / +2.49 / **+8.3** | 0.74 / +2.79 / +9.2 | 0.51 / −0.0 |

The P2 channel is not an artefact of reading OL at an untrained layer: at layer 40, which OL was trained on,
it responds to the J-removed col push as strongly as at 42, the descriptive template is flat at both, and
layer 42 reproduces round 5 within noise. **Layer 44 is void under the leak rule:** 4r\*(44) = 1024 (r\*(44) =
256 from 02's rank calibration) removes fewer J directions, and exp 08's own leak check marks the jrem4 q90
dose at 44 as leaking (J-Lens EOL set z +2.6 vs the J-complement band; 40 and 42 are clean at −0.3 and
+0.1). So OL's larger response at 44 (z +8.3) cannot be read as non-J reading; the trained-layer evidence for
point 1 rests on 40 vs 42. (Run time 55 min on one A100-80GB.)

### B3. Oracle Lens on its trained grid, sweep directions (`results/phaseB.md`)

Same construction at every layer: the layer's own class-mean direction (see the direction caveat above), J-removed
at 1536, q90 dose, K = 10 + 10 nulls, read token (position −2 as a check at 42 / 44). z = mean-shift z against the
pooled bands; "leak z" = J-Lens shift of the same push vs the J-complement band (void if > 3).

| OL layer | col nl_cont, J-removed: AUROC / shift / z | col leak z | tense f07, J-removed: AUROC / shift / z | tense raw AUROC | tense leak z |
|---|---|---|---|---|---|
| 36 | 0.56 / +0.58 / +1.7 | −0.1 | 0.57 / +0.91 / **+5.3** | 0.95 | +0.2 |
| 40 | 0.55 / +0.51 / +1.5 | +0.2 | 0.60 / +1.23 / **+6.7** | 0.85 | +1.5 |
| 42 | 0.56 / +0.52 / +1.6 | 0.0 | 0.60 / +1.16 / **+9.1** | 0.85 | +0.5 |
| 44 | 0.55 / +0.51 / +2.4 | +0.4 | 0.62 / +1.33 / **+7.3** | 0.84 | +1.1 |
| 48 | 0.55 / +0.56 / +0.9 | +0.2 | 0.59 / +1.03 / **+8.9** | 0.81 | +1.5 |
| 52 | 0.50 / +0.06 / +0.3 | **+6.2 (void)** | 0.65 / +1.59 / **+9.9** | 0.83 | +1.4 |

Position −2 (the token before the read token): col at 42 z +0.1, at 44 z +3.2; tense at 42 z +13.4, at 44 z +7.5.

Reading. (i) **Tense's J-removed tail is read by OL at every trained layer from 36 to 52**, leak-clean, with
the same magnitude (z +5 to +10); the P3 cell is not a layer-42 artefact, and the raw tense direction is read
at 0.81–0.95 everywhere. (ii) With the noisier sweep directions the col continuation channel is a small,
consistently positive shift (+0.5 nats, z +1.5 to +2.4) at 36–48; with the bank directions (B2) it is z +4.8
to +5.3 at 40 / 42. Layer 52's J-removed col push leaks (z +6.2), so its null is void: by 52 the top-1536
J-space no longer removes the newline readout, matching the leak at 56 in B1. (iii) The rank correlation
between the per-cell probe AUROC and OL's J-removed z is +0.82 across the 16 OL cells, but this is driven by
the feature contrast (tense has both the higher probe AUROC and the higher z); within col or within tense it
is not meaningful at n = 6–8.

**Punctuation mass (point 3, closing the loop).** Splitting the P3 cell (OL f07, J-removed, layer 42) by
whether the model's actual next token is mostly punctuation (p_punct > 0.5; 16 of 120 items): shift 1.34 vs
1.14 nats, win 1.00 vs 0.97. The sentence-template channel is not punctuation-driven; only the bare
tense-token channel (A3) was.

### B4. Free-text side-by-sides and two sanity numbers (point 4 / 6; `results/sidebyside_B.md`)

Oracle Lens free text (k = 4, WorkspaceBench sampling) on the exp 05 planted-bridge vectors at 42 and 44, on
120 of the exp 10 operand vectors, and k = 8 on the col P2 cell at 42 and 44; J-Lens top-10 for the same vectors.

| check | 42 | 44 |
|---|---|---|
| OL names the planted country Y, planted vectors (any of 4 samples) | 0.965 | 0.955 |
| OL names the original X, natural vectors | 0.985 | 0.980 |
| OL names Y on natural / X on planted | 0.000 / 0.005 | 0.005 / 0.005 |
| J-Lens top-10 contains Y: natural / planted | 0.005 / 0.87 | – |

The free text reproduces the log-odds result of exp 05 (p(reads Y) 1.00 planted / 0.005 natural) and is the
same at the trained layer 44. On the operand vectors OL writes an arithmetic expression in 120 / 120 items but
quotes the prompt's operands correctly in **7 / 120 (5.8%)**; the J-Lens top-10 never contains a digit. So
both prose readers misquote the operands (Kenney's NLA: 5% / 38% per operand; our catalogue: 588 / 600 NLA
readouts quote a different expression), which is point 6 in one number: the readers hallucinate *the same
kind* of thing (a plausible expression) while rarely the same wrong numbers.

### B5. NLA at layer 42: position check (point 5) and the sweep-direction cells

The NLA reads only at block 42, so the layer axis does not apply; the position axis does. Template LO on the
natural vectors at the read token and the three tokens before it, scored against the label (col: long vs
short line; tense: Past vs Pres), and the same steering cells as B3 at positions −1 and −2.

| feature, NLA family | natural AUROC vs label at pos −1 / −2 / −3 / −4 | J-removed steer, pos −1: AUROC / shift / z | pos −2 | OL at the same cells (z) |
|---|---|---|---|---|
| col, m2 (P1 template) | 0.51 / 0.51 / 0.56 / 0.46 | 0.52 / +0.05 / +0.7 | 0.53 / +0.14 / +0.7 | +1.6 / +0.1 |
| tense, f07 (P4 templates) | 0.85 / 0.81 / 0.80 / 0.80 | 0.56 / +1.00 / **+4.1** | 0.55 / +0.82 / **+5.5** | +9.1 / +13.4 |

Reading. The NLA reads tense from any of the four positions' vectors (0.80–0.85; the feature is spread across
the clause) and follows the tense tail at both steered positions, at about half OL's z. On col it is at
chance at every position, natural or steered. The col steering z here (+0.7) is lower than round 5's +3.5
for the same P1 template because the sweep direction has cosine 0.41 with the bank direction (B1 caveat);
the main run uses the bank direction. Position −2 adds nothing for col and slightly more for tense, so the
pre-registered read token stays.

### B6. Phase B cost

| function | GPU time | ≈ $ (A100-80GB list) |
|---|---|---|
| `agam_main` (capture, bases, probes, J-Lens, OL sweep 16 cells, OL text 6,560 generations) | 12,401 s = 3.44 h | ≈ 9 |
| `agam_ol_bank` (OL P2 cell at 40/42/44, bank directions) | 3,285 s = 0.91 h | ≈ 2.5 |
| `agam_nla` (template LO at 4 positions, 1,440 explanations) | 5,203 s = 1.45 h | ≈ 4 |
| total | 5.8 A100-h | **≈ 15–18** incl. container start-up |

## C. Phase C (judges)

### C2. Do the two readers assert the *same* wrong content? (point 6)

`a11_consensus.py shared` → `results/c2_shared_wrong.json`. For each of the 75 WorkspaceBench items on which
both NLA@42 and OL@42 have a claim contradicted by the text (exp 06 judge), a judge saw the text, the gold
answer and the two readers' contradicted claims and said whether they assert the same wrong content. Baseline:
the same question with the OL claims taken from a different item of the same family.

| judge | matched items: same wrong content | mismatched baseline | parsed |
|---|---|---|---|
| Gemini 3.8 Flash (T=0) | **12 / 74 = 0.16** | 0 / 75 = 0.00 | 74 + 75 |
| kimi-k3 (second judge) | 2 / 7 = 0.29 | 0 / 29 | 7 + 29 of 150 (most kimi responses did not parse as JSON with or without the schema; reported for completeness only) |

Examples of shared wrong content: "carbonara" (gold amatriciana), "adrenal gland" (gold thyroid), "mint"
(gold chamomile), "tuatara", "the artist is female", "Pope Gregory VII and Henry IV" (gold Concordat of
Worms), "the kwacha" (gold Mozambique's metical). So on this benchmark the readers' errors are mostly
*different* errors: 16% of co-failing items share the wrong content, against 0% for mismatched items. That
is a real but minority overlap, consistent with Guiv's section-1 "same wrong answer 2× chance" on
two-candidate scoring (there the candidate set forces shared errors onto one alternative; in free text they
spread). Spend $0.06 + $0.23.

### C1. Consensus over k=8 samples (point 2)

`a11_consensus.py consensus` → `results/c1_consensus.json`. For each of the 60 col pilot items × 3 conditions
(zero, +Δ, −Δ on the J-removed 4r\* col direction at q90) × 3 readers (NLA@42, OL@42, OL@44), the 8 free
readouts were shown together to Gemini 3.8 Flash (T = 0) with the T1 instruction's scope (claims about
position within a line, line length, wrapping or an upcoming break in the passage being described; the
readouts' own formatting never counts). The judge marks each sample yes/no and lists each distinct claim with
the samples that support it. 540 judge calls, 4,320 samples, $0.66, 0 failed.

| reader | condition | samples with a line-position claim | items with any | items with ≥ 2 of 8 (consensus) | items with the *same* claim in ≥ 2 samples |
|---|---|---|---|---|---|
| NLA@42 | zero / +Δ / −Δ | 0 / 0 / 1 of 480 | 0 / 0 / 1 of 60 | **0 / 0 / 0** | 0 / 0 / 0 |
| OL@42 | zero / +Δ / −Δ | 0 / 1 / 0 of 480 | 0 / 1 / 0 | **0 / 0 / 0** | 0 / 0 / 0 |
| OL@44 | zero / +Δ / −Δ | 0 / 0 / 0 of 480 | 0 / 0 / 0 | **0 / 0 / 0** | 0 / 0 / 0 |

Two of 4,320 samples carry a line-position claim (one NLA at −Δ, one OL@42 at +Δ, opposite signs); no item has
two supporting samples; the + vs − McNemar count is 0 / 0 for every reader. Consensus sampling therefore
confirms the single-sample null rather than rescuing a signal: with eight draws and a judge instead of a regex,
neither decoder describes line position at the P1/P2 cell, steered or not. This is the predicted T1 outcome at
k = 8 (prereg rev 4.3: consensus rate < 2%); it is a pilot number on 60 items and the main run tests it on 120
fresh passages. The template log-odds channels (P1 m2, P2 nl_cont) are what carry the pilot's candidate
signal, not anything the readers say in free text.

## D. Readout viewer (Agam, round 2) — PILOT / EXPLORATORY

Agam asked for side-by-side readouts with the full prompt across layers and positions, interactive if possible.
`viewer/template.html` + `a11_viewer_data.py` + `viewer/build_viewer.py` build `results/readout_viewer.html`, a
single self-contained page: WorkspaceBench (48 items, 12 per family; Oracle and J-lens at layers 20–60, NLA at 42,
last token), col and tense (12 + 12 pilot items, unsteered, from `agam_grid`: J-lens at all 10 captured layers x
positions −4…−1, Oracle k=2 at 36–52 x the same positions, NLA k=2 at 42 x the same positions; col also shows the
± J-removed edit at 42 / 44 with the Phase B k=8 samples), planted bridge (24 items, Oracle k=4 at 42 / 44, J-lens,
NLA for the 10 items with a stored explanation) and arithmetic (24 '=' prompts). Samples are the first generated;
items are seeded draws (col evenly spaced over the col value, tense 6 past / 6 present). `agam_grid` reuses the
Phase B activations (`acts.pt`); nothing is scored, so it changes no number above.

One count from the grid: across all layers and positions, 0 of 576 Oracle and 0 of 96 NLA col readouts match exp 08's
strict line-position regex (the same holds for the tense items). Run time 1,015 s on one A100-80GB.

## Cost ledger

| item | GPU / API | ≈ $ |
|---|---|---|
| Phase A | none | 0 |
| Phase B | agam_main + agam_ol_bank + agam_nla, 5.8 A100-80GB-h | ≈ 15–18 |
| Phase C | Gemini 3.8 Flash (+ kimi-k3 for C2) | ≈ 2–4 |
| Phase D | agam_grid (viewer readouts), 1,015 s ≈ 0.3 A100-80GB-h | ≈ 1 |
