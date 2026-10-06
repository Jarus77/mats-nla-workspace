# 09: OL mechanism (S6 a/b/c): REPORTING vs RE-INSTATEMENT — PILOT, EXCLUDED FROM MAIN ANALYSIS

**This is a declared pilot split and is excluded from the main analysis.** It uses fresh items: col09 (80),
tense09 (80), codeNL (80), codeIN (80), codeTR (400, direction fitting only) and sel09 (60, random-direction
selection only). Their ids are in `pilot_split.json`. Per prereg §3.5 S6, these numbers are *cited* in §14
and do **not** change S6's predictions or the §10.1 rule. The binding S6 run is on fresh items (sprint D6).

Full output: `results/o9_stdout.txt` and `results/o9_metrics.json`, from `o9_analyze.py`
(`../../.venv/bin/python o9_analyze.py`; CPU only, deterministic seeds).

## TL;DR

- **Literal §10.1 verdict: S6 = READING.** But the verdict is an **artefact**: it fires through rule (b)
  only because tense 4r\*'s gain ratio is undefined. The model's own tense readout does not move under
  the tense 4r\* edit (−0.04 nats, z −1.4/−2.8), so its ratio is −15. That drags the "mean of the other
  five" down to −1.51, and col 4r\* (1.06) "exceeds" it.
- **Dropping the one undefined ratio** (a model shift that is not ≥ +3 z, the same requirement the
  prereg places on the random directions) gives **S6 = INCONCLUSIVE.**
  - Not acting: the ratios are 0.99 / 1.06 / 3.40 / 1.65 / 1.89, far from ±25% of one value.
  - Not reading: indentation stays below +3 z, and col 4r\*'s ratio is the *lowest* of the set, not
    the highest.
- **col specifically looks like re-instatement.**
  - OL's nl_cont shift equals the model's own newline shift, 1:1. The ratio is 0.99 for raw and 1.06
    for 4r\* at r108, and 0.96 / 1.12 at q90.
  - J-removal keeps 0.79 of the per-unit effect for OL and 0.75 for the model. This replicates R5's
    78% / 70%.
  - The effect vanishes with the adapter off.
- **(a) Indentation does not fire.**
  - The largest cell is codeIN indent_raw at r108: z +2.7 / +2.6, while the model's own indentation
    readout stays flat.
  - The trend is reading-like but sub-threshold, and it needs the LoRA (adapter off: flat).
- **(c) With the adapter off, tense collapses as col does.**
  - f07_tense goes from 0.65–1.00 to 0.49–0.51; tense_tok 4r\* goes from z +7 to 0.
  - tense_tok raw keeps a tiny base-Qwen residue (+0.06 nats vs +1.85, z +3–5).
- **New and important for P3:** OL tracks tense 4r\* strongly (f07_tense z +12 to +16, tense_tok
  z +7), yet the model's next-token tense readout does not move under that edit. Prereg §4.4.5 applies.
- **Implied P2 wording.**
  - Literal rule: "reading", i.e. "OL carries non-J col content in its continuation channel".
  - On substance, the col evidence points the other way (acting). The recommended reading is the
    **inconclusive** row.
  - Before the freeze, rule (b) needs a valid-denominator clause. This is informed by pilot data and
    must be disclosed (see the end of this file).

## Design (what was run)

- **Construction.** Read layer L42, centred: h − (v·h)v ± Δv.
- **Doses.**
  - q90 is the bank constant: col reg_raw 9.465, reg_jrem4 7.098; tense raw 8.392, jrem4 5.228.
  - The relative doses are r054 / r108 / r216 = 0.054 / 0.108 / 0.216·‖h_i‖ (mean ‖h‖ 81–89).
- **Null bands.** K = 20 isotropic and K = 20 J-complement (nonj4, ⟂ top-1536 J) vectors, the 08 pilot
  set. They exist at r108 in every set, and at q90 for col and tense (matched to each direction's
  constant). r054 and r216 have no matched null.
- **OL channels.** These are fixed in `o9_templates.py` before any 09 scoring. The candidate pairs are
  scored teacher-forced as OL's first `- ` bullet, and LO = mean over pairs of
  log p(a) − log p(b).
- **Model readouts.** These use the same tokens at the model's own next position after the read token.

| test | items | direction(s) | OL channel (analogue) | model readout |
|---|---|---|---|---|
| (a) | codeNL (read = "\n", next line indented 4/8) | indent_raw, indent_jrem4 (04 gate dirs); nextind (exploratory) | ind_cont (`-` + 8 vs 4 spaces + word) | ind8v4 (7 vs 3 spaces) |
| (a) | codeIN (read = last token of an indented line; "\n" appended) | same | ind_nlcont (`- \n` + 8 vs 4 sp + word) | ind8v4 |
| (b)/(c) | col09 | reg_raw, reg_jrem4 | nl_cont (`- \nthe` vs `- the`, and/of) | nl_vs_content (newline vs the/and/of) |
| (b)/(c) | tense09 | raw, jrem4 | tense_tok (`- was/were/had` vs `- is/are/has`); f07_tense | tense_tok (same tokens); tense_broad (02 G4 10 vs 10 verbs) |
| (b) | col09 + tense09 pooled | randA–D (fixed rule, `random_selection.json`) | `-` + top-3 vs bottom-3 token | same tokens |
| (c) | r5 tense120 (08 round 5 activations) | raw, jrem4 at q90/r11 | f07_tense, tense_tok, adapter off; on = re-score | — |

Cell format (as in 08 round 5): AUROC [95% CI] / paired win / mean shift (nats, mean of p − m) /
z_shift (iso, Jc) / z_AUROC (iso, Jc). \* = outside the two-sided 95% band.

**Analysis choices made here** (the prereg does not fix them; they are stated in the
`o9_analyze.py` docstring):
1. **Gain-ratio dose = r108.** It is the only relative dose with a null block in every set, and it
   matches relative norm across all six directions. q90 (col/tense) and r216 (random) are
   sensitivity analyses.
2. **Random directions are pooled over col09 + tense09** (160 items). The two used are the two with
   the largest behavioural z: **randC and randD** (the isotropic pool, amendment A1).
   - randA and randB (nonj pool) have weak behaviour: iso z +1.3 to +2.1.
   - This matches A1's observation on sel09.
3. **S6(a) "fires"** if any of the four primary cells has z_shift ≥ +3 against both bands. The four
   cells are indent_raw and indent_jrem4 at r108, each with its set's analogue channel.

## Sanity checks (known positives)

| cell (col09, n=80 fresh) | 09 | 08 round 5 (60 pilot items) |
|---|---|---|
| OL-on nl_cont reg_raw q90 | 0.69 [.66,.74] / 0.93 / +2.53; +5.6\* +5.8\* / +5.2\* +5.3\* | 0.76 / 0.97 / +2.88; +6.6\* +6.7\* |
| OL-on nl_cont reg_jrem4 q90 | 0.63 [.61,.66] / 0.99 / +1.60; **+4.7\* +5.0\*** (matched q90 null) | 0.67 / 0.98 / +1.72; +3.9 +3.9 (proxy, larger-norm null) |
| OL-on nl_cont reg_raw / reg_jrem4 at r108 (vs r5 r11) | 0.70 / +2.60; 0.66 / +2.07 | 0.77 / +3.00; 0.72 / +2.32 |
| model newline log-odds, (p−m)/2, raw / 4r\* q90 | +1.48 (z +64/+60) / +0.79 (z +42/+43) | +1.44 / +0.72 |
| r5 tense120 f07_tense, adapter-on re-score, raw q90 / jrem4 q90 / jrem4 r11 | 0.99 / +7.59; 0.66 / +1.93; 0.79 / +3.55 | 0.99 / +7.60; 0.66 / +1.94; 0.79 / +3.54 (max item \|Δ\| 0.10–0.15 nats) |

- **The known positive reproduces on fresh items.** It is 7–12% smaller in shift, and AUROC is
  0.04–0.07 lower: the raw q90 CI [.66, .74] just excludes 0.76.
  - The z values are similar. jrem4's z is larger here because 09 has a matched q90 null, where R5
    used a conservative proxy.
  - This is within what a new item sample plus batch-composition noise (D5: 0.26–0.64 nats) can
    produce. It is not a failure.
- **The model's behaviour reproduces within 0.07 nats.**
- **The r5 tense re-score reproduces R5.5 exactly**, to two decimals and the same z.

## (a) INDENTATION: does OL follow a direction the model does not act on?

OL-on uses the analogue channel; the model readout is ind8v4. Adapter-off is shown for the same
channel. Only r108 has a null.

| cell | model ind8v4 | OL-on analogue | OL-off analogue |
|---|---|---|---|
| codeNL indent_raw r108 (ind_cont) | 0.50 / 0.52 / +0.03; +0.3 +1.5 | 0.57 [.55,.60] / 0.81 / +0.64; **+1.9\* +2.3\*** / +1.6 +2.0\* | 0.49 / 0.53 / −0.01; −0.3 −0.3 |
| codeNL indent_jrem4 r108 (ind_cont) | 0.50 / 0.52 / +0.03; +0.3 +1.5 | 0.52 [.50,.54] / 0.61 / +0.18; **+0.4 +0.6** | 0.46 / 0.45 / −0.03; −0.9 −0.7 |
| codeIN indent_raw r108 (ind_nlcont) | 0.50 / 0.56 / +0.01; +0.7 +0.9 | 0.64 [.61,.68] / 0.88 / +0.72; **+2.7\* +2.6\*** / +2.6\* +2.6\* | 0.55 / 0.62 / +0.03; +1.0 +0.7 |
| codeIN indent_jrem4 r108 (ind_nlcont) | 0.50 / 0.61 / +0.03; +1.7\* +1.9\* | 0.61 [.58,.64] / 0.88 / +0.53; **+2.0\* +1.9\*** | 0.59 / 0.70 / +0.06; +1.8\* +1.6\* |
| *dose scaling, OL-on (no null)*: codeNL raw r054 / r216 | +0.02 / +0.07 | +0.33 / +1.16 (0.54 / 0.64) | −0.00 / −0.00 |
| codeIN raw r054 / r216 | −0.00 / +0.06 | +0.37 / +1.27 (0.58 / 0.73) | +0.02 / +0.05 |
| *exploratory*: codeNL nextind_raw r108 | +0.55; +16.5\* +21.3\* | +2.02; +6.5\* +7.3\* | −0.07; −1.7 −1.3 |
| codeNL nextind_jrem4 r108 | +0.24; +6.8\* +9.4\* | +0.22; +0.5 +0.7 | −0.05 |
| codeIN nextind_raw / jrem4 r108 | +0.11 (+7.2\*) / +0.05 (+3.1\*) | +1.44 (+5.4\*) / +0.38 (+1.4) | +0.03 / +0.06 |

- **S6(a) does not fire.** No primary cell reaches +3 z on both bands; the maximum is codeIN
  indent_raw at +2.7 / +2.6.
- **The pattern is reading-like but sub-threshold:**
  - OL's indentation continuation moves monotonically in dose: shift +0.33 → +0.64 → +1.16 (codeNL
    raw) and +0.37 → +0.72 → +1.27 (codeIN raw), with AUROC 0.73 at r216.
  - The model's own next-indentation readout stays flat throughout (≤ +0.07 nats; |z| ≤ 1.9 at r108).
  - The other OL indentation channels agree in sign (ind_any +1.5 to +2.9 z) and are also
    sub-threshold.
- **The OL indentation response needs the LoRA**, as col does: adapter off is flat for codeNL, and
  near zero in shift for codeIN.
- **J-removal kills most of it in codeNL** (+0.4 / +0.6 z). It does so less in codeIN (+2.0 / +1.9).
- **Opposite dissociation (exploratory).** nextind_jrem4 moves the model (z +6.8 / +9.4 codeNL)
  while OL's ind_cont does not (+0.5 / +0.7). OL does not simply follow everything the model acts on
  either.

## (b) GAIN RATIO (OL continuation shift / model behavioural shift; adapter ON; per unit dose)

| direction | dose | OL shift | model shift | **ratio [95% CI]** | OL z (iso/Jc) | model z (iso/Jc) |
|---|---|---|---|---|---|---|
| col raw | r108 | +2.60 | +2.62 | **0.99 [0.83, 1.20]** | +5.8 / +6.0 | +52 / +71 |
| col 4r\* | r108 | +2.07 | +1.96 | **1.06 [0.89, 1.26]** | +4.6 / +4.7 | +39 / +53 |
| tense raw | r108 | +2.07 | +0.61 | **3.40 [2.79, 4.05]** | +23.9 / +22.7 | +20.6 / +41.1 |
| tense 4r\* | r108 | +0.62 | **−0.04** | **−15.5 [−86, −6.8] (undefined: denominator ≈ 0)** | +7.3 / +6.9 | **−1.4 / −2.8** |
| randC (iso) | r108 | +0.45 | +0.27 | **1.65 [1.45, 1.87]** | +3.0/+4.6 (col); +3.2/+4.4 (tense) | +6.9/+19.3; +6.5/+13.6 |
| randD (iso) | r108 | +0.43 | +0.23 | **1.89 [1.69, 2.13]** | +3.8/+4.5; +3.7/+4.8 | +5.1/+10.7; +4.9/+10.3 |
| *sensitivity* col raw / 4r\* | q90 | +2.53 / +1.60 | +2.64 / +1.43 | 0.96 [0.80,1.14] / 1.12 [0.93,1.36] | | |
| col raw / 4r\* | r054 | +1.36 / +1.06 | +1.29 / +0.93 | 1.05 / 1.15 | | |
| tense raw / 4r\* | q90 | +1.85 / +0.34 | +0.54 / −0.02 | 3.46 / −14.5 | | |
| tense raw / 4r\*, tense_broad readout | r108 | +2.07 / +0.62 | +0.52 / −0.05 | 4.00 [3.27,4.87] / −12.8 (model z −1.7/−3.2) | | |
| randC / randD | r216 | +0.89 / +0.85 | +0.54 / +0.45 | 1.64 / 1.87 | | |
| randA / randB (not used: weak behaviour) | r108 | +0.13 / +0.16 | +0.08 / +0.08 | 1.72 [1.28,2.29] / 1.93 [1.43,2.58] | | |

**The rule, applied literally** (six directions; r108):
- Ratios: [0.99, 1.06, 3.40, −15.46, 1.65, 1.89]; mean −1.08.
- **Acting(b) fails:** the deviations are not within ±25%.
- **Reading(b) fires:**
  - col 4r\* = 1.06 [0.89, 1.26] > 1.25 × (mean of the other five = −1.51), with the CI excluding it.
  - This holds at q90 / r108 and q90 / r216 as well.

**Why that is an artefact.**
- The rule's mean includes a ratio whose denominator is zero. Under the tense 4r\* edit, the model's
  next-token tense contrast does not move: −0.04 nats, and z is ≤ 0 on both readouts.
- The prereg already demands "nonzero behavioural effect" for the random directions. The same
  precondition applied to all six (model shift ≥ +3 z on both bands) drops only tense 4r\*:
  - ratios [0.99, 1.06, 3.40, 1.65, 1.89]; mean 1.80;
  - **acting(b) fails** (col −45%, tense +89%);
  - **reading(b) fails:** col 4r\* 1.06 is *below* the others' 1.98.
  - Using the tense_broad readout gives the same answer: mean 1.92, both false.

**What the ratios do say:**
1. **For col, OL's continuation shift is the model's own newline shift, one-for-one**, for both raw
   and J-removed and at every dose (0.96–1.15).
2. **J-removal costs the same fraction on both sides:** per unit dose at r108, 4r\*/raw = 0.79 for
   OL and 0.75 for the model.
3. **The gain is not universal.** Random token pairs are amplified about 1.7–1.9×, and tense raw
   about 3.4–4.0×. So "constant ratio" re-instatement in the strict sense of the rule does not hold
   across directions.
4. **col sits at the low end of the gain range.** It is the direction where OL adds *least* beyond
   the model's own decision.

## (c) ADAPTER OFF (base Qwen in OL's carrier, identical slot injection)

| cell | adapter ON | **adapter OFF** |
|---|---|---|
| col09 nl_cont reg_raw q90 | 0.69 / 0.93 / +2.53; +5.6\* +5.8\* | **0.42 [.38,.46] / 0.33 / −0.10; −2.3\* −2.3\*** |
| col09 nl_cont reg_jrem4 q90 | 0.63 / 0.99 / +1.60; +4.7\* +5.0\* | **0.43 [.40,.46] / 0.25 / −0.09; −3.1\* −2.3\*** |
| col09 nl_cont reg_raw / reg_jrem4 r108 | 0.70 / +2.60; 0.66 / +2.07 | 0.45 / −0.09 (−2.2\* −1.9\*); 0.42 / −0.11 (−2.7\* −2.5\*) |
| tense09 f07_tense raw q90 / r108 | 0.99 / +7.91 (+31\* +40\*); 1.00 / +8.72 | **0.51 / −0.02 (−0.7 −0.3); 0.51 / −0.01** |
| tense09 f07_tense jrem4 q90 / r108 | 0.65 / +2.02 (+12.7\* +16.4\*); 0.78 / +3.71 (+12.5\* +16.4\*) | **0.50 / −0.00 (−0.4 −0.0); 0.49 / −0.01 (−0.6 −0.1)** |
| tense09 tense_tok raw q90 / r108 | 0.83 / +1.85 (+24.7\* +23.3\*); 0.85 / +2.06 | 0.60 / +0.06 (+3.7\* +5.2\*); 0.62 / +0.06 (+3.3\* +4.2\*) |
| tense09 tense_tok jrem4 q90 / r108 | 0.56 / +0.34 (+7.3\* +6.9\*); 0.62 / +0.62 | **0.50 / −0.01 (−0.5 −0.1); 0.48 / −0.01 (−0.4 +0.0)** |
| r5 tense120 f07_tense raw q90 / r11 | 0.99 / +7.59; 0.99 / +8.41 | **0.49 / −0.01 (−0.3 +0.0); 0.49 / +0.00** |
| r5 tense120 f07_tense jrem4 q90 / r11 | 0.66 / +1.93; 0.79 / +3.55 | **0.50 / +0.00 (+0.1 +0.6); 0.51 / +0.00 (+0.0 +0.5)** |
| r5 tense120 tense_tok raw / jrem4 q90 | 0.87 / +1.88; 0.58 / +0.33 | 0.59 / +0.04 (+2.7 +3.4\*); 0.49 / −0.00 (−0.3 −0.0) |

- **Yes: tense's effect vanishes without the LoRA, as col's does.**
  - The f07_tense effect (the P3 template family) is completely flat with the adapter off, on both
    item sets and for both raw and 4r\*.
  - tense_tok 4r\* is flat too.
- **The only adapter-off residue is tense_tok raw:** +0.04 to +0.06 nats, 30–45× smaller than with
  the adapter on, but z +3 to +5. Base Qwen given the raw (in-J) tense vector as a slot embedding
  leans very slightly toward past-tense tokens. There is no such residue for 4r\*.
- **col reproduces R5.2d on fresh items:** adapter off gives 0.42–0.45 with slight anti-tracking
  (z −1.9 to −3.1).
- **Zero-dose LO:** col nl_cont −27.9 with the adapter on and −10.1 off. tense f07_tense −0.01 on and
  −3.28 off.

## Interpretation

- **Adapter-off rules out the trivial account for both features:** base Qwen reacting to the injected
  vector does not produce the effect.
- **For col, the gain ratio of ≈ 1 and the matched J-removal retention (0.79 vs 0.75) favour
  re-instatement.** In that account, the LoRA hands the activation back to Qwen's own next-token
  circuitry, which makes the same newline decision the model makes.
- **Two observations do not fit a pure re-instatement story:**
  1. **Tense 4r\*:** OL follows it strongly (f07_tense z +12 to +16), but the model's own next-token
     tense contrast does not move (z −1.4 to −3.2).
  2. **Indentation:** OL's continuation grows with dose (AUROC up to 0.73 at r216), but the model's
     indentation readout is flat. This one is sub-threshold at the matched-null dose.
- **Both observations are "reader moves, model's next token doesn't".** For tense, though, the
  model's next-token readout at the read position is a weak behavioural test: the next token is often
  not a verb. A delayed or generation-level tense readout could still move. So this is evidence about
  the readout as much as about OL.
- **For the main run:**
  - The §4.4.5 tense behavioural check is the next-token log P(past) − log P(present) at L42. Here it
    fails for tense 4r\*: model shift −0.05 nats, z −1.7 / −3.2.
  - If that replicates, P3 must be worded "reader sensitivity to a behaviourally inactive direction"
    under the prereg's own rule.
  - **Before the freeze, the lead should check §4.4.5's tense readout on this pilot result.**

## Provenance and cost

- **Candidates and code unchanged since they were fixed.**
  - The sha256 of o9_items.py, o9_lib.py, o9_templates.py, o9_modal.py, env.py and pilot_split.json
    all match `results/fixed_files.txt` (verified by `o9_analyze.py`).
  - o9_templates.py is unchanged since 2026-10-06 06:34 IST.
- **The run file `o9_modal_rfm.py`** differs from the hashed `o9_modal.py` only in execution
  parameters (verified by diff):
  - app name `rfm` and volume `rfm-results`;
  - GPU A100-80GB instead of H100;
  - model batch 8 instead of 16; OL `logprob_of_strings` batch 24 instead of 120;
  - `score_ol` with max_containers=4, **max_inputs=1**, retries=2; max_containers=1 elsewhere.

  It has no change to items, conditions, templates or statistics.
- **Timeline (UTC, from call-id ULIDs and file stamps):**
  - prep spawned 04:13Z;
  - `random_selection.json` written 04:17Z;
  - behave (codeNL/IN, tense) 05:05Z;
  - behave col and tense_r5 re-spawned 08:20Z;
  - all 34 score_ol calls spawned 10:33–10:34Z, i.e. **random channels and templates were fixed
    before any OL scoring**.
- **Earlier failed attempts:** the first attempt (`results/watch.log`) failed on behave:col,
  tense_r5 and every score_ol call (CUDA OOM). The A100 re-run (`deploy_rfm4.log`, 16:03 IST) completed
  all 39 calls with no errors (`results/run_summary.json`, `results/watch_rfm.log`). All OL chunks
  load with every condition present and no NaNs (asserted).
- **Cost (lower bound):** in-function GPU time from the per-call `t` fields.

  | stage | GPU time |
  |---|---|
  | prep | 0.06 h |
  | behave | 0.84 h |
  | tense_r5 | 2.18 h |
  | score_ol | 29.64 h |
  | **total** | **32.7 A100-80GB h ≈ $82** at $2.50/h |

  This excludes cold starts and image pulls, and excludes the failed earlier attempts, whose
  durations are not recorded locally.

## Verdict

**S6 (pilot) = READING by the §10.1 rule as written.** That implies the P2 row "OL carries non-J col
content in its continuation channel". **The verdict is not trustworthy, for two reasons:**
- it is produced solely by tense 4r\*'s undefined gain ratio (model shift ≈ 0);
- under the minimal correction (exclude directions without a ≥ +3 z behavioural effect, as the prereg
  already does for random directions), S6 = **INCONCLUSIVE**. That gives the P2 row: "OL's
  continuation channel is sensitive to the J-removed col edit; whether it carries the model's own
  decision or col content beyond it is unresolved (S6 inconclusive)".

The col-specific evidence leans toward **acting**: gain 1:1, matched J-removal retention, and
adapter-dependent.

**Recommended (lead's decision; pre-freeze, pilot-informed, to be disclosed):**
1. Add a validity clause to §10.1(b): a direction enters the gain-ratio set only if its model
   behavioural shift is ≥ +3 z on both bands. Otherwise S6(b) is void, or the direction is replaced by
   the pre-declared next candidate.
2. Fix the gain-ratio dose (r108, matched relative norm) and the random-direction pooling in the S6
   spec.
3. Note that strict "within ±25%" is unlikely on fresh items (random 1.7–1.9× vs col 1.0×). The
   *acting* branch is then effectively unreachable, and S6 will most likely be inconclusive unless
   rule (b) is re-thought (e.g. col's ratio vs the in-J raw col ratio).
