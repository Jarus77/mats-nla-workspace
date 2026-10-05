# 07: template mining (D9 item 6), frozen before any steering run

**Frozen file:** `templates_frozen.json`
**sha256:** `09156722b059edb99d01e919c29b2ff660e37cba3152f2d3ce67f3272d3e923b`
(also in `results/frozen_sha256.txt`). Frozen 2026-10-05; no steering data was used anywhere.

## What was done

This follows the lead's brief plus the two coordinator updates (priority order; Fable round 4).

- **Features (14):**
  - `nj1_col` (count family) and `nj1_col_break` (decision family);
  - in-J calibrations `ij2_lang` (FR/DE), `nj2_tense`, and `ij_listnum` ("Item N:", numerals visible: the in-J count control);
  - backups `bk_indent`, `bk_wordpos`, `bk_listidx` (unnumbered bullets), `bk_quote`, `agr_gender` (FR/DE), `agr_number` (FR), `bk_caps`, `bk_code` (Python/JS);
  - `nj3_predpos`, deprioritised but kept because it was cheap.
  - `col_report` and the typo target were dropped, per the lead.
- **Items.** All items are new, with 20+20 for mining and 80+80 for selection per feature. Each split is balanced by value and clustered by passage; code pairs stay together.
  - `col` reuses 02's `s1_data.build_wrap` (W ~ U{30..90}, same `col` definition, read at a token inside a line) on fresh wikitext paragraphs. Every paragraph 02 used is excluded.
  - Tense reuses 02's `build_prose` / spaCy `_clause_tense`, and NJ3 reuses 02's POS-lexicon recipe.
  - The other generators are ours (`items_build.py`). Language uses the papluca **test** split; agreement uses the **train** split.
  - Bins:
    - col: short 10-30 vs long 55-85 chars;
    - col_break: soon (W−col ≤ 6) vs far (W−col ≥ 25), with col 40-75 in both classes, so the decision is separated from the count;
    - wordpos: 2-5 vs 18-40 words;
    - list index: 2-3 vs 8-12;
    - indentation: level 1 vs 3-4.
- **Mining.** k=8 free readouts per mining item from NLA@42 (iter_000400, T=1) and from OL at 42, 40 and 44 (WorkspaceBench sampling). That is 320 NLA and 960 OL readouts per feature. J-Lens top-50 lists were also taken at L42.
- **Extraction.** Gemini 3.8 Flash saw each reader's mining readouts grouped by true value and proposed template families with one `{v}` slot, in the reader's native format.
  - The same prompt was used for every reader. Candidates were then validated mechanically (one slot, OL starts with `- `, candidates differ only in the slot) and stripped of a copied display artefact.
  - For col, extraction ran as two separate calls, one for the long-line/line-length family and one for the line-break-coming family. Each asked for qualitative templates first and numeric ones second.
  - J-Lens word sets were mined statistically: tokens over-represented in one value's top-50 lists.
  - Cost: **$0.61 over 92 calls** (cap $15).
- **Selection.** Each candidate was scored by paired log-odds on the selection split, keeping both candidates of an activation in one forward batch.
  - Selection metric: AUROC of LO for true vs alternative value at the primary layer (NLA@42, OL@42, J-Lens@42), with d′ as tie-break.
  - The top 3 distinct mined templates were kept.
  - For col / col_break, the 3 primary templates are **qualitative**, and up to 3 **numeric** templates are kept as `secondary_numeric`.
  - The fixed label template is the 4th column and was never eligible for selection.
  - The col fixed label is the lead's qualitative example ("at the start of a line" / "near the end of a long line"). The numeric label is kept as an extra `fixed_numeric` column.

## Selection-split AUROC (n=160 per cell; 95% cluster-bootstrap CIs are in the JSON)

Columns:
- **sel 1/2/3**: the three frozen templates at L42.
- **mean3**: AUROC of the template-averaged LO (prereg §6.1).
- **fixed**: the fixed label template.
- **OL mean3 @40/44**: the OL sensitivity layers.

| feature | NLA sel 1/2/3 | NLA mean3 | NLA fixed | OL sel 1/2/3 | OL mean3 | OL fixed | OL mean3 @40/44 | J-Lens mean3 / fixed |
|---|---|---|---|---|---|---|---|---|
| nj1_col (count, qualitative) | .58/.56/.55 | **.60** [.51,.69] | .50 (num .45) | .56/.49/.48 | .55 [.45,.63] | .45 (num .49) | .55/.54 | – / .48 |
| nj1_col secondary numeric | .51/.50 | | | .54/.47 | | | | |
| nj1_col_break (decision) | .56/.52/.46 | .54 [.45,.63] | .51 | .55/.54/.50 | .55 [.46,.64] | .46 | .56/.55 | – / .45 |
| ij2_lang | 1.0/1.0/1.0 | 1.00 | 1.00 | 1.0/1.0/.96 | 1.00 | .98 | 1.0/1.0 | .67 / .98 |
| nj2_tense | .85/.85/.83 | .85 | .90 | .92/.90/.88 | .92 | .56 | .93/.94 | – / .53 |
| ij_listnum (in-J count ctrl) | .80/.71/.71 | .84 | .98 | .99/.97/.95 | .99 | .97 | .99/1.0 | .64 / .54 |
| bk_indent | .88/.84/.49 | .84 | .87 | .89/.82/.78 | .89 | .44 | .86/.91 | .60 / .68 |
| bk_wordpos | .84/.61/.61 | .72 | .62 | .55/.49/.42 | .51 | .55 | .51/.51 | – / .46 |
| bk_listidx (no numerals) | 1.0/1.0/.94 | 1.00 | .88 | .98/.95/.92 | .99 | .96 | .99/1.0 | .55 / .57 |
| bk_quote | .80/.80/.71 | .83 | .78 | .76/.73/.56 | .76 | .60 | .77/.75 | – / .49 |
| agr_gender | .92/.78/.50 | .81 | .89 | .99/.80/.77 | .92 | **.26** | .92/.92 | .60 / .56 |
| agr_number | .93/.93/.74 | .98 | .99 | .91/.84/.59 | .86 | .73 | .87/.90 | .84 / .79 |
| bk_caps | .84/.79/.79 | .86 | .51 | .92/.92/.85 | .93 | .54 | .93/.93 | – / .52 |
| bk_code | .99/.99/.98 | .99 | .98 | .99/.99/.98 | .99 | .93 | .99/1.0 | .73 / .95 |
| nj3_predpos | .70/.63/.57 | .63 | .72 | .64/.64/.57 | .66 | .54 | .67/.66 | – / .52 |

"–" for J-Lens means no token was differentially present in the mining top-50 lists, so only the fixed word set exists. That is itself consistent with non-J for col, col_break, tense, caps, quote and wordpos.

## Findings

1. **Neither reader ever spontaneously mentions line length or line breaks.**
   - We ran an LLM yes/no pass on every col and col_break mining readout. It found mentions of line length, column, width/wrapping or an upcoming break in **0/320 NLA and 0/960 OL readouts for col, and 0/320 and 0/960 for col_break**.
   - The regex hits (NLA 6%/11%, OL 1%/4%) are false positives, such as "character" in the narrative sense.
   - So the col templates are **reader-style paraphrases, not recurring phrasings**. The extraction prompt allowed this when the feature is never mentioned.
   - OL writes content continuations of the passage, as the smoke test showed. The NLA describes genre and content.
2. **Col is barely readable from natural items with any template.**
   - The best is the NLA mean3 at 0.60 [0.51, 0.69]. Every OL cell, every col_break cell and every fixed or numeric label template has a CI that includes 0.5.
   - Spearman of LO against the raw count on the selection items is about −0.1 to +0.1.
   - The selected AUROCs carry a best-of-~7 winner's-curse bias. **Re-estimate natural-arm col AUROC on the evaluation items; do not reuse these.**
3. **Mined templates beat fixed labels mainly for OL.** OL's native, continuation-style templates help on:
   - tense (`- It {was|is}`): .92 vs .56 fixed;
   - indentation: .89 vs .44;
   - caps: .93 vs .54;
   - gender (`- {le|la}`-style): .92 vs **.26**. The fixed OL label "- masculine noun" anti-tracks.

   For the NLA, the fixed label is often as good as the mined templates or better: tense .90, listnum .98, agr_number .99. The NLA is already label-like on the features it mentions.
4. **NLA-mined templates encode co-variates, not labels.** For tense they say "Historical" vs "Descriptive" register, and for language "French/German product review". This is what the readouts contain. Downstream should report the fixed column next to them, as D9 requires.
5. **Base rates of explicit mention** (LLM estimate, NLA / OL): language .85 / 0; code .55 / .03; quote .28 / 0; listnum .15 / .35; tense .02 / 0; caps .05 / 0; agreement ≤.05 / 0. Full numbers are in `results/extraction_log.json`.

## Caveats

- `col_break` secondary numeric templates ("0-6 characters", "25+") echo the bin definitions from the extraction prompt rather than the readouts. They are secondary only.
- Three cells have a weak third pick that was forced by k=3: NLA indent .49, NLA gender .50, OL quote .56.
- `agr_gender` is 80% French (160 FR / 40 DE items). German uses only einen/eine.
- Items other than col, tense and NJ3 are tokenized as truncated prefixes, so the read token is the last token of the prefix. Col, tense and NJ3 are slices of an in-context tokenization.
- **02's dose files were not used.** Selection is by natural-item discriminability, which is dose-independent by design. Col items come from fresh paragraphs, so there is no overlap with the data 02's directions and doses were fit on.
- **Evaluation must exclude every id in `items[feature].mine/select`** of the frozen JSON. All of them are new, but downstream generators reusing the same fresh pool could collide.

## Compute

- Modal: gen_ol about 14 × 7-27 min on H100, gen_nla 14 × 2.5-5 min on H200, scoring 14 × (1.5-3.5 min H100 + 1-3 min H200). In total about 4 H100-h + 1.5 H200-h, roughly $25 at list price.
- One H200 scoring call hit an OOM from warm-container reuse. It was fixed with `max_inputs=1` and rerun.

## Files

| file | contents |
|---|---|
| `feature_spec.py` | feature values, bins, fixed templates, validator |
| `items_build.py` | generators (02 reuse via a stub import, read-only) |
| `env.py` | images (03 pins) |
| `tm_modal.py` | Modal stages: `build`, `build_extra`, `gen_ol`, `gen_nla`, `score` |
| `extract_templates.py` | Gemini extraction, base rates, J-Lens word sets |
| `select_freeze.py` | selection and freeze |
| `results/` | `gen_*`, `score_*`, `candidates.json`, `extraction_log.json`, `selection_table.json` |

The volume also holds `/results/07_template_mining/acts_<feature>.pt`: L40/42/44 captures at the read token for all items.

## Natural col vs TextNLA (Fable round 5)

**Setup.** Same 160 `nj1_col` selection items and the same frozen candidate pairs. `templates_frozen.json` is unchanged; its sha256 is still `0915…923b`.

- **TextNLA-Qwen (local).** Base Qwen3.6-27B with no activation, given the passage text up to the read token. It uses the NLA warm-start instruction with `mode="prompt"` (`src/mats/textnla.TextNLALocal`) and is teacher-forced on each frozen pair. All templates were scored: selected, secondary numeric and fixed, for both NLA and OL.
- **TextNLA-Gemini.** Gemini 3.8 Flash, elicited JSON P(A), both option orders averaged, max_tokens 600. The question asks which description better fits the model's state at the last token.
  - Budget: it ran out after **7 col templates**: the 3 NLA-selected, the 3 OL-selected, and the NLA fixed label.
  - Spend: $2.20 measured on the rerun. An earlier attempt crashed unsaved on a connection reset (`RemoteDisconnected` is not caught by `TextNLAGemini`; the script now subclasses it). Its spend is unmeasured because the OpenRouter key is shared, but the total for this follow-up is estimated under the $5 cap.
- **`nj1_col_break`.** Scored by TextNLA-Qwen only (no Gemini budget left): NLA mean3 .54, OL .55, TextNLA-Qwen .50 / .51. None of these is distinguishable from chance.

**AUROC, selection split** (n=160; 95% cluster-bootstrap CIs; ρ = item-level Spearman of reader LO vs TextNLA LO):

| template (candidate for "long") | reader @42 | TextNLA-Qwen | TextNLA-Gemini | ρ reader~Qwen | ρ reader~Gemini |
|---|---|---|---|---|---|
| NLA "Readout taken late in the span of the wrapped line." | .58 | .50 | .92 | +.11 | +.13 |
| NLA "Mid-sentence continuation located near the end of a hard-wrapped line." | .56 | .55 | .92 | +.28 | +.14 |
| NLA "The token appears far into of the wrapped line." | .55 | .40 | .97 | +.06 | +.12 |
| NLA fixed "The current position is near the end of a long line" | .50 | .54 | .97 | .00 | +.01 |
| OL "- deep after a newline" | .56 | .53 | .91 | +.17 | +.07 |
| OL "- line position: late in the line" | .49 | .65 | .97 | −.15 | +.07 |
| OL "- current line depth: deep" | .48 | .53 | .63 | +.16 | −.05 |
| **NLA mean3** | **.60** [.51,.69] | .48 [.39,.57] | **.97** [.94,.99] | +.31 | |
| **OL mean3** | **.55** [.45,.63] | .62 [.54,.71] | **.97** [.93,.99] | +.24 | |

Each reader's mean-of-3 LO was also regressed (OLS) on each text-only LO, and the AUROC of the residual recomputed:

| reader | after regressing out TextNLA-Qwen | after regressing out TextNLA-Gemini |
|---|---|---|
| NLA | .61 | .51 |
| OL | .52 | .47 |

Numerals and the remaining fixed labels were Qwen-only: every one is .40–.54.

**Reading.**
1. **The expectation holds in its strong form: line position is plainly visible in the text.** Gemini, reading only the passage, separates 10–30 from 55–85 characters into the line at AUROC .97 with the very same candidate pairs. It does so for qualitative and fixed wordings alike, and needs no counting.
2. **Base Qwen does not express this through teacher-forced candidate log-probs**: .48 with the NLA templates, .62 with the OL ones. The local TextNLA is therefore a weak text ceiling for col, and Gemini is the relevant one.
3. **The NLA's small natural col signal (.60) carries no information beyond the text.**
   - It survives regressing out base-Qwen TextNLA (.61) but vanishes after regressing out Gemini (.51).
   - Its item-level agreement with either text-only reader is low (ρ ≤ .31). The NLA tracks a sliver of the text-visible signal, not something the text lacks.
   - OL is at chance throughout.
4. **Natural-arm col AUROC is uninformative about activation reading,** just as language identity is. The headline must come from the steering arm. Any natural col number should be reported next to the Gemini text ceiling (.97).

**Files.**

| file | contents |
|---|---|
| `textnla_col.py` | Modal, H100: TextNLA-Qwen scoring |
| `textnla_col_analyze.py` | local: Gemini and the comparison |
| `results/textnla_local_nj1_col{,_break}.json` | TextNLA-Qwen scores |
| `results/textnla_gemini_nj1_col.partial.json` | Gemini, the 7 templates scored |
| `results/textnla_col_summary.json` | all AUROCs, CIs, ρ and residual AUROCs |
