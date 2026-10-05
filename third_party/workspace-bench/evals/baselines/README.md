# Baselines

Measured floors for the judged families. A family's pass rate means nothing on its own: the
judge picks from option lists, and a strong model handed only those lists is not a uniform
guesser (golds tend to be more specific and more answer-shaped than distractors drawn from
other items). The floors here are what a real arm has to clear.

## Lucky guessing (`lucky_guessing.json`)

A model is shown NOTHING but a family's option lists (no question, no passage, no readout) and
asked to guess. The lists come from the judges' own builders, so option text and gold
positions are byte-identical to what the judge sees, and so is the order wherever the judge
fixes one (brew reshuffles per cell, so its order is a seeded one of its own); the trailing
"cannot tell" escape is dropped (a guesser with no readout has an honest reason to abstain that
a judge does not, and the gold is never the escape). Ported from the source repo's
`lucky_guessing_baseline.py`.

Variants, five draws per item at temperature 1.0 with the repo judge (`google/gemini-3.8-flash`):

- **blind**: the lists and nothing else.
- **described**: one honest sentence saying what the family tests (the option domain, never the
  judge's question) plus three OTHER items of the family with their gold marked: the
  informed-guesser ceiling.
- **uniform**: seeded uniform picks over the same lists, no model; lands on the analytic floor
  (`mean over items of prod(1/n)`) and catches option sets that are not what the floor assumes.

Pass rule per draw = the family's own: every list right (role-bound's three questions,
relational's two hops, a multilingual family's concept and language). `mean` is the pass rate
averaged over draws with its std, `majority` the per-item plurality vote. Families and lists:

| family | lists per item | options | notes |
|---|---|---|---|
| conjunctive_association | 1 | 10 | |
| role_bound_association | 3 | 5 | agent / action / patient; a people list may repeat a label (the judge's list does) |
| relational_multihop | 2 | 10 | one pooled list asked twice (outer, inner) |
| moral_rationale | 1 or 2 | 5 | committed items one list, deliberative items one per side |
| user_modeling | 1 | 5 | |
| directed_modulation | 1 | 5 | |
| typo_mt | 1 | 5 | the correction |
| multihop_mt | 1-2 | 5 | one list per bridge |
| multilingual_mt, multilingual_multihop, multilingual_typo, basic_readout_mt | 1-2 | 5 | concept/correction/readout plus the language list for L2 items |
| multi_concept_directed_modulation | 1 (multi-select, one to three picks) | 6 | pass = the FIRST pick is a dictated concept (one guess per draw, like every family); `any_hit` = some pick is, `exact` = the picks equal the dictated set, `mean_picks` beside them; controls dropped |
| brew_intermediates | 1 | 5 | the gold, the start, the answer and two off-trajectory colours. It answers one question: does the colour list itself give the intermediate away (the gold colours are not uniform over the palette). Its like-for-like arm number is `extras.cell_primary_gold_rate`, the judge's single `primary` pick per cell; the lens comparison is `extras.baseline` / `extras.null`, not this floor, because the headline rule aggregates emission cells and the judge's multi-select names as many colours as the readout mentions. `described` is weak here: brew's judge is deliberately told not to guess the task, so only `blind` has a story |

**Reading a floor against a headline.** A family headline is a max over the read grid (any
layer, any position), while the floor is one guess per item per draw. The two are comparable
only when the guesser repeats itself across draws (`majority` close to `mean`): a guesser that
spreads its picks would clear an any-of-grid rule more often than `mean` says. Check that
before reading a floor beside a headline. Role-bound's people lists can repeat a label; the
analytic floor counts a repeated gold twice.

Every entry is stamped with the family's judge `prompt_version` (`instrument`); `wsbench report`
draws a floor only while it matches the family's current instrument, so a changed judge drops
its floor until re-measured.

The six multi-token families (`basic_readout_mt`, `multihop_mt`, `multilingual_mt`, `typo_mt`,
`multilingual_multihop`, `multilingual_typo`) are scored by the deterministic regex contract since
2026-09-23 (`mt-regex-2026-09-23`). Their lucky-guessing entries are MC-specific — a guesser over
option lists the regex scorer never sees — so they no longer apply and are no longer drawn (the
stamp `mc-2026-09-16` no longer matches); they stay in the file for `opts=judge=mc` diagnostics.

```
wsbench baseline variant=uniform                       # no API key; every family
wsbench baseline families=typo_mt variant=blind limit=3  # pilot
wsbench baseline                                       # blind, described, uniform; every family
wsbench freeze                                         # outputs/baselines/lucky_guessing -> this file
wsbench report dir=outputs/<arm>                       # lucky guess column beside each family
```

Runs write `outputs/baselines/lucky_guessing/<family>/<variant>.json` (per-item prompts, picks,
aggregate); `freeze` refuses a `limit` pilot and merges per (family, variant) into the tracked file.
A blind floor can sit below the uniform one (relational 0.000, typo_mt 0.134 vs 0.166): the
guesser's shape heuristics are anti-correlated with the gold there, not a bug.

## Prompt-only (`prompt_only.json`)

Stock Qwen3.6-27B given the exact prompt text up to the read token and asked what a language
model would be thinking there, no activation (the source repo's `prompt_only_summary`, k=1,
T=1.0, one generation per prompt position replicated into every layer file). The summaries are
judged by each family's own instrument at ONE layer (identical rows across layers make any-layer
equal per-layer), and `wsbench freeze kind=prompt_only` records the rate with the judge model
and prompt version. Not item-blind: a floor, never a competitor lens.

Covered: 22 of the 27 families — every family whose headline is a pass rate, plus
`buggy_code`'s 1-10 score. Not measured: `jailbreak_recognition` and
`agentic_misalignment`, whose judges are pinned to Claude Sonnet 5, and the three exclusions at
the bottom. Entries carry `complete: false` by construction, because judging one layer is a layer
subset of the family's grid.

**The judged layer differs by vintage** and is recorded per entry in `layers_judged` for the
families that report it (seven do not; for those the run's nominal layer is the only record):

| entry | judged at | why |
|---|---|---|
| the 13 frozen 2026-09-16 (basics, directed_modulation, the multi-token six) | 20 | the layer their generation replicated |
| the nine frozen 2026-09-18 | 44 | the nominal layer of that generation |
| `buggy_code`, `arithmetic_intermediates` | 56 | their activations were captured at layers 56/60 only |

The six multi-token entries were re-scored on 2026-09-23 with the regex scorer from the SAME
prompt-only summaries (`gen-prompt-only-hard-mt`, layer 20, `judge_model` `regex`, instrument
`mt-regex-2026-09-23`): `basic_readout_mt` 0.40 -> 0.59, `multihop_mt` 0.52 -> 0.69, `multilingual_mt`
0.37 -> 0.42, `typo_mt` 0.46 -> 0.67, `multilingual_multihop` 0.36 -> 0.45, `multilingual_typo` 0.48 ->
0.66. The stock model's summary usually states the bridge or the correction the prompt makes
recoverable, so these are high, not item-blind floors, and several lens arms sit below them
(`multihop_mt`: s3d-rl600 0.32 against 0.69); read them as the text-leakage caveat above says.

The `poetry` and `user_modeling` entries were measured at the read sites those families used
until 2026-09-23 (the final prompt token / the assistant-onset token); both families now read
wider spans, and their prompt-only floors are being re-measured on the same spans.
The `multihop` entry was measured at the final-prompt-token site the family used until
2026-09-23 (a bare space token on 33 of the 100 items); the family now reads the last word.

The `arithmetic_intermediates` entry was measured at the variant's frozen cell (one cell per
item, layer 56). Since 2026-09-23 the family's run of record is judged over every position of the
prompt at layers 56 and 60 with an any-cell rule (`opts=cells=all`), which the one-cell floor does
not match: read it as the frozen-cell floor until it is re-measured on the same grid.

`role_bound_association` reads the top-up capture, the only one holding all 100 items, so its
number covers the whole bank while a lens arm reading the 20-item capture does not.

### Two different reasons a prompt-only number is not a bar

*Saturated* — the prompt states or dictates the answer, so the rate is pinned near the ceiling
and says nothing about a lens. These carry `saturated: true` and `wsbench report` marks them
instead of drawing them as a floor; compare the family's own null instead.

| family | prompt-only | why |
|---|---|---|
| role_bound_association | 1.00 | the scene states who did what to whom, so a summary of it answers all three questions |
| typo | 1.00 | the correction is recoverable from the misspelling (its README already says so) |
| brew_intermediates | 0.96 (92 of 96 decided items) | the rule table and the start colour are both in the prompt |
| relational_multihop | 0.82 | the cloze states both hops |

*Not item-blind but informative* — the answer is derivable from the prompt, yet the stock model
rarely states it, so the rate IS the family's measured floor and is the only one those families
have: `arithmetic_intermediates` 0.15, `buggy_code` (re-measured under the 1-10 score instrument;
the stamped entry is dropped from `report` until then), `chain_intermediates` 0.03.

The remaining new entries behave as intended floors: `user_modeling` 0.07,
`conjunctive_association` 0.07 and `moral_rationale` 0.50. The 13 entries frozen 2026-09-16 are
pass rates, some
of them high for the same text-leakage reason (`multihop` 0.90, `multilingual` 0.76,
`basic_readout` 0.72, `poetry` 0.71): the two groups above are the flagged cases, not a claim
that every other entry is a clean bar.

### Reading one of these numbers

- **Clearing the floor is not the same as clearing chance.** Two floors sit below their family's
  analytic chance line (`user_modeling` 0.07 against 1/6, `conjunctive_association` 0.07 against
  1/11), because a prompt-only summary commits to a wrong option rather than abstaining. Read a
  lens against both.
- **Not every column is a pass rate.** `buggy_code` is a mean 1-10 score, which `report` labels.
- **The summaries are capped at 256 new tokens** (the sampling of the first vintage, kept so the
  two vintages are one instrument), and 40-73% of them are cut mid-sentence, most often on the
  long prompts (scenes, whole programs, rule tables). That biases every rate here DOWN.
- **An empty generation counts as a miss.** `arithmetic_intermediates` has 138 empty cells of
  596; on non-empty cells its rate is 0.20 rather than 0.153.
- `_source` describes the LATEST generation (model, adapter, prompt kind and template,
  sampling). The per-entry `instrument`, `judge_model` and `layers_judged` are what a given rate
  belongs to.

Excluded, neither frozen nor shown:

- **multi_concept_directed_modulation.** Its prompt dictates the concepts ("Think about the
  plumber's blue ladder ..."), so the stock model names them every time (measured 1.0) and the
  number says nothing about the lens.
- **hallucination** and **jlens_concept_pr.** Neither headline is a bar a lens clears: one is
  the rate at which a readout invents things about the conversation (lower is better), the other
  is precision against the J-lens top-10. A prompt-only description has its own invention rate
  (measured 0.53) and its own precision (0.19), but neither bounds a lens, and putting them in
  the floor column invites reading them in the wrong direction.
