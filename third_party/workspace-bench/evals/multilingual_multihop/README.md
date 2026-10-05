# multilingual_multihop

A non-English two-hop prompt; the lens must write the bridge concept (its English or Chinese form) in one layer.

98 items (source bank `lens-eval-multilingual-multihop.json`, parent `multihop-mt`,
frozen 2026-09-10). Plain render, one read position: the final prompt token, immediately
before the answer word. Every item and every bridge was gated on Qwen3.6-27B (greedy-verified,
then >= 8/10 at temperature 0.7).

Example: an Arabic two-hop prompt through kente cloth → `kente` / `kente cloth` / `肯特布` in one sample.

Scored units: `bridge_native` (required; `en` + `zh` forms of the bridge). `bridge_l2` (the bridge in the source language) and `language` are OPTIONAL in this bank — they feed `any_hit` and `unit_any_layer`, never the pass. This is the one family where the regex contract is LOOSER than the MC judge, which also required the language pick.

The bank file is a frozen copy of the source repo's hard-tier bank; its `contract` block, per-unit `forms` and `probe_token_lens` counts are exactly what the scorer reads.

## Scoring

- **Headline: regex pass rate** — `scorer_version` `mt-regex-2026-09-23` on prose readouts:
  deterministic, no judge call, `spend_usd` 0, `judge_model` `regex`. Token-bag readouts carry
  `scorer_version` `mt-regex-summarized-2026-09-23`: each bag is first interpreted by the shared
  summarizer (below), so for those arms the instrument is the summarizer model (`judge_model` =
  `google/gemini-3.8-flash`, `pinned_instrument` follows its pin, `spend_usd` counts its calls, a
  failed summary leaves the item undecided). The scorer is `wsbench.multitoken.regex`, a port of
  the source repo's conjunctive bank scorer (`olens_suite/bank/matching.py`, `contract.py`,
  `conjunctive.py`, `olens_sglang/score_targets.py`); goldens computed by the source code pin it
  (`tests/golden/mt_regex_units.json`, `tests/golden/mt_regex_matching.json`).
- **The contract, verbatim from the source:** (1) a cell's samples are its prose samples, or each
  top-k token string of a token lens; chat scaffolding (`<|im_start|>`, `<|im_end|>`, `<think>`,
  `</think>`, `<explanation>` tags) is stripped first. (2) The item's `units[]` are scored; under the
  bank's `multi_token` contract only forms the bank stamped as strictly multi-token stay
  (`probe_token_lens` > 1: the source's count is the minimum over case variants and bare /
  leading-space encodings, so a form any single Qwen3.6-27B token could equal never earns
  credit); units marked `multi_token: false` (the language units) keep every form. (3) A form hits
  a sample when, after folding both sides (NFKD, combining marks dropped, curly apostrophe
  straightened, casefold), a wordy form matches at word boundaries with `[\s-]+` between its words
  (`Bohr magneton` = `Bohr-magneton`; `kentes` is not `kente`), a form containing CJK or Hangul or
  non-word characters is a plain substring, and a purely numeric form must sit in answer position
  (after `=`, `->`, `equals`, `answer/result/value/total/sum/product/quotient [is][:]`, their
  Chinese counterparts, or alone at the start of the sample, never as a prefix of a longer number;
  a CJK numeral there counts when it parses to the value). A form is matched against ONE sample at
  a time — never a join of samples or of adjacent top-k tokens. (4) A unit hits at a layer when any
  of its forms hits any sample at that layer, in any listed language. (5) A layer passes the item
  when EVERY required unit hits there; the item passes at any layer.
- **Scored units of this family:** `bridge_native` (required; `en` + `zh` forms of the bridge). `bridge_l2` (the bridge in the source language) and `language` are OPTIONAL in this bank — they feed `any_hit` and `unit_any_layer`, never the pass. This is the one family where the regex contract is LOOSER than the MC judge, which also required the language pick.
- **Token lenses — summarized first, then matched** (Camila, 2026-09-23 16:52: "for the token
  lenses on mt i thought we were doing an llm summarizer step?"). A strictly multi-token form can
  never equal one vocabulary token, so on the raw bag a required multi-token unit is unreachable
  by construction (the source's stated asymmetry: "multi-token targets can only be hit by the
  oracle lens"). The bench therefore keeps its bag -> prose step: every cell's top-k bag is
  interpreted once by the shared summarizer (`docs/summarizer.md`, `interp-v1`, the same
  `render_bag` text, cache key `summ:<cell>` and fingerprint the MC judge used, so its cached
  interpretations are reused), and the contract is applied to the interpretation as ONE more
  sample beside the raw tokens: a unit hits when a form is found in the summary OR in a single
  token, never across them. This is the faithful reading of the intent — the summary is the
  token lens's prose, the bag stays scorable for single-token units (a language name) and for
  producers whose "tokens" are phrases or labels (the template lens, an SAE's auto-interp
  labels). The `language` unit of an L2 item is scored on the summary like any unit: it hits
  when the interpretation names the language in English or Chinese, or writes the answer in the
  source script (the summarizer is told to translate foreign tokens, so it usually names the
  language); a bare language-name token in the bag also hits. Per layer, `layers[L].source`
  records which side carried each unit (`bag` / `summary`); `extras.passing_layers_by_source`
  splits the passing layers into `bag_only` / `summary_only` / `both` / `mixed` (neither side
  alone passes), `extras.n_summaries[_cached|_failed]` count the summarizer work. Token strings
  are matched as the producer wrote them (`Ġ`/`▁` as a space).
- **Item rule:** an item whose every expected layer is scored and never passes fails; a cell whose
  samples are all blank is a negative at that layer; an item with a MISSING (item, layer) cell and
  no pass is undecided (`extras.n_items_undecided`). Exactly one read position per (item, layer) is
  expected; a file with more is refused. Missing cells are fatal (exit 2) unless
  `allow_missing=True`; `layers=` restricts the grid and counts as a subset.
- **Extras:** `any_hit_rate` (some unit hit somewhere — the parent-comparable number; Camila's
  2026-09-23 substring preview corresponds to this, not to the conjunctive pass), `unit_any_layer`
  per role, `language_of_readout` (which language each unit surfaced in first), and `mc` (the
  forced-choice judge's number when `out/results.json` still holds one, so both instruments sit
  side by side without a call).
- **Floors:** the prompt-only baseline was re-measured under this scorer on 2026-09-23
  (`evals/baselines/README.md`); the MC lucky-guessing floor has no meaning without option lists
  and is no longer drawn.

## Instrument history

- Source repo (bank frozen 2026-09-10): this regex contract was the headline
  (`score_targets.py --match word --exact`).
- 2026-09-16 -> 2026-09-23 (PR #20): a forced-choice Gemini judge (`mc-2026-09-16`) replaced it —
  one prompt-blind five-way call per (item, layer, unit) with a cannot-tell escape and a
  verbatim-quote gate. Values of record then: s3d-rl600 0.26, s3d-sft251 0.19, nla-rl-L42 0.14.
- 2026-09-23, Camila: "regex on the multitoken please" — the regex contract is the headline again;
  on the same readouts: s3d-rl600 0.40, s3d-sft251 0.28, nla-rl-L42 0.34. The MC judge stays reachable with `opts=judge=mc`
  (its result is never pinned, so never `complete` and never in the macro; its prompts are kept
  verbatim below).
- 2026-09-23 16:52, Camila: token-bag arms keep the LLM summarizer step (above); on the same
  readouts, MC / regex-raw / regex-summarized: jlens 0.00 / 0.00 / 0.05; rlens 0.00 / 0.00 / 0.10; template-v3 0.00 / 0.00 / 0.08; logit 0.00 / 0.00 / 0.06; tuned 0.01 / 0.00 / 0.13; sae-l42 0.00 / 0.05 / 0.09.

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}` rows; `tokens` rows for a token
lens. `examples/readouts/multilingual_multihop.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=multilingual_multihop readouts=examples/readouts/multilingual_multihop.jsonl out=outputs/toy/multilingual_multihop dry_run=True
wsbench judge family=multilingual_multihop readouts=<arm>.jsonl out=outputs/<arm>/multilingual_multihop limit=3
```

## Judge prompts (the optional forced-choice judge, `opts=judge=mc`)

Not the instrument of record since 2026-09-23 (see Instrument history); run with
`opts=judge=mc` to obtain it beside the regex number.

`prompt_version`: `mc-2026-09-16`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. Output schema: `{"choice": "A".."F", "quote": str}`. The user message is the
`USER` template with the readout, one question (`QUESTION_UNIT` with the unit's role, or
`QUESTION_LANGUAGE`) and the lettered options, `CANNOT` last.

`SYSTEM`:

```text
You judge the READOUT of an interpretability lens: text decoded from a language model's internal activations while it read a passage you cannot see. The readout may be noisy, verbose, in any language, and may speculate. You are given ONE multiple-choice question about what the readout NAMES. Pick an option ONLY if the readout clearly names that one option (the same thing, in any language or spelling). If the readout names NONE of the options, or names TWO OR MORE of them (a hedge, a list of candidates, 'X or Y'), choose the "cannot tell" option — a readout must commit to a single answer to be credited. Do not use your own knowledge to fill gaps or to decide which of several named candidates is right. Give a SHORT VERBATIM quote (a few words copied exactly from the readout) that supports your choice; leave it empty when you choose cannot-tell.
```

`USER`:

```text
READOUT:
"""
{readout}
"""

{question}

{options}

Answer with the letter.
```

`QUESTION_UNIT`:

```text
Which ONE of these does the readout name (unit: {role})?
```

`QUESTION_LANGUAGE`:

```text
In which LANGUAGE is the passage the readout describes written? Naming the language counts; so does the readout quoting the passage's words in that language's own script. English and Chinese are the readout's own working languages and are not evidence by themselves.
```

`CANNOT`:

```text
cannot tell: the readout names none of these, or several of them
```
