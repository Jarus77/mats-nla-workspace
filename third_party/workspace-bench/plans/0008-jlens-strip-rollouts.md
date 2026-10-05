# 0008 — jlens_concept_pr: strip the Qwen rollouts from the shipped manifests (2026-09-17)

Camila: "get rid of all the rollouts on jlens precision recall".

## What the rollouts are

`evals/jlens_concept_pr/manifest.json` and `manifest_L42.json` each carry, per `prompts` row,
a `tokens` list: the decoded token sequence of the templated prompt **plus Qwen3.6-27B's
sampled continuation** (T 1.0, 512 new tokens; `items_manifest.json` §`rollout_sampling`).
That is the seed text (LMSYS / DailyDialog first turns, pile / fineweb prefixes) plus the
rollout, ≈ 1.3 MB of each 1.8 MB file (the two files are byte-identical except for `layers`);
stripped, each file is ≈ 70 KB. Nothing else in the family carries rollout text: the
reference `gen-jlens-pr-jlens/<label>/L###.jsonl` files are J-lens top-10 tokens, the gold
files carry concepts and J-lens tokens, `items_manifest.json` is counts and sampling metadata.

## What reads the manifests

`judge.manifest_items` reads `label`, `family`, `eval_positions[0]` only; `wsbench list`
counts `prompts`; `tests/test_jlens_concept_pr.py::test_bank_manifest_and_reference` reads
`layers`, `label`, `family`, `eval_positions`. `tokens` and `n_pos` are read nowhere.

## Change

1. Rewrite both manifests with the `tokens` key deleted from every `prompts` row; every other
   key (`label, family, file, n_pos, targets, look_for, eval_positions, units, contract`) and
   the top-level `model_id` / `layers` are retained verbatim, in the same order, with the same
   formatting (`json.dumps(indent=1, ensure_ascii=True)`, no trailing newline — the files have
   none today). One-off script, not committed.
2. `tests/test_jlens_concept_pr.py::test_bank_manifest_and_reference`: add
   `assert all("tokens" not in p for p in man["prompts"] + l42["prompts"])` and
   `0 <= eval_positions[0] < n_pos` for every row (the position stays valid without the text).
3. `evals/jlens_concept_pr/README.md` bank table: the `manifest.json` row no longer says
   "decoded `tokens`"; it lists the retained keys and says the rollout text was stripped
   (2026-09-17) and lives only in the source repo's acts manifest. The "Not shipped" paragraph
   gains the rollouts. `NOTICE.md` (LMSYS, DailyDialog, pile/fineweb sections) and README
   §Credits said the seed prompts are "included in `manifest.json`": reworded to say the seed
   text was carried there until 2026-09-17 and is no longer shipped (the credits stay: the items
   still derive from those corpora). Nothing else in the README changes; the instrument, `PROMPT_VERSION`
   (`jlens-pr-v2`) and the judge are untouched (the manifest is bank metadata, not a prompt).

## Invariants restated

- The 299 labels, the one eval position per label, the `family` split (150 chat / 149 pt) and
  the `layers` list are unchanged; `manifest_items` returns exactly the same list as before.
- Not a prompt edit: no `prompt_version` bump.
- No test other than the one above changes; the bank is frozen otherwise.

## Verification

`PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_jlens_concept_pr.py tests/test_readme.py
tests/test_prompts_in_readme.py`, `ruff check`, `ruff format --check`, plus a diff of the two
manifests showing only `tokens` lines removed (≈ −130k lines per file).
