# 0007 — Phase 7: judge-swap validation (Opus → Gemini 3.8 Flash) and port-parity checks

Design §3 phase 7. Infra: 0001–0006 merged — **this PR opens only after phases 3–6 have
landed on `main`**; six of the adapters and all parity rows depend on those families existing. This PR adds one script, one doc, and README
lines; the runs themselves are executed by the main session with real keys (`OPENROUTER_API_KEY`
from `/workspace/camila/.env`, read per command, never exported) and their outputs are committed
as small summary JSON, not raw verdicts.


> **Status (2026-09-16):** only §1 (the script) and the tests ship in this PR. The runs (§2), the
> κ doc and the README agreement lines (§3) are deferred until Camila asks — the repo carries the
> capability, not the spend.

## Reference arm and inputs (all local; `A` = `/workspace/agam/global-workspace/outputs/oracle_lens_evals`,
`C` = `/workspace/camila/global-workspace-clean/.claude/worktrees`)

Arm = **s3d RL iter600** (`ckpts/ao/rl/s3d.ddp600.s0/iter_000600`), the arm the design names.

| family | readouts (source layout) | Opus baseline verdicts (per-cell) |
|---|---|---|
| moral_rationale | `A/olens_sglang/gen-s3d-rl600-ec-audit` (gen dir, 200 labels × 6 layers) | `A/moral_rationale_eval/verdicts_s3d-rl600_fj.json` (`verdicts[]` keyed `id|L|pos|side`) |
| relational_multihop | `A/olens_sglang/gen-s3d-rl600-relf-audit` | `C/relational-near-miss/outputs/oracle_lens_evals/relational_eval/verdicts_pblank_s3d-rl600.json` (the p-blank instrument; `verdicts` keyed `label|L`) |
| role_bound_association | `A/olens_sglang/gen-s3d-rl600-oaeb-audit` — **only 20 OA labels exist** (plus 48 retired `eb-*`); `--ids-from-bank` intersects bank ids with gen-dir labels and prints the count (20) | `A/oa_eb_eval/verdicts_s3d-rl600_fj.json` (`verdicts[]`, `family == "oa"` rows only, 14,449 cells; fields `id, layer, pos, correct[3], pass`) |
| conjunctive_association | `A/olens_sglang/gen-s3d-rl600-latent-audit` | `A/oracle_latent_eval/verdicts_mc_s3d-rl600_fullblob_fj.json` (`per_item`) |
| user_modeling | `A/olens_sglang/gen-s3d-rl600-sbum-audit` (filter to `um-*` labels; 11 layers) | `A/olens_sglang/gen-s3d-rl600-sbum-audit/judge/user-modeling.json`: rows key on `name` (= port `id`), `layer`, `pos`, `sample_idx` (always 0). **Only 88 of 749 `verdicts[]` rows carry Opus `pick`/`basis` (`judge == "claude-opus-5"`); 661 are `judge: unavailable`.** Primary comparison = those 88 cells (the Haiku-escalated stratum, NOT grid-representative — say so). Secondary comparison = `screen_verdicts[]` (Haiku, 1,099 cells, all with `pick`/`basis`), labelled as Haiku. |

hallucination (already Gemini of record), jlens_concept_pr (Stage B/P already Gemini),
agentic_misalignment (pinned Sonnet) and jailbreak_recognition (pinned Sonnet since
2026-09-16) are **not** swapped; the script runs them only as a
no-op sanity check that the port reproduces the source numbers on the same readouts
(hallucination: `C/halbench-chat/.../hallucination_chat/s3d/summary.json`; jlens: the
Gemini-of-record s3d run is `C/jlens-pr/.../jlens_pr/judge_gemini38/s3d/` (stage_a/b/p) with the
arm summaries in `C/jlens-pr/.../jlens_pr/gemini38_scored_all/summary.json` under `arms["s3d"]`
(`by_layer["44"].precision`, `recall_at_10`);
agentic: `C/am-foils/.../narrative_design_s3d_rl_2026-09-11.json`; jailbreak: the Sonnet run
in `C/jailbreak-wire/outputs/wsbench/situation_mining/jailbreak/readout_judge.jsonl` deduped
by the last `status == "ok"` row per composite id (`arm|item|L20|p552|hash`, parsed), filtered
to `arm == s3d-rl600` and the 86 bank ids, asserted against `readout_eval.json`
`arms["s3d-rl600"].pass_rate` = 0.8256, readouts from `lens_readouts_s3d-rl600-clean.jsonl`)
— reported as "port parity", see §3. Parity for jailbreak is same-judge, so its κ is a
rerun-noise measurement, not a swap measurement.

## 1. `scripts/judge_swap.py` (new; a dev script, not part of the package)

```
uv run python scripts/judge_swap.py convert  --family F --src <gen dir or jsonl> --out outputs/swap/F.jsonl [--ids-from-bank]
uv run python scripts/judge_swap.py compare  --family F --results outputs/swap/F/results.json --baseline <verdict file> [--baseline-eval readout_eval.json] [--baseline-tier opus|screen] --out docs/judge_swap/F.json
uv run python scripts/judge_swap.py summary  docs/judge_swap/*.json --out docs/judge_swap_2026-09.md
```

- `convert`: wraps `wsbench convert-gen-dir` (prose) and adds the two special cases: the
  jailbreak flat jsonl (`{id, layers, olens: {L: {pos: [K samples]}}, tokens}` → one contract
  row per (id, layer, pos) with `token`), and the sbum/oaeb dirs filtered to the family's bank
  ids (`--ids-from-bank`). Prints the grid it produced.
- `compare`: joins the port's `rows[]` with the baseline's per-cell verdicts on a join key
  BUILT FROM FIELDS (never parsed from the baseline's `key` string) — with one exception:
  the jailbreak log rows carry only `{id, status, attempts, prompt_hash, verdict}`, so its
  composite `id` (`arm|item|L20|p552|hash`) IS parsed (5 parts, strip `L`/`p`, drop the hash).
  Port rows with a failed verdict (`pick == "api_fail"` or no verdict) are excluded from
  `n_cells_both` and reported as `n_port_failed`. Per-family adapter:
  moral `(id, layer, pos, side)` → `correct`; relational `f"{id}|L{layer}"` (the baseline
  keys carry the `L` prefix; a baseline-missing cell such as the one failed L60 call is
  `n_only_port`, not an error) → `pass`, `x_ok`, `y_ok`; role-bound `(id, layer, pos)` on
  `family == "oa"` rows → `pass`; conjunctive `id` → `pick == "gold"` (baseline `pick` is
  4-way gold/other/contrast/cannot_tell; cell = item, so `item_level` is omitted for this
  family); user_modeling `(id, layer, pos, sample_idx)` with port `id` ↔ baseline `name` →
  `pick == "gold"` and `basis == "inferred_characterization"`, restricted to Opus rows
  (`--baseline-tier opus`, the default) or `screen_verdicts` (`--baseline-tier screen`); jailbreak (parity row) `(id, layer, pos)` after
  the dedupe rule → `any_recognition`. Emits `{family, n_cells_both, n_only_port,
  n_only_baseline, agreement, cohen_kappa, headline: {port, baseline, delta}, item_level:
  {agreement, kappa} | null, confusion, baseline_model, baseline_note, port_model}`. Cohen's κ
  is on the binary cell label; κ is `None` (reported, not 0) when either side is constant.
  Item-level = the family pass rule applied to both sides over the SAME cells: for
  user_modeling the port's item rule is restricted to the 88 joined cells; for role-bound the
  baseline is 19/20 items so item κ is reported `None` with reason "near-constant".
- `summary`: one markdown table (family | n cells | cell agreement | κ | baseline headline
  (model) | Gemini headline | Δ | verdict; the user_modeling row notes that the delta
  conflates the judge change with the dropped screen→escalate pipeline) with the design's rule: **κ < 0.7 → "flag: prompt review
  before quoting"**, else "ok". Also the port-parity rows for the three unswapped families
  (port headline vs source headline on identical readouts; expected |Δ| ≤ judge rerun noise).

## 2. Runs (main session; each family once, default judge, full arm)

Cost estimate at 240 rpm: moral ≈ 7k calls, relational 1.2k, role-bound ≈ 14.4k (the 20 OA
items × all positions × 6 layers; the expensive family — run it last and stop early if spend
exceeds $30), conjunctive 100, user_modeling 1.1k, jailbreak ≈ 5.3k (Sonnet, parity); hallucination
≈ 5.6k, jlens ≈ 35k (skip the jlens re-run if spend is a concern — its Stage B/P judge is
unchanged, so parity is established by the ported unit tests; note that in the doc), agentic
(Sonnet, ~5k calls) — run.

Raw judge outputs go under `outputs/swap/` (gitignored). `compare --out
docs/judge_swap/<family>.json` writes the small comparison JSON straight into the committed
dir, and `summary` globs `docs/judge_swap/*.json`. The doc's example cells show keys and
labels only — never readout text or quotes (jailbreak items are WildChat-derived).

## 3. Docs

- `docs/judge_swap_2026-09.md`: the table, one paragraph per flagged family (what disagrees,
  with 3 example cells), the port-parity rows, the exact commands, spend.
- README §Judges gets a "Judge change 2026-09" paragraph linking the doc and quoting κ per
  family; each swapped family's README gets a "Judge agreement" line (κ, n, date).
- Jailbreak is pinned to Sonnet 5 (decided 2026-09-16); its row is a same-judge parity
  check and the README's judge-failure rate is recorded from that run.

## Tests

`scripts/` is new (ruff already lints the whole tree); tests import the script via
`importlib.util.spec_from_file_location` (it is not part of the package).
`tests/test_judge_swap.py`: `convert` on a temp gen dir and a temp jailbreak jsonl produces
valid contract rows (loader reports no malformed rows, `token` carried); `compare` on
hand-built port rows + baseline files for each family adapter gives the expected agreement /
κ (incl. κ undefined when one label is constant → `None`, reported as such); `summary`
renders the flag rule.

## Acceptance

Script + tests green; six `compare.json` committed; the doc and README lines in place; the
jailbreak failure rate recorded. No raw verdicts committed.
