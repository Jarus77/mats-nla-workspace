# 0002 — Phase 2: the four MC families

Design: `plans/0000-bench-v2-design.md`. Infra: `plans/0001-scaffold.md` (merged). This PR ports
**moral_rationale**, **relational_multihop**, **role_bound_association**,
**conjunctive_association** onto the `wsbench` package. All four are "MC over frozen or seeded
options" judges; their prompts are copied **verbatim** from the source scripts named below
(read-only, outside this repo; `R` = `/workspace/camila/global-workspace-clean`):

| family | source judge | source bank |
|---|---|---|
| moral_rationale | `R/scripts/oracle_lens_evals/ec_readout_judge.py` | `R/evals/workspace-bench/hillclimbing_evals/moral_rationale/items.json` (200) |
| relational_multihop | `R/scripts/oracle_lens_evals/judge_relational_multihop.py` | `.../relational_multihop/cloze_items_final.json` (100) |
| role_bound_association | `R/scripts/oracle_lens_evals/oa_eb_readout_judge.py` | `.../role_bound_association/items.json` (100) |
| conjunctive_association | `R/scripts/oracle_lens/latent_eval/judge_mc.py` | `.../multi_token/conjunctive_association/items.json` (100) |

Judge for all four: the repo default (`JudgeConfig()` → `google/gemini-3.8-flash`), i.e. this PR
changes the instrument from Opus to Gemini; the source-repo `--model` defaults are NOT carried
over. Phase 7 measures the change.

## Shared additions to the package (small, all in this PR)

1. **`readouts.Cell.token: str | None`** (optional, default `None`). Loader accepts an optional
   string `token` field per row; converter passes the gen-dir row's `token` through. Needed by
   relational_multihop's blank-token check. Extend `test_readouts.py` accordingly.
2. **`wsbench/summarizer.py`** — the one token-bag → prose step for J-lens (`kind == "tokens"`)
   readouts. Prompt = the `INTERP_SYSTEM` / `INTERP_SCHEMA` of
   `R/scripts/oracle_lens_evals/oa_eb_readout_judge.py` L52-58 (identical to
   `judge_relational_multihop.py` L88-93), user message `f"TOKEN READOUTS:\n{txt}"`.
   `SUMMARIZER_PROMPT_VERSION = "interp-v1"`.
   ```python
   def render_bag(tokens: Sequence[str], scores: Sequence[float] | None) -> str   # "tok (score)" list, best first, scores to 2 dp; no scores → tokens joined by " | "
   def summarize(bundles: Mapping[str, str], *, judge: ResolvedJudge, cache: Cache, spend: Spend,
                 concurrency: int, rpm: float, dry_run: bool = False) -> dict[str, str | None]
       # key → interpretation text (None on failure); cache key f"summ:{key}", fp = fingerprint(SUMMARIZER_PROMPT_VERSION, model, bundle_text)
       # spend is accumulated in place; dry_run → prints the first bundle's rendered prompt and returns {} (no client)
   ```
   The `bundle_text` handed to the summarizer per family: moral / role-bound =
   `render_bag(cell.tokens, cell.scores)` for that row; relational = the bundle template with
   `txt = render_bag(...)` (the `[position …]` prefix IS part of the summarizer input, as in the
   source L269; the source sends that bundle with NO `TOKEN READOUTS:` prefix, whereas the
   shared summarizer always adds it — a deliberate deviation, recorded in the relational
   README legacy note); conjunctive = the per-(layer, id) `" | "` blob of `norm_token`'d tokens across
   positions (source L151-155). The judge then sees the interpretation text in place of the
   readout string.
   The summarizer model is `args.aux_models.get("summarizer", judge.model)`: add
   `aux_models: Mapping[str, str]` to `registry.JudgeArgs` (default empty) and have
   `cli._judge_family` pass `spec.judge.aux_models`. All four families leave it at the default. Families with a failed summary skip that cell (counted in
   `n_unjudged_cells`), never judge an empty string. Note in each family README that the source
   scripts each carried their own near-identical interp prompt (ec: "top-k" wording;
   conjunctive: `JLENS_SUMMARIZER_*` from `oracle_lens_judge.py` L1021-1039) and that this repo
   uses one shared prompt; the originals are quoted in the README as "legacy" for provenance.
3. **`wsbench/mc.py`** — helpers every MC family uses:
   ```python
   CANNOT = "cannot tell from the readout"
   def seed_int(s: str) -> int                    # int.from_bytes(sha256(s.encode()).digest()[:8], "big")
   def seeded_shuffle(items: Sequence[T], seed: str) -> list[T]   # random.Random(seed_int(seed)).shuffle on a copy
   def listing(options: Sequence[str]) -> str     # "\n".join(f"  {i+1}. {o}")
   def classify(choice: Any, gold_pos: int, n_shown: int, *, contrast_pos: int | None = None) -> str
       # "invalid" (not int or out of 1..n_shown) | "cannot_tell" (== n_shown, CANNOT is always last) | "gold" | "contrast" | "distractor"
   def join_samples(samples: Sequence[str]) -> str   # "\n".join(s for s in samples if s.strip())
   ```
4. **`registry.EvalSpec.positions: Callable[[list[Cell]], dict[str, list[int]]] | None`** — NOT
   added; each family selects its cells inside `judge.py` (see per-family rules). Grid
   expectations are computed from the readouts file itself (bank has no per-item position list
   for these families: `read_position == "end_of_stimulus"` everywhere).
5. **Family package layout** (each): `src/wsbench/evals/<family>/{__init__,prompts,judge,score}.py`;
   `__init__` registers the `EvalSpec`. `judge.run(args: JudgeArgs) -> FamilyResult` is the
   entrypoint. Per-cell verdict rows go into `FamilyResult.rows`; the append-only cache lives at
   `<out>/cells.jsonl`; `results.json` is written by the CLI.
6. **README-prompt equality test**: `tests/test_prompts_in_readme.py` iterates every registered
   family, reads `evals/<family>/README.md`, and asserts every string in the family's
   `prompts.PROMPTS: dict[str, str]` appears verbatim inside a fenced block of the README. The
   shared summarizer prompt is covered the same way against `docs/summarizer.md` (new file:
   what the summarizer is, the prompt, the schema, the version).
   Each family's `prompts.py` therefore exports `PROMPTS = {"SYSTEM": SYSTEM, ...}` covering
   every constant and every f-string *template*. Templates use `{name}` placeholders and are
   rendered by `prompts.render_*` functions with **`str.replace` per placeholder, never
   `str.format`** — the role-bound header (source L126-130) contains literal
   `{person A} -> {action} -> {person B}` which must survive unescaped so README equality holds.

## Common judge loop (implement once in `wsbench/mcjudge.py`, used by all four)

```python
@dataclass
class Call: key: str; system: str; user: str; meta: dict   # meta carries gold positions etc.
def run_calls(calls: list[Call], *, schema: dict, judge: ResolvedJudge, prompt_version: str, cache: Cache,
              spend: Spend, concurrency: int, rpm: float, dry_run: bool) -> dict[str, dict | None]
    # one schema per batch (every family uses exactly one judge schema); Call has no schema field
    # fp = fingerprint(prompt_version, judge.model, judge.reasoning, system, user)
    # cached (non-failed) results are reused; the rest go through llm.stream_json in ONE batch with spend=spend;
    # every landed result (incl. None) is cache.put(...)'d immediately; returns key → result|None.
    # dry_run: prints the first call's system+user and returns {} without any client.
```

**Cache payload and rows.** `run_calls` writes `cache.put(call.key, fp, {"result": r, "meta": call.meta})`
(the payload never contains `key` / `fp` / `ts`, which `Cache.put` reserves). `FamilyResult.rows`
are rebuilt from results + meta by the family; a row's `key` is the call key (relational: the
cell key, with the two calls keyed `f"{cell.key}:X"` / `f"{cell.key}:Y"`).

**Dry run.** `judge.run` still returns a `FamilyResult` (the CLI always writes it):
`value=None, ci95=None, rows=[]`, counts with `spend_usd=0.0`, `config["dry_run"]=True`,
`complete=False`. For `tokens` readouts a dry run prints only the summarizer prompt (no judge
prompt exists before a summary).

**`config` and `complete`, all families.** `config = {judge_model, prompt_version, reasoning,
layers, dry_run, concurrency, rpm, allow_missing, items, limit}` plus family extras (e.g.
`char_cap`); `complete = completeness(pinned=judge.pinned, subset=(items or limit or layers given),
n_expected=..., n_missing=0, n_unjudged=..., n_empty=...)`; `counts["skipped_rows"] = sum(rep.skipped.values())`.

`preflight` runs once per family run, before the first uncached call of either stage (the
summarizer batch precedes the judge batch), and never under dry_run. The family creates one
`Spend`, passes it to both stages, and writes `spend.usd` into `counts["spend_usd"]`.

**Counts, for all four families** (no bank position list exists): `n_expected_cells` = cells
selected by the family rule (tail-5 rows / blank rows / all rows / (layer, id) bags);
`n_missing_cells` = 0 always and `--allow-missing` is a no-op (say so in `--help`);
`n_empty_cells` = selected cells whose readout text is empty after `join_samples` — these are
**skipped, not judged** (source L273-275 / L228-230); `n_unjudged_cells` = selected non-empty
cells with no verdict (API failure, summary failure, or a dropped X/Y pair). Denominators:
`n_items` = items in scope = bank ∩ `--items`, then `--limit`; `value = passes / n_items`; an
in-scope item with no judged cell counts as a fail.

## Per-family specs

### moral_rationale (`evals/moral_rationale/items.json` = source bank verbatim, 200 items)

- prompts.py: copy `EC_SEED = 20260812`, `SYSTEM` (L68-79), `SCHEMA` (L81-85, via
  `llm.schema_block("ec_reason_mc", …)`), `committed_question` (L161-165) and
  `deliberative_question` (L168-179) as templates; user message template
  `"READOUT:\n{readout}\n\n{q}"` (L303). `PROMPT_VERSION = "ec-v1"`.
- Options (L102-154, L182-209): pools from the bank — committed pool = every item's
  `look_for_reasons[0]`; yes/no pools = every `look_for_reasons` entry whose matching
  `reasons[].supports` is `yes`/`no`. Filter out same `topic_id`, normalized duplicates
  (`norm` L88-89) and token-Jaccard > 0.6 (`jaccard` L92-95) vs the gold. Distractor seed string
  `f"{EC_SEED}:{id}:{side}"` with side ∈ {`committed`, `yes`, `no`}, `Random(seed_int).shuffle(uniq)`,
  take 4 (`short_pool` if fewer). Order seed `f"{EC_SEED}:{id}:{side}:order"`. Shown =
  seeded_shuffle([gold] + dists) + [CANNOT] → 6 lines.
- Cells: rows from the readouts file; per (id, layer) keep the **last 5 positions** (sorted by
  pos) — `--tail-pos` is not exposed; 5 is the instrument. Call unit = (id, layer, pos, side);
  committed items 1 call, deliberative (`reason_class == "deliberative"`) 2 calls. Readout text =
  `join_samples(samples)`; tokens kind → summarizer per row first.
- Verdict row: `{key, id, layer, pos, reason_class, side, choice, gold_pos, n_options,
  short_pool, pick, correct, quote}`. Item pass: committed = any correct; deliberative = (any
  yes correct) and (any no correct), sides independent.
- FamilyResult: `metric="pass_rate"`, `value` = passes / n_items (200 when unrestricted), `ci95` =
  bootstrap over item pass indicators, `chance=None`,
  `chance_label="per call 1/6 (committed), 1/36 (deliberative); any-of-grid floors saturate — do not quote"`,
  `extras={"committed": {"n","pass","rate"}, "deliberative": {"n","both_sides","yes_any","no_any","rate"},
  "any_of_grid_floor": {"committed","deliberative"}, "short_pool_items": [...], "n_api_failed"}`.
  Copy the 🚨 DO-NOT-QUOTE comment from L386-395 above the floor computation.
- README: sources block = none (in-house). Judge prompts section with SYSTEM, both question
  templates, user template, schema, option rule, seeds, CANNOT wording.

### relational_multihop (`evals/relational_multihop/items.json` = `cloze_items_final.json` verbatim, 100)

- prompts.py: `REL_SEED = 20260813`, `KINSHIP` set, `KIN_EXTRA`, `PROF_EXTRA` (+ rationale
  comment L77-83), `JUDGE_SYSTEM` (L95-104), `MC_SCHEMA` (L105-107), question template
  (L190-193) with `{readout}`, `{role}`, `{listing}`; role strings
  `"X (the OUTER/first relation word)"` / `"Y (the INNER/second relation word)"`; bundle
  template `"[position {token!r}] {txt}"` with samples joined by `" | "` (L209-214).
  `PROMPT_VERSION = "rel-v1"`.
- Options (L117-165): kin pool = bank kinship relations ∪ KIN_EXTRA; prof pool = bank
  professional ∪ PROF_EXTRA (assert disjoint). **Both pools are `sorted(...)` lists** (source
  L120-121) before sampling. `rng = Random(seed_int(f"{REL_SEED}:dist:{id}:pooled"))`;
  `rng.sample(sorted kin − hops, 4)` THEN `rng.sample(sorted prof − hops, 4)` (order matters);
  the pre-shuffle list is exactly `[hop1, hop2, *kin_dist, *prof_dist]` (L161); shuffle seed
  `f"{REL_SEED}:{id}:pooled"`; assert 10 distinct; + CANNOT → 11 lines. Both sub-questions share
  the list.
- Cells: per (id, layer) the **max-pos row** is the blank; if the row has a `token`, assert it
  equals `"'s"` (raise with the item id otherwise); if `token` is None, accept and render the
  bundle prefix as `[position p{pos}]` instead of `[position {token!r}]`. Two calls per
  cell (X, Y), same readout, same list. Tokens kind → summarizer on the bundle first.
- Verdict row: `{key, id, layer, pos, x_choice, x_gold_pos, x_ok, x_quote, y_choice, y_gold_pos,
  y_ok, y_quote, pass}`; cell pass = both; item pass = any layer.
- FamilyResult: `value` = items passing / n_items, `chance = None`,
  `chance_label="per cell 1/121; any-of-layers floor 1-(120/121)^n_layers"`,
  `extras={"per_layer_pass": {"L20": {"pass", "judged"}, …}, "pair_consistency": {"n": scenes where both `rel-{scene}-a` and `-b` pass (source L321-323), "of": <number of scenes in the bank>},
  "any_of_layers_floor", "n_layers", "n_cells_judged", "n_cells_api_failed", "n_cells_interp_missing"}`.
- README: in-house. Prompts section as above incl. KIN_EXTRA/PROF_EXTRA and the draw order.

### role_bound_association (`evals/role_bound_association/items.json` verbatim, 100)

- prompts.py: `DATASET_SEED = 20260810`, `SYSTEM` (L60-67), schema (L72-83, `with_foils=False`
  only: `q1_choice, q2_choice, q3_choice, evidence`), Q texts (L121-125), header (L126-130),
  block assembly (L131), `mc_block` (L93-99), user template `"READOUT:\n{readout}\n\n{block}"`.
  The entity-binding branch (`EB_VERBS`, `eb_question`, foils) is NOT ported.
  `PROMPT_VERSION = "oa-v1"`.
- Options (L102-131): labels `f"{names[a]}, the {role_a}"`; `rng = Random(seed_int(f"{DATASET_SEED}:d:{id}"))`;
  people = [a, b] + 3 `a`-labels sampled from `other` (items with a different `pair_id`, in
  bank order) — sampled FIRST; then actions = [action] + `rng.sample(sorted(other actions − {action}), 4)`
  from the SAME `rng` (source L112-119); gold agent/patient from
  `direction` (`"ab"` → a agent). Per-question shuffle seeds `f"{DATASET_SEED}:{id}:q{1|2|3}"`;
  + CANNOT → 6 lines each; Q1 and Q3 independent shuffles of the same people list.
- Cells: **every row** (all layers × all positions) — the instrument; `--layers`/`--items`
  restrict. One call per row, three MCs in one response. Tokens kind → summarizer per row.
- Verdict row: `{key, id, layer, pos, correct: [bool,bool,bool], pass, evidence}`; item pass =
  any row.
- FamilyResult: `value` = items passing / n_items, `chance=None`,
  `chance_label="per site (1/6)^3; any-of-grid floor saturates — do not quote"`,
  `extras={"any_of_grid_floor", "sites_per_item": {min, median, max}, "n_api_failed",
  "by_stereotypicality": {congruent|incongruent|neutral: {n, pass}}}` (stratum from the bank).
  Copy the 🚨 banner L350-358.
- README: in-house.

### conjunctive_association (`evals/conjunctive_association/items.json` verbatim, 100)

- prompts.py: `COMP_SEED = 20260805`, `SYSTEM` (L46-58), `SCHEMA` (L60-62), question template
  (L76-85), user template `"READOUTS:\n{r}\n\n{q}"`. `PROMPT_VERSION = "comp-v1"`.
- Options: the item's frozen `mc_options` deduped order-preserving, gold asserted present,
  shuffle seed `f"{COMP_SEED}:{id}"`, + CANNOT → 11 lines.
- Cells: **one call per item** over a blob built as the source does (`score_lens_readouts.py`
  L26-46, copied into `judge.py` as `norm_token`/`BPE_JUNK`/`build_blob`): per (layer, id) join
  every row's samples with `" | "` after `norm_token` (strip `Ġ Ċ ▁`, `_`→space; note this is
  applied to prose samples too, so AO text loses underscores — say so in the README) in **file
  order** (not sorted by pos, as the source), then `blob += f"\n[L{layer}] " + text` over
  ascending layers. Items missing any selected layer are excluded with a warning (counted
  `n_items_excluded`, outside the denominator). `char_cap` default **200000** (the full-blob
  instrument of record; the source default 24000 is not carried): slice to `char_cap` FIRST,
  then strip non-printables (source L172). Tokens kind → summarizer per (layer, id) bag before blobbing.
- Verdict row: `{id, choice, gold_pos, contrast_pos, pick, correct, quote, quote_ok}`
  (`quote_ok` recorded, not enforced). Item pass = pick == gold.
- FamilyResult: `value` = passes / n_items where n_items = in-scope items minus excluded-layer
  items (api_fail rows stay in the denominator, source L241), `chance = 1/11`,
  `chance_label="1/11 per item (single call, no grid)"`,
  `extras={"breakdown": {gold, contrast, distractor, cannot_tell, invalid, api_fail}, "char_cap",
  "n_items_excluded"}`.
- README: in-house.

## CLI additions

`judge` gains nothing new; `--layers`, `--items`, `--limit`, `--allow-missing` apply. `--dry-run`
prints the first rendered call. Each family also accepts family-specific options via
`JudgeArgs.extra: dict[str, str]` populated from `--opt key=value` (repeatable) — used only by
conjunctive (`char_cap`). Add `extra` to `JudgeArgs` and `--opt` to the parser.

## Examples

`examples/readouts/<family>.jsonl`: hand-written toy prose readouts for the first 2 bank items,
2 layers, enough positions to exercise the cell rule (moral: 6 positions so tail-5 drops one;
relational: 2 positions with `token` on the last; role-bound: 3 positions; conjunctive: 2
positions). Plus `examples/readouts/<family>.tokens.jsonl` for relational only (summarizer path).

## Tests (offline; fake `llm.stream_json` via monkeypatch as in phase 1)

Per family `tests/test_<family>.py`:
- bank loads with the expected count and required keys;
- option construction is deterministic and matches a **golden file**
  `tests/golden/<family>_options.json` generated ONCE by running the *source* script's option
  functions over the whole bank. The source scripts import `global_workspace.judges.llm_client`
  (and judge_mc pulls `score_lens_readouts` → `score_readout`, `lemma_scan`, `oracle_lens_judge`),
  so the maker must run **inside the source repo's environment**: `cd R && uv run --no-sync
  python <path to this repo>/tests/golden/make_<family>.py`, importing the source script by
  path after `sys.path.insert(0, R/src)` and `R/scripts/...`. Golden contents: moral
  `{id: {side: {shown, gold_pos, short_pool}}}`; relational `{id: {shown, x_gold_pos, y_gold_pos}}`;
  role-bound `{id: {block, golds: [g1, g2, g3]}}` (the full `oa_question` block, since
  `mc_block` is internal); conjunctive `{id: {shown, gold_pos}}`. Goldens are committed; makers
  are committed under `tests/golden/` and excluded from pytest collection — this is the parity
  gate for the port;
- rendered prompt for item 1 equals a golden string;
- cell selection rule (tail-5 / blank / all / blob) on the toy readouts;
- verdict parsing incl. invalid / cannot_tell / None → per-cell labels → item pass → numbers;
- resume: second run makes zero calls;
- tokens kind routes through the summarizer (fake) before judging;
- `--dry-run` makes no calls and prints a prompt.
Shared: `test_mc.py` (classify, seeds, listing), `test_summarizer.py`, `test_prompts_in_readme.py`.

## Acceptance

ruff + pytest green; `wsbench list` shows the four families with n items 200/100/100/100 and
judge `google/gemini-3.8-flash`; `wsbench judge <family> --readouts examples/readouts/<family>.jsonl
--out /tmp/x --dry-run` prints a prompt for each; goldens match the source scripts.
