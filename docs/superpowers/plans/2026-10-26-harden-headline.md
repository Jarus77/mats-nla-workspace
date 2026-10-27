# Harden the Headline Cells Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-score the two headline cells with full candidate sets, put bootstrap CIs on exps 15 and 16, test whether the statement direction is lexical, cross-score with Guiv's bank when it arrives, and pre-register the week-two prediction test.

**Architecture:** One new experiment directory `experiments/17_harden/` in the exp 16 shape (env / spec / pure lib with tests / Modal stages with opaque names / report script). Pure functions (candidate construction, multi-candidate metrics, vocabulary statistics, bootstrap wrappers) live in `lib17.py` and are unit-tested on toy arrays; the Modal stages only load readers, call the existing `logprob_of_strings` family, and save `.pt` / `.json`. CI computation for the earlier experiments is a CPU script reading the local `results/vol/*.pt` files. The pre-registration is a markdown document with a hash step.

**Tech Stack:** Python 3.12, torch, numpy, Modal (apps/volumes as in `infra/modal_common.py`), the repo's `src/mats` readers and metrics, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-harden-headline-design.md`

## Global Constraints

- Modal profile `glitchcraft-inc`; `max_containers=int(os.environ.get("RFM_MAXC", "1"))`; never more than two GPUs.
- Modal app name `b7kx`; function names `q2vn` (OL + J-lens multi-candidate, H100), `f8rt` (NLA multi-candidate, H200), `y4lm` (vocabulary readout, CPU), `n6wd` (cross-score OL + J-lens, H100), `k9zc` (cross-score NLA, H200), `g1ps` (input check, CPU). CLI keys: `mc_ol`, `mc_nla`, `vocab`, `xs_ol`, `xs_nla`, `noop`.
- Every stage returns `{"feature": f, "skipped": True}` when its output path exists.
- Outputs on the volume under `/results/17_harden/`; small JSON prints under `experiments/17_harden/results/`; `.pt` under `experiments/17_harden/results/vol/` (git-ignored); `experiments/17_harden/inputs/` git-ignored.
- Frozen phrasings read through `templates_path("13_new_rows")` (same resolver as exps 15/16); the file is never edited.
- Commits as Jarus77, `<type>: <description>`, no trailers; no push.
- Tests: `.venv/bin/python -m pytest experiments/17_harden -q`.

## Review Focus

1. A table-column item whose `meta.col` is outside {1, 4, 5, 6} (generator drift): `true_index` must raise, not silently mis-score. Test in Task 1.
2. A parenthesis item whose text contains an unmatched `"` or `[`: the four-way distractors would be true, so the item must be excluded and counted. Test in Task 1.
3. Candidate strings of unequal token length bias summed log-probs: the metrics must be computed per template, and the report must show per-template rows, never only the mean. Test in Task 1 (`metrics_multi` keeps the template axis).
4. A trimmed or partial `.pt` from exp 15 (NaN null rows) must not crash the CI script: NaN rows are dropped as in `analyze15.py`. Test in Task 4.
5. `lm_head.weight` absent from the safetensors index (tied embeddings): the vocabulary stage falls back to `model.embed_tokens.weight` and records which it used. Test in Task 5 on a fake index dict.

---

### Task 1: Candidate sets and multi-candidate metrics (pure, tested)

**Files:**
- Create: `experiments/17_harden/spec17.py`
- Create: `experiments/17_harden/lib17.py`
- Test: `experiments/17_harden/test_lib17.py`

**Interfaces:**
- Produces, in `spec17.py`:
  - `ROWS = {"tb_tablecol": {"values": ["first", "late"], "n_cands": 6}, "tb_paren": {"values": ["inside", "outside"], "n_cands": 4}}`
  - `COL_ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth"]`, `COL_CARDINAL = ["one", "two", "three", "four", "five", "six"]`, `COL_DIGIT = ["1", "2", "3", "4", "5", "6"]`
  - `PAREN_STATES = ["paren_in", "paren_out", "quote_in", "bracket_in"]`
  - `MC_TEMPLATES: dict[str, dict[str, list[dict]]]` keyed row then reader (`nla` / `ol` / `jlens`), each entry `{"id": str, "template": str, "slots": list[str]}` with `len(slots) == n_cands`, in candidate order. Table column: NLA `{"id": "nla:mc:fixed", "template": "The current cell is in the {v} column of the table", "slots": COL_ORDINAL}`; OL `ol:mc:fixed` `- column {v}` with `COL_CARDINAL`, `ol:mc:m3` `- col_{v} |` with `COL_DIGIT`, `ol:mc:m0` `- column {v} |` with `COL_DIGIT`; J-lens `jlens:mc:fixed` with slots as word lists `[["first","one"],["second","two"],["third","three"],["fourth","four"],["fifth","five"],["sixth","six"]]`. Parenthesis: NLA `nla:mc:state` `The current position is {v}` with `["inside a parenthesis", "outside any parenthesis", "inside a quotation", "inside square brackets"]`; OL `ol:mc:state` `- {v}` with `["inside parentheses", "outside parentheses", "inside quotes", "inside brackets"]`; J-lens `jlens:mc:state` with `[["inside","parenthesis"],["outside"],["quotation","quote"],["bracket","brackets"]]`.
  - `CLASS_SETS = {"tb_tablecol": {"first": [0], "late": [3, 4, 5]}, "tb_paren": {"inside": [0], "outside": [1]}}` (0-based candidate indices per class).
  - `N_BOOT = 2000`, `SEED = 17`, `LAYER = 42`, `OL_LAYERS = (42,)`.
- Produces, in `lib17.py`:
  - `candidates(template: dict) -> list[str]`: the `n_cands` filled strings in slot order (J-lens entries are returned as the word lists unchanged).
  - `true_index(row: str, item: dict) -> int`: table column → `item["meta"]["col"] - 1`, raising `ValueError` unless `col in {1, 4, 5, 6}`; parenthesis → 0 for `value == "inside"`, 1 for `"outside"`.
  - `paren_item_ok(text: str) -> bool`: False when the count of `"` is odd or `[` count exceeds `]` count.
  - `metrics_multi(lp: np.ndarray, truth: np.ndarray, y: np.ndarray, class_sets: dict[str, list[int]], values: list[str], clusters: np.ndarray, n_boot: int, seed: int) -> dict`: `lp` is `[N, C]` summed log-probs for one template. Returns `{"top1": {"point", "lo", "hi", "chance": 1 / C}, "mrr": {...}, "auroc_lse": {"point", "lo", "hi"}, "spearman_argmax": float | None, "n": N}` where `auroc_lse` is the AUROC of `logsumexp(lp[:, class_sets[values[1]]]) - logsumexp(lp[:, class_sets[values[0]]])` for `y == 1` (values[1]) against `y == 0`, `spearman_argmax` is Spearman between `argmax(lp)` and `truth` (None when truth has fewer than 3 distinct values), and every CI is `mats.metrics.cluster_bootstrap` over `clusters`.
  - `lse_pair_lo(lp: np.ndarray, class_sets, values) -> np.ndarray`: the `[N]` log-sum-exp contrast used by `auroc_lse` (exposed for the report).

- [ ] **Step 1: Write the failing tests**

```python
def test_candidates_fill_in_slot_order():
    t = {"id": "ol:mc:fixed", "template": "- column {v}", "slots": ["one", "two"]}
    assert L.candidates(t) == ["- column one", "- column two"]

def test_true_index_table_column_and_drift():
    assert L.true_index("tb_tablecol", {"meta": {"col": 5}}) == 4
    with pytest.raises(ValueError):
        L.true_index("tb_tablecol", {"meta": {"col": 2}})

def test_paren_item_filter():
    assert L.paren_item_ok('He said (as before) that')
    assert not L.paren_item_ok('He said "as before (that')
    assert not L.paren_item_ok('He said [as before (that')

def test_metrics_multi_perfect_and_chance():
    lp = np.full((6, 6), -5.0); truth = np.array([0, 0, 0, 3, 4, 5]); y = np.array([0, 0, 0, 1, 1, 1])
    lp[np.arange(6), truth] = 0.0
    m = L.metrics_multi(lp, truth, y, S.CLASS_SETS["tb_tablecol"], S.ROWS["tb_tablecol"]["values"],
                        np.arange(6), n_boot=50, seed=0)
    assert m["top1"]["point"] == 1.0 and m["top1"]["chance"] == pytest.approx(1 / 6)
    assert m["auroc_lse"]["point"] == 1.0 and m["mrr"]["point"] == 1.0
    flat = np.zeros((6, 6))
    m0 = L.metrics_multi(flat, truth, y, S.CLASS_SETS["tb_tablecol"], S.ROWS["tb_tablecol"]["values"],
                         np.arange(6), n_boot=50, seed=0)
    assert m0["auroc_lse"]["point"] == pytest.approx(0.5)

def test_lse_pair_lo_matches_two_candidate_when_classes_are_singletons():
    lp = np.array([[0.0, -1.0], [-2.0, 0.0]])
    out = L.lse_pair_lo(lp, {"inside": [0], "outside": [1]}, ["inside", "outside"])
    assert np.allclose(out, lp[:, 1] - lp[:, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest experiments/17_harden/test_lib17.py -q`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` on `lib17`.

- [ ] **Step 3: Implement `spec17.py` constants and the five `lib17.py` functions**

`metrics_multi` uses `mats.metrics.auroc` for the point AUROC and `cluster_bootstrap` for each CI; `top1` is `mean(argmax(lp, 1) == truth)`; `mrr` is the mean of `1 / (1 + rank of truth)` with rank counted as the number of candidates with a strictly larger log-prob. `spearman_argmax` uses `scipy.stats.spearmanr`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest experiments/17_harden/test_lib17.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add experiments/17_harden/spec17.py experiments/17_harden/lib17.py experiments/17_harden/test_lib17.py
git commit -m "feat: exp 17 candidate sets and multi-candidate metrics (six column values, four enclosure states), tested"
```

---

### Task 2: Modal stages for the multi-candidate re-score

**Files:**
- Create: `experiments/17_harden/env.py` (copy `experiments/16_difference_term/env.py`; set `OUT = "/results/17_harden"`, `SRC13 = "/results/13_new_rows"`, `SRC15 = "/results/15_ar_steering"`, `_LOCAL = ("modal_common", "env", "spec17", "lib17")`)
- Create: `experiments/17_harden/modal17.py`
- Create: `experiments/17_harden/.gitignore` with `inputs/` and `results/vol/`

**Interfaces:**
- Consumes: Task 1's `MC_TEMPLATES`, `candidates`, `true_index`, `paren_item_ok`; `mats.readers.olens.load_olens().logprob_of_strings(acts, layer, cands, batch)`; `mats.readers.nla.logprob_of_strings_batch(nla, acts, cands, batch)`; `mats.readers.jlens.load_jlens(model, tok, layers).scores(acts, layer)` and `token_variants(tok, words)`.
- Produces: `/results/17_harden/mc_ol_{row}.pt` and `mc_nla_{row}.pt`, each `{"ids": [...], "truth": np.ndarray, "y": np.ndarray, "clusters": [...], "excluded": [...ids], "lp": {template_id: tensor[N, C]}}`; for J-lens the per-candidate score is `max over token_variants(words) of jl.scores(acts, 42)[:, ids]` stored under the same `lp` key (`jlens:mc:*`).
- Item selection: `split == "select"` items of the row from `/results/13_new_rows/items_base.json` (160), activations from `/results/13_new_rows/acts_{row}.pt["acts"][42]` with index alignment asserted on ids, parenthesis items filtered by `paren_item_ok` (excluded ids recorded).
- Entry point: `modal run modal17.py --stage mc_ol|mc_nla|vocab|xs_ol|xs_nla|noop --features tb_tablecol,tb_paren`; after each call `results/{stage}17_{row}.json` holds the stage's small print.

- [ ] **Step 1: Write `env.py`, `modal17.py` with `q2vn` (OL + J-lens) and `f8rt` (NLA) only, plus `g1ps` and the entry point**

`q2vn` loads the base model once, `load_olens`, scores every `ol:mc:*` template at layer 42 with `batch=max(64, 2 * C)`, then `load_jlens(ol.model, tok, layers=[42])` for the `jlens:mc:*` template. `f8rt` mirrors exp 16's `u4yd` loading (`NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)`).

- [ ] **Step 2: Dry-run the input check**

Run: `cd experiments/17_harden && modal run modal17.py --stage noop --features tb_tablecol,tb_paren`
Expected: two lines `noop {"feature": ..., "missing": [], "skipped": true}`.

- [ ] **Step 3: Run the OL + J-lens stage, then the NLA stage (one GPU each, sequential)**

Run: `RFM_MAXC=1 modal run modal17.py --stage mc_ol --features tb_tablecol,tb_paren` then `--stage mc_nla`.
Expected: `results/mc_ol17_{row}.json` and `results/mc_nla17_{row}.json` written, each reporting `n`, `n_excluded`, `wall_s`; `.pt` files on the volume.

- [ ] **Step 4: Fetch the `.pt` files locally**

Run: `modal volume get rfm-results 17_harden/ experiments/17_harden/results/vol/`
Expected: four `.pt` files present.

- [ ] **Step 5: Commit (code and small JSON only)**

```bash
git add experiments/17_harden/env.py experiments/17_harden/modal17.py experiments/17_harden/.gitignore experiments/17_harden/results/mc_*17_*.json
git commit -m "feat: exp 17 multi-candidate re-score stages (OL + J-lens, NLA) on the exp 13 selection items"
```

---

### Task 3: Multi-candidate analysis and report

**Files:**
- Create: `experiments/17_harden/analyze17.py`
- Create: `experiments/17_harden/report17.py`
- Create: `experiments/17_harden/RESULTS.md`
- Test: `experiments/17_harden/test_lib17.py` (add one test)

**Interfaces:**
- Consumes: Task 2's `mc_*.pt`, Task 1's `metrics_multi`.
- Produces: `analyze17.mc_table(row: str, vol_dir: Path) -> dict` returning `{"row": row, "n": N, "excluded": [...], "cells": {reader: {template_id: metrics_multi(...)}}, "two_candidate_reference": {reader: auroc of the frozen fixed pair from exp 13 score files}}`, written to `results/analysis17_mc_{row}.json`; `report17.markdown(results: list[dict]) -> str` with one row per (reader, template): point and CI for top-1 (with chance), MRR, log-sum-exp AUROC, Spearman; and a verdict line per row applying the spec's declared criteria.

- [ ] **Step 1: Write the failing test**

```python
def test_mc_table_reads_pt_and_keeps_template_axis(tmp_path):
    blob = {"ids": ["a", "b"], "truth": np.array([0, 3]), "y": np.array([0, 1]), "clusters": ["a", "b"],
            "excluded": [], "lp": {"ol:mc:fixed": torch.zeros(2, 6), "ol:mc:m3": torch.zeros(2, 6)}}
    torch.save(blob, tmp_path / "mc_ol_tb_tablecol.pt")
    out = A.mc_table("tb_tablecol", tmp_path, n_boot=20)
    assert set(out["cells"]["ol"]) == {"ol:mc:fixed", "ol:mc:m3"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest experiments/17_harden/test_lib17.py -q -k mc_table`
Expected: FAIL, `analyze17` missing.

- [ ] **Step 3: Implement `analyze17.mc_table` and `report17.markdown` / `report17.main` (`--analysis` paths, `--out results/mc17.md`)**

`mc_table` reads whichever of `mc_ol_{row}.pt` / `mc_nla_{row}.pt` exist (missing reader → omitted, not an error). The two-candidate reference reads `experiments/13_new_rows/results/score_{ol,nla}_{row}.json` `scores[...]["lo"][...fixed]` and recomputes the AUROC with `mats.metrics.auroc_labels`.

- [ ] **Step 4: Run tests, then the analysis on the real files**

Run: `.venv/bin/python -m pytest experiments/17_harden -q` then `.venv/bin/python experiments/17_harden/analyze17.py --rows tb_tablecol,tb_paren` then `.venv/bin/python experiments/17_harden/report17.py --analysis results/analysis17_mc_*.json`.
Expected: tests pass; `results/mc17.md` contains one table per row and a verdict line reading either "survives" or "two-candidate artefact" per the spec's thresholds.

- [ ] **Step 5: Write `RESULTS.md` (question, design, the two tables, verdict, caveats) and commit**

```bash
git add experiments/17_harden/analyze17.py experiments/17_harden/report17.py experiments/17_harden/RESULTS.md experiments/17_harden/results/analysis17_mc_*.json experiments/17_harden/results/mc17.md
git commit -m "docs: exp 17 multi-candidate re-score of table column and parenthesis, with CIs and the declared verdict"
```

---

### Task 4: Bootstrap CIs for exps 15 and 16 (CPU, local files)

**Files:**
- Create: `experiments/17_harden/ci_prior.py`
- Test: `experiments/17_harden/test_ci_prior.py`
- Modify: `experiments/15_ar_steering/RESULTS.md` (add a "Confidence intervals" paragraph + table), `experiments/16_difference_term/RESULTS.md` (same)

**Interfaces:**
- Consumes: `experiments/15_ar_steering/results/vol/{behave15_{row}_c{0,1}.pt, score15_{nla,ol}_{row}.pt, spec15_{row}.pt}` through `analyze15._series` / `_index` (import from that directory), `experiments/16_difference_term/results/vol/{s1,s2}_{row}.pt` and `lib16.follow`.
- Produces:
  - `ci15(row: str, vol: Path, n_boot: int, seed: int) -> dict`: for every `(direction, "r11")` cell and every series key (model metric, each reader's mean phrasing), the paired mean shift with a cluster-bootstrap CI over items (clusters = `spec["ids"]` cluster prefix, i.e. the id up to the last `_`), the null-band z as already reported, and `z_lo`/`z_hi` = the shift CI endpoints divided by the null SD. NaN null rows are dropped as in `analyze15`.
  - `ci16(row: str, vol: Path, n_boot: int, seed: int) -> dict`: for every reader and variant, `follows_j` with a cluster-bootstrap CI over `s1["clusters"]`.
  - `main()` writes `results/ci15_{row}.json`, `results/ci16_{row}.json`, and `results/ci_prior.md` (two tables: exp 15 r11 shifts with CI and z; exp 16 follows_j with CI).

- [ ] **Step 1: Write the failing tests** (synthetic `.pt` files in `tmp_path` shaped like the real ones; a NaN null row for exp 15)

```python
def test_ci16_follows_j_ci_contains_point(tmp_path):
    # 8 items, 4 clusters, readout equal to partner's label => follows_j == 1
    ...
    out = C.ci16("tb_tablecol", tmp_path, n_boot=50, seed=0)
    r = out["readers"]["nla"]["swap_res"]["mean"]
    assert r["lo"] <= r["point"] <= r["hi"] and r["point"] == 1.0

def test_ci15_drops_nan_null_rows(tmp_path):
    ...  # one null condition stored as all-NaN
    out = C.ci15("tb_tablecol", tmp_path, n_boot=50, seed=0)
    assert np.isfinite(out["cells"]["cm_raw:r11"]["nla_mean"]["z"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest experiments/17_harden/test_ci_prior.py -q`
Expected: FAIL, `ci_prior` missing.

- [ ] **Step 3: Implement `ci_prior.py`** (sys.path insert of the two experiment directories; reuse `analyze15._series`, `_index`, `lib15.band`, `lib16.follow`, `mats.metrics.cluster_bootstrap`)

- [ ] **Step 4: Run tests, then the real files**

Run: `.venv/bin/python -m pytest experiments/17_harden -q` then `.venv/bin/python experiments/17_harden/ci_prior.py --rows15 tb_tablecol,tb_paren --rows16 tb_tablecol,tb_paren,ij2_lang,bk_listidx`
Expected: `results/ci_prior.md` with both tables; every exp 16 headline `follows_j` has a CI; every exp 15 r11 cell has `z_lo`/`z_hi`.

- [ ] **Step 5: Paste the two tables into the exp 15 and exp 16 `RESULTS.md` under "Confidence intervals (added 26 Oct, exp 17)" and commit**

```bash
git add experiments/17_harden/ci_prior.py experiments/17_harden/test_ci_prior.py experiments/17_harden/results/ci15_*.json experiments/17_harden/results/ci16_*.json experiments/17_harden/results/ci_prior.md experiments/15_ar_steering/RESULTS.md experiments/16_difference_term/RESULTS.md
git commit -m "docs: cluster-bootstrap CIs for the exp 15 r11 shifts and the exp 16 follow-AUROCs"
```

---

### Task 5: Vocabulary readout of the exp 15 directions

**Files:**
- Modify: `experiments/17_harden/lib17.py` (add `vocab_stats`, `pick_unembedding`)
- Modify: `experiments/17_harden/modal17.py` (add `y4lm`, CLI key `vocab`)
- Modify: `experiments/17_harden/report17.py` (add `vocab_markdown`)
- Test: `experiments/17_harden/test_lib17.py`

**Interfaces:**
- `pick_unembedding(weight_map: dict[str, str]) -> tuple[str, str, str | None]`: returns `(key, shard, norm_shard)` with key `"lm_head.weight"` if present else `"model.embed_tokens.weight"`, and the shard holding `"model.norm.weight"` (None if absent).
- `vocab_stats(logits: np.ndarray, top_k: int = 30) -> dict`: `{"peak_ratio": max|logit| / sd, "top_mass": sum of top-50 |logit| / sum |logit|, "top_pos": [(idx, value)] top_k, "top_neg": [...]}`.
- `y4lm(row: str)` (CPU, `memory=24576`, both volumes): reads the Qwen snapshot's `model.safetensors.index.json`, loads the unembedding (bf16 → fp32) and `model.norm.weight` with `safetensors.safe_open`, applies gain `g = 1 + w_norm` (D10's convention; recorded in the output as `"gain": "1+w"`), loads `/results/15_ar_steering/spec15_{row}.pt`, and for every direction in `dirs` plus the 20 `iso_*` nulls computes `logits = W_U @ (g * v)` and `vocab_stats`, decoding the top tokens with the tokenizer. Adds per-row token-of-interest ranks: table column `["\n", " |", "|", " first", " fifth", " column"]`, parenthesis `[")", " )", " inside", " outside", " parenthesis"]`. Saves `/results/17_harden/vocab17_{row}.json` and the small print `results/vocab17_{row}.json`.
- `vocab_markdown(results) -> str`: per row, a table with direction, peak ratio, z against the 20 iso nulls (`lib15.band`), top-10 positive and top-10 negative tokens; and the spec's lexical verdict for `arstmt_raw`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pick_unembedding_prefers_lm_head_and_falls_back():
    assert L.pick_unembedding({"lm_head.weight": "a", "model.norm.weight": "b"}) == ("lm_head.weight", "a", "b")
    assert L.pick_unembedding({"model.embed_tokens.weight": "a"}) == ("model.embed_tokens.weight", "a", None)

def test_vocab_stats_peak_ratio_is_large_for_one_hot():
    x = np.zeros(1000); x[7] = 10.0
    s = L.vocab_stats(x, top_k=3)
    assert s["peak_ratio"] > 20 and s["top_pos"][0][0] == 7
```

- [ ] **Step 2: Run to verify they fail** (`-k "unembedding or vocab_stats"`, expect AttributeError)

- [ ] **Step 3: Implement the two functions, the stage, and the markdown**

- [ ] **Step 4: Run tests, then the stage**

Run: `.venv/bin/python -m pytest experiments/17_harden -q` then `cd experiments/17_harden && modal run modal17.py --stage vocab --features tb_tablecol,tb_paren` then `.venv/bin/python report17.py --vocab results/vocab17_*.json --out results/vocab17.md`.
Expected: `results/vocab17.md` with the two tables; for each row the `arstmt_raw` line shows its z and verdict.

- [ ] **Step 5: Append a "Vocabulary readout" section to `RESULTS.md` and commit**

```bash
git add experiments/17_harden/lib17.py experiments/17_harden/modal17.py experiments/17_harden/report17.py experiments/17_harden/test_lib17.py experiments/17_harden/results/vocab17_*.json experiments/17_harden/results/vocab17.md experiments/17_harden/RESULTS.md
git commit -m "feat: exp 17 vocabulary readout of the exp 15 directions through the unembedding (is the statement direction lexical)"
```

---

### Task 6: Cross-scoring with Guiv's bank (runs only when his files exist)

**Files:**
- Modify: `experiments/17_harden/lib17.py` (add `load_guiv_items`, `guiv_pairs`)
- Modify: `experiments/17_harden/modal17.py` (add `n6wd` H100 and `k9zc` H200; CLI `xs_ol`, `xs_nla`; `g1ps` reports which input files are present)
- Modify: `experiments/17_harden/analyze17.py` (add `xs_table`), `report17.py` (add `xs_markdown`)
- Create: `experiments/17_harden/inputs/README.md` (the two input contracts, committed; the data files are ignored)
- Test: `experiments/17_harden/test_lib17.py`

**Interfaces:**
- Input contract A `inputs/guiv_items_{row}.json`: list of `{"id": str, "text": str, "value": "first"|"late"|"inside"|"outside", "cluster": str}`; the read position is the last token of `text`.
- Input contract B `inputs/guiv_phrasings.json`: `{row: {"nla": [{"id", "template", "fillers": {value: slot}}], "ol": [...]}}` in the exp 13 template shape so `feature_spec.fill` applies.
- `load_guiv_items(path: Path, row: str) -> list[dict]`: validates every record has the four keys and a legal value; raises `ValueError` naming the first bad record.
- `guiv_pairs(phrasings: dict, row: str, reader: str) -> list[list[str]]`: `[[id, A, B], ...]` as exp 16's `_pairs` returns.
- `n6wd(row)`: if `inputs/guiv_items_{row}.json` is missing, returns `{"feature": row, "skipped": True, "reason": "no guiv items"}`. Otherwise tokenises (`add_special_tokens=False`), captures L42 at the last token under `ol.base()` as exp 13 `s1` does, and scores four cells for OL and J-lens: his items × our frozen pairs, his items × his pairs, our 160 selection items × his pairs (ours × ours already exists in exp 13). Saves `/results/17_harden/xs_ol_{row}.pt` with `{"his_ids", "his_y", "his_acts42", "lo": {cell: {pair_id: tensor[N]}}}`. `k9zc` is the NLA mirror (`xs_nla_{row}.pt`), reading `his_acts42` from the OL file so the capture happens once (so `xs_ol` must run first; `k9zc` returns skipped with reason if the file is absent).
- `xs_table(row, vol_dir) -> dict`: 2×2 AUROC with cluster-bootstrap CIs per reader; `xs_markdown` renders it.

- [ ] **Step 1: Write the failing tests** (`load_guiv_items` rejects a record with value `"middle"`; `guiv_pairs` returns `[id, A, B]` triples)

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement the lib functions, the two stages, `xs_table`, `xs_markdown`, and `inputs/README.md`**

- [ ] **Step 4: Run the tests and the skip path**

Run: `.venv/bin/python -m pytest experiments/17_harden -q` then `cd experiments/17_harden && modal run modal17.py --stage xs_ol --features tb_tablecol`
Expected: tests pass; the stage prints `skipped: true, reason: no guiv items` and exits 0.

- [ ] **Step 5: Commit** (`feat: exp 17 cross-scoring stages for Guiv's bank (items x phrasings), skip when inputs are absent`)

- [ ] **Step 6 (when the files arrive): drop them in `inputs/`, run `xs_ol` then `xs_nla`, run `analyze17.py --xs`, append the 2×2 table to `RESULTS.md`, commit the JSON and the doc**

---

### Task 7: Pre-registration for the prediction test

**Files:**
- Create: `notes/prereg_prediction_test.md`
- Modify: `notes/decisions.md` (append D40), `README.md` (experiment list gains 17; D-count)

**Interfaces:** none in code. The document must contain, in this order:

1. **Claim under test.** One sentence: a concept's readable / sayable-by-span / sayable-by-statement / used profile, measured without running a reader, predicts each reader's prompted score and spontaneous behaviour.
2. **Instruments, each with the exact existing function and the exact threshold.** Readable: `mats.probes.cv_auroc` on the 160 selection items, grouped 5-fold, "readable" if AUROC ≥ 0.80. Sayable by span: exp 14's gap-kept statistic for the NLA reconstructor on the true 32-token span, "sayable" if ≥ 0.50 of the class gap. Sayable by statement: the fixed-label statement pair through the NLA reconstructor, "statement-sayable" if ≥ 0.30. Used: exp 15's centred class-mean push at r11 with 20 isotropic nulls on the row's declared next-token metric, "used" if z ≥ 3. J-enrichment of the class-mean direction at rank 1536 (`mats.jspace.enrichment`), "J-enriched" if ≥ 1.5.
3. **Prediction rules** (written as a table, one row per reader × output):
   - NLA prompted ≥ 0.85 iff readable.
   - NLA states spontaneously (≥ 20% of readouts) iff used and the concept constrains the next token (declared per concept in the list, before scoring).
   - Oracle Lens prompted ≥ 0.70 iff sayable by span; Oracle Lens exhibits in free text iff sayable by span and the value is expressible in ≤ 2 tokens; Oracle Lens states: never (predicted 0%).
   - J-lens prompted ≥ 0.70 iff J-enriched or statement-sayable.
4. **Concept list** of 12 to 15 rows, each with: name, two values, item generator (exp 13's `items_build` style or Guiv's bank), the declared "constrains the next token" flag, and a blank prediction cell per reader to be filled from the instruments before readers run. Include at least two concepts chosen to be sayable-by-statement but not used (so the account can fail) and two in-workspace controls.
5. **Scoring rule.** A prediction counts as correct if the observed cell lands on the declared side of the threshold; report accuracy over cells with a binomial CI and the confusion table; declare the account "predictive" if accuracy ≥ 0.80 over ≥ 36 cells.
6. **Procedure and hash.** Instruments run first (one GPU job), predictions filled in, the file's sha256 recorded in `decisions.md`, then readers run. Layers: Oracle Lens also at 36, 44, 52 for the sayable-vs-readable test across layers.
7. **Budget and timing.** Two GPU-days at most; cost ceiling $40.

- [ ] **Step 1: Write the document with sections 1 to 7 complete; the concept list may carry "to confirm with Guiv" on the rows taken from his bank, but every row has a generator named**

- [ ] **Step 2: Self-check** against the spec: every instrument names an existing function (grep it), every threshold is a number, every prediction rule is falsifiable by a cell in the table.

- [ ] **Step 3: Append D40** to `notes/decisions.md`: "exp 17 built (multi-candidate verdict: <fill from Task 3>; CIs; vocabulary verdict: <fill from Task 5>); prediction-test prereg written, hash pending until the instruments run". Update `README.md`.

- [ ] **Step 4: Commit**

```bash
git add notes/prereg_prediction_test.md notes/decisions.md README.md
git commit -m "docs: pre-registration for the week-two prediction test (instruments, rules, concept list); decisions D40; README lists exp 17"
```

---

## Self-review notes

- Spec coverage: goal 1 → Tasks 1–3; goal 2 → Task 4; goal 3 → Task 5; goal 4 → Task 6; goal 5 → Task 7. Declared criteria → Task 3's verdict line and Task 5's verdict line. Constraints → Global Constraints.
- Names used consistently: `metrics_multi`, `lse_pair_lo`, `mc_table`, `ci15`, `ci16`, `vocab_stats`, `pick_unembedding`, `load_guiv_items`, `guiv_pairs`, `xs_table`; stage functions `q2vn`, `f8rt`, `y4lm`, `n6wd`, `k9zc`, `g1ps`.
- Review Focus items 1–5 each have a test in Tasks 1, 1, 1 (template axis), 4, 5.
- Order of execution: Task 1 → 2 → 3 can run today; Task 4 and Task 5 need nothing from Tasks 2–3 and can run in parallel with them; Task 6's skip path can be built any time, its data step waits on Guiv; Task 7 waits only for the two verdicts.
