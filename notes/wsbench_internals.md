# WorkspaceBench (wsbench) internals: notes for jlens vs olens vs nla on Qwen3.6-27B @ L42

Source: `github.com/camilablank/workspace-bench` @ `92d763e` (cloned to scratchpad). Read: README,
AGENTS, CLAUDE, docs/*, plans/*, `src/wsbench/produce/*`, judges/report/baselines code, all
`evals/*/README.md`. Also checked the HF repos directly (configs, file listings) and the LessWrong
write-up text.

Verified locally (no GPU): `uv sync --extra dev` OK (needs network; `--offline` fails, nothing
cached), `uv run pytest -q` -> **534 passed**, `wsbench list` and `wsbench plan` OK. Cell counts
below come from rendering every plan row with the real `Qwen/Qwen3.6-27B` tokenizer
(`produce.render` + `readplan.resolve`). The arithmetic count matches the README exactly (22,022
cells at L42), which checks the method.

---

## 0. TL;DR / decisions to make before spending

1. **olens at L42 is off-contract.** The olens model card says its trained band is 20-60 step 4,
   and "layers outside 20-60 step 4 are off-contract". So comparing all three methods at L42
   handicaps olens. Recommendation: run olens and jlens at **40, 42 and 44**. jlens is cheap
   anyway, and the olens cost is only GPU time. Report L42 as the like-for-like cell and 40/44 as
   olens's on-contract neighbours.
2. **The default olens LoRA repo is gated, not open.** `agu18dec/local-workspace` (HF *dataset*)
   has `private: false, gated: "auto"`: file downloads return "Access ... is restricted" until
   you log in and click accept. Accept is automatic, but you still need `HF_TOKEN` on Modal. The
   copy at **`agu18dec/olens_and_ar/olens_s3d_rl600`** (model repo) is open and byte-identical
   (see §1.4). `OLens.bind` hard-codes `repo_type="dataset"` for `repo:sub` specs, so using the
   model repo needs a 6-line subclass (§1.4).
3. **The producer CLI cannot pass method kwargs.** k, lora, adapter and alpha are only settable
   from Python (`Producer.load(model, "olens", sampling=Sampling(k=3))`). Hallucination wants
   k=3 for prose lenses.
4. **NLA needs two 27B models in memory (~110 GB, so H200).** Alternative on one H100: capture
   all L42 activations with the base model to disk, unload it, then load the NLA reader alone.
   Capture-once is also how to guarantee that all three methods read identical vectors.
5. **Possible NLA layer off-by-one.** The bench defines layer L as the output of
   `model.layers[L]`. NLA's `nla_meta.yaml` says `layer_index: 42`. If EasyNLA used
   `hidden_states[42]`, that is the output of block 41. Check against the NLA repo's
   `data/example_activations.parquet` (64 L42 activations with their source text) before trusting
   NLA numbers.
6. **No-key replication path:** the six multi-token families (`*_mt`, `multilingual_multihop`,
   `multilingual_typo`) are regex-scored with **zero API calls for prose arms** (olens, nla). For
   jlens (token bags) the regex on raw tokens scores about 0.00 by construction. The instrument
   of record first runs a Gemini summarizer over each bag: about 100 cheap calls per family at
   L42.

---

## 1. How each method in `produce` is implemented

Common path (`produce/backend.py`, `producer.py`, `render.py`):

- `Backend.load(model_id="Qwen/Qwen3.6-27B", dtype=bfloat16, device_map="cuda")` uses
  `AutoModelForCausalLM` plus `AutoTokenizer`. The HF config says
  `Qwen3_5ForConditionalGeneration` (a VLM), but transformers 5.x maps `qwen3_5` to
  **`Qwen3_5ForCausalLM`** ("VLM compatibility"), so the model is text-only and the blocks sit at
  `model.layers`. The backend also probes `model.language_model.layers` and
  `base_model.model.model.layers`.
- **Layer L = residual stream after decoder block `model.layers[L]`** (forward hook on the block
  output). The repo says every bank was captured this way. One forward pass with
  `logits_to_keep=1` captures all requested (layer, pos) cells as fp32 CPU tensors `[n_pos, d]`
  (d = 5120). If the model is a `PeftModel`, capture runs under `disable_adapter()`, so it always
  reads the base model.
- Renders: `plain`/`bare` = raw tokens, no specials. All chat renders use the tokenizer's own
  template with **`enable_thinking=False`**, which produces an empty `<think>\n\n</think>\n\n`
  block (`chat`, `chat_context`, `chat_prefill`, `chat_dm`, `chat_summarize`). `captured` = token
  ids shipped in the bank.
- The positions rule is resolved over display tokens. `pos` is reported as an absolute index,
  except `offset_from_end` and `all_from_end`, which use negative offsets.
- `run_family(family, out, layers=..., items=..., limit=...)` is resumable: it skips item ids
  already in `out` and writes one JSONL row per (item, layer, pos). It runs at **batch 1**, one
  prompt and one cell at a time. The README says "not fast; the in-house runs fan out over Modal".
- `Producer.read(text, pos, layer)` uses the method's own layer, else 44. `read_prompt` uses the
  method's layers, else GRID = 20..60 step 4. `read_spec` and `run_family` use the plan's layers
  unless the method has fixed layers or you pass `layers=`.

### 1.1 `logit_lens`
`W_U · final_norm(h)` (the model's own norm module, fp32), top-10 tokens plus raw logits. No
artifact.

### 1.2 `jlens` (J-lens, readout of record)
- Artifact: HF `neuronpedia/jacobian-lens`, file
  `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` (3.30 GB, so about
  63 source layers × 5120² in bf16). `torch.load` on CPU, then the key `J` (dict by layer) or
  `jacobians` (stack), checked to be contiguous from 0. Moved to the GPU as **fp32 (~6.6 GB)**.
- Readout (the "magnitude-free cosine"): `scores = (W_U · J_L · h) / ‖J_Lᵀ W_U[t]‖`, with the
  denominator computed once per layer and cached. Top-10 tokens with cosine-like scores. The
  layer must be < `J.shape[0]`, so 42 is fine.
- Display tokens: byte-level BPE with `Ġ`/`▁` shown as a space.

### 1.3 `rlens` (not needed, for completeness)
`camilablank/workspace-lenses : qwen3.6-27b/r-lens/lens.pt`. Readout `W_U · norm(J·h)`.

### 1.4 `olens` (Oracle Lens, s3d RL iter 600)
- Default `lora="agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600"`, downloaded
  with `snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{sub}/*"])`, then
  `PeftModel.from_pretrained(base, dir)`. **One model serves both roles**: the adapter is off
  during capture and on during generation.
- Adapter (`adapter_config.json`): LoRA r=16, alpha=32, dropout 0, peft 0.19.1, base class
  `Qwen3_5ForCausalLM`. Targets cover attention, MLP **and the DeltaNet projections**
  (`q,k,v,o_proj, gate,up,down_proj, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`).
  `adapter_model.safetensors` is 467 MB.
- `run.json`: `run_name s3d.rl.ddp600.s0`, `transform "unit"`, `alpha 16000.0`, trained layers
  `[20..60 step 4]`, prompt `concepts_raw`. Initialised from `ao.iolens.ptag.s3d.omp4bp.s2/step251`
  with GRPO for 600 steps, β=0.02, joint bullet-FVE reward on the ptag AR. Held-out joint FVE
  **0.487** (on that AR only).
- **Dataset vs model repo match, verified from the HF tree API:** the git blob oids of
  `adapter_config.json` (`b75427ab…`), `adapter_model.safetensors` LFS pointer (`5e156b48…`, same
  size 467,047,680), `run.json` (`7c6f5612…`) and the README (`0e88429b…`, renamed
  `README_original.md` in the model repo) are identical between
  `datasets/agu18dec/local-workspace/ckpts/ao/rl/s3d.ddp600.s0/iter_000600/` and
  `agu18dec/olens_and_ar/olens_s3d_rl600/`. Identical pointer oids imply the same sha256, so the
  files are byte-identical. Model-repo sha256 of the safetensors:
  `7d4b301e002246af3c3f2bb6e768f8fa62f8aac2c231a0bd02a8d4f9ecafd9aa`.
  - Side note: the model-repo README calls the dataset's `iter_000600/reference/` subdir "a
    byte-identical second copy", but its blob oids differ (`1d41144b…`, `f6bb014d…`). It is
    probably the GRPO reference (init) policy. It does not matter for the default path.
- Carrier prompt (one user turn, chat template, `enable_thinking=False`):
  `An activation vector from layer {layer} of a language model is enclosed in activation tags: <activation>{char}</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line.`
  `{char}` comes from scanning U+3200–U+33FF for a char that stays one token inside the full
  render. For Qwen3.6 this is **㈜ (id 158983)**, the same marker NLA uses. Note `{layer}` is in
  the prompt, so L42 shows the lens a layer number it was never trained on.
- Injection: the input embedding row at the slot is **replaced** by `alpha · h/‖h‖` with
  alpha=16000. Generation uses `generate(inputs_embeds=…)`.
- Sampling (the `Sampling` defaults, "the settings the in-house arms used"): T=1.0, top_p=0.95,
  top_k=64, max_new_tokens=256, **k=1** sample per cell. The write-up says olens gives "1 sample of
  4 bullets (~90 tokens)" per cell. The model card's own example uses T=0.8 and 128 tokens: a
  different experiment, so stay on the bench settings.
- Model card warnings: capture with the adapter disabled (the producer does), the model card
  merges the adapter for about 4× faster generation (the producer does not), and layers outside
  20–60 step 4 are off-contract.
- To use the open model repo instead of the gated dataset:
  ```python
  from wsbench.produce import Producer, OLens
  class OLensModelRepo(OLens):
      def bind(self, backend):
          from huggingface_hub import snapshot_download
          from peft import PeftModel
          d = snapshot_download("agu18dec/olens_and_ar", allow_patterns=["olens_s3d_rl600/*"])
          backend.model = PeftModel.from_pretrained(backend.model, f"{d}/olens_s3d_rl600")
          backend.model.eval(); self._b = backend
  p = Producer.load("Qwen/Qwen3.6-27B", OLensModelRepo())   # instances pass through method()
  ```

### 1.5 `nla` (Karvonen-style NLA, Celeste's GRPO run)
- `repo="ceselder/qwen3.6-27b-nla-rl"` (open, not gated). `av_base/` is a **full separate 27B**
  (`Qwen3_5ForCausalLM`, bf16, saved with transformers 5.5.4, 49.8+3.97 GB of shards). It already
  has the warm-start AV LoRA (r128, all-module) merged in. On top goes
  `adapter="av_rl_adapters/iter_000400"` (r64 rsLoRA; iters 100..800 exist; the model card says
  "my guess is 300"). `nla.layers = [42]`, so it ignores the plan's layers.
- Training per the model card: layer-42 activations (`⌊2·64/3⌋`), `norm: none`, FineFineWeb 100k
  docs × 10 positions. Held-out FVE 53% at RL step 0, **76.6% at step 400**, ~78% by step 800.
- Prompt from `av_base/nla_meta.yaml` (`prompt_templates.av` or `.actor`), rendered with av_base's
  own chat template and `enable_thinking=False`:
  `You are a meticulous AI researcher ... We will pass the vector enclosed in <concept> tags ... The explanation consists of 2-3 text snippets ... <concept>㈜</concept> Please provide an explanation.`
  Marker id 158983 must appear exactly once.
- Injection: a forward hook on the reader's **block 1 output** at the marker position, prefill
  only (it skips steps with seq_len ≤ marker idx). It is a **norm-matched ADD**:
  `h[m] ← h[m] + ‖h[m]‖ · v/‖v‖`. The layer is not in the prompt. Output is split at
  `</explanation>` with the `<explanation>` prefix removed.
- Sampling: the same `Sampling` (T=1, top_p .95, top_k 64, 256 new tokens, k=1). The write-up
  says NLA readouts are ~250 tokens, so a 256 cap can truncate a little.

### 1.6 Memory and hardware
| method | GPU resident | box |
|---|---|---|
| base model alone (bf16, text-only) | ~54 GB + activations | — |
| logit_lens | + W_U fp32 5.1 GB | H100-80 |
| jlens | + J fp32 6.6 GB + W_U fp32 5.1 GB + a 5.1 GB transient (`W_U @ J_L`, once per layer) ≈ 71 GB peak | H100-80 (tight); CPU RAM ≥ 16 GB for torch.load + .float() |
| olens | + LoRA 0.47 GB, batch-1 generation | H100-80 |
| nla | base ~54 GB + av_base ~54 GB + r64 adapter ≈ 110 GB | **H200** (docs say so). Or capture-to-disk then reader-only on H100 |

Disk: ~54 GB (base) + 54 GB (av_base) + 3.3 GB (J) + 0.5 GB, so use a Modal Volume as the HF
cache. GPU time is my estimate, not a measurement: batch-1 HF generation of 256 tokens on 27B is
about 5–15 s per cell. Batch the generations (pad several cells per `generate`) if you go beyond
a few thousand cells.

---

## 2. Families at a glance (cells and calls are for **one layer = 42**)

Calls = judge calls per arm (Gemini unless noted), estimated from each judge's call unit and the
L42 cell counts. "+tok" = extra summarizer calls for a token-bag arm (jlens). Plan layers:
GRID = 20..60 step 4, SIX = 20/28/36/44/52/60, FIVE = 20/36/44/52/60.

| family | n | judge | read positions (render) | plan layers | L42 OK? | cells @L42 | calls/arm @L42 (prose) | +tok |
|---|---|---|---|---|---|---|---|---|
| association | 100 | Gemini bank judge (named + verbatim quote) | final token (plain) | GRID | yes | 100 | 100 | 0 |
| basic_readout | 100 | Gemini bank | final token; 32 implicit items: chat_prefill "My one-word answer is:" | GRID | yes | 100 | 100 | 0 |
| multihop | 100 | Gemini bank | last word (33 items `offset_from_end k=2`) | GRID | yes | 100 | 100 | 0 |
| multilingual | 100 | Gemini bank | final token | GRID | yes | 100 | 100 | 0 |
| poetry | 100 | Gemini bank | newline ending line 1 | GRID | yes | 100 | 100 | 0 |
| typo | 100 | Gemini bank | final token | GRID | yes | 100 | 100 | 0 |
| directed_modulation | 100 | Gemini 6-way MC per row | carrier-sentence tokens (chat_dm) | GRID | yes | 808 | ~808 | 0 |
| basic_readout_mt | 100 | **regex** | final token | GRID | yes | 100 | **0** | 100 |
| multihop_mt | 100 | **regex** | final token | GRID | yes | 100 | **0** | 100 |
| multilingual_mt | 100 | **regex** | final token | GRID | yes | 100 | **0** | 100 |
| typo_mt | 100 | **regex** | final token | GRID | yes | 100 | **0** | 100 |
| multilingual_multihop | 98 | **regex** | final token | GRID | yes | 98 | **0** | 98 |
| multilingual_typo | 100 | **regex** | final token | GRID | yes | 100 | **0** | 100 |
| multi_concept_directed_modulation | 27 | Gemini multi-select | last 12 tokens (prefilled sentence) | 44/52/56/60 | yes | 324 | 324 | 324 |
| chain_intermediates | 120 | Gemini free recall | plan: " What"→end (3000 cells); judged: **last token only** by default | GRID | yes | 3000 (120 judged) | 120 | 0 |
| brew_intermediates | 100 | Gemini multi-select | 24 explicit positions; judged: 9 emission + stir | GRID | yes | 2400 (900 judged) | 900 | 0 |
| buggy_code | 49 | Gemini, 2 stages | EOF token (bare); record layer 60 (py) / 56 | per item | **needs `layers=42`** (subset, so complete=False) | 49 | 98 | 0 |
| arithmetic_intermediates | 596 | Gemini free recall | every position (`all_from_end`, chat) at 56/60; frozen cell −7/−8 | 56, 60 | **needs `layers=42`** (the README does this for NLA) | 22,022 (or 596 frozen) | 596 batched (big prompts) | 0 |
| user_modeling | 100 | Gemini 6-way MC per sample | last sentence of user turn → assistant onset (chat_context) | GRID | yes | 2,112 | ~2.1k × k | 2.1k |
| conjunctive_association | 100 | Gemini 11-way MC, one blob per item | 19 summarize-suffix + tail sites (chat_summarize) | SIX | yes | 1,900 | 100 | 100 |
| role_bound_association | 100 | Gemini 3 MCs per cell | same 19 sites | SIX | yes | 1,900 | ~1.9k | 1.9k |
| relational_multihop | 100 | Gemini two 11-way MCs | blank / final token (bare) | GRID | yes | 100 | 200 | 100 |
| moral_rationale | 200 | Gemini 6-way MC | last 5 tokens (chat) | GRID | yes | 1,000 | ~1.2k | 1k |
| hallucination | 149 | Gemini span judge + claim verify | 1,123 punctuation/newline sites in Qwen's own response (captured ids) | FIVE | yes (`layers=` is not a subset here) | 1,123 | 2,246 | 1.1k |
| jlens_concept_pr | 299 | Gemini 3 stages (A/B+foil/P) | 1 captured pos per label | GRID (+L042 ref shipped) | **not producible**: rollout ids not shipped; prose arms only | — | ~3.2k | n/a |
| jailbreak_recognition | 86 | **Claude Sonnet 5** | every token of last user turn (chat_context) | FIVE | yes | 28,799 | 28.8k (≈ $257 per L42 arm, README) | +28.8k |
| agentic_misalignment | 32 | **Claude Sonnet 5**, 3-stage narrative | every prompt token (chat) | GRID+63 | yes | 50,612 | ≈ 50.6k (Stage A, one call per position) | — |

How the judges pick layers: every judge except buggy_code and arithmetic reads its layer set from
the readouts file (`rep.layers`). An L42-only file is judged as is, and with no `layers=` key it
can even be `complete`. Passing `layers=` (including through `wsbench run`) marks the result a
subset, so `complete=False` and it drops out of the macro. Hallucination is the exception: there
`layers=` is not a subset. So run the main batch with `wsbench run families=... ` and no
`layers=`, then run buggy_code and arithmetic separately with `layers=42`.

### 2.1 Cheapest to replicate first (no API key)
- **The six MT families × {olens, nla}**: regex, 0 calls, `spend_usd 0`, fully offline once the
  readouts exist. Each L42 arm is about 100 cells per family. Command:
  `wsbench run families=basic_readout_mt,multihop_mt,multilingual_mt,typo_mt,multilingual_multihop,multilingual_typo readouts_root=outputs/readouts/<arm> out=outputs/judged/<arm>`
  (`run` skips preflight for `scorer="regex"` specs).
- jlens on those families needs `OPENROUTER_API_KEY` for the summarizer (≈ 600 short calls in
  total at L42, well under $1). Without it every token cell is undecided. Raw-bag regex is about
  0.00 by construction, because multi-token forms cannot match a single token.
- Also free: `wsbench baseline variant=uniform`, any `dry_run=True` judge, and `wsbench plan`.

### 2.2 Most informative for NLA vs jlens vs olens
- **MT six (free)**: these separate the methods cleanly in the stored numbers (§5). olens >
  nla >> jlens-summarized, with the olens/nla gap largest on multilingual_mt and basic_readout_mt.
- **Single-token basic six (Gemini, 100 calls each, ≈ $0.2/family/arm)**: jlens's home turf (they
  come from the J-lens paper). This is the check of whether the verbalizers lose what jlens gets.
- **hallucination (~2.2k calls/arm)**: the write-up's central NLA claim ("nearly every NLA
  readout contains a contradicted claim"). Use k=3 for the prose arms to match.
- **chain_intermediates (120 calls) and arithmetic_intermediates (596)**: computed, never-written
  latents that come with nulls (decoy / permutation). These are the closest analogue to
  "information in the residual stream that the model doesn't write".
- **relational_multihop (200) and role_bound (1.9k)**: test order/binding, where a bag of words
  should fail. Their prompt-only floors are saturated (0.82 / 1.00), so read them against each
  other or a null, not against the floor.
- Skip on budget grounds: jailbreak (Claude, ~$257 per L42 arm), agentic (Claude, ~51k calls),
  jlens_concept_pr (not producible without the source repo's private capture rows).

---

## 3. Floors and hallucination measurement

### 3.1 Lucky guessing (`evals/baselines/lucky_guessing.json`, FROZEN)
- Gemini 3.8 Flash is shown **only the option lists** (the escape option is dropped). The lists
  come from each family's own judge builders, so the text and gold positions are byte-identical
  to what the judge sees. Five draws per item at T=1.0, seed 20260908. Variants: `blind`,
  `described` (one sentence on the domain plus 3 other items with gold marked: an informed
  ceiling) and `uniform` (seeded uniform picks, no model; this is the analytic floor
  `mean ∏ 1/n`). The pass rule is the family's own. `mean`, `std` and `majority` are stored.
- Produced by `wsbench baseline` (to `outputs/baselines/lucky_guessing/<fam>/<variant>.json`),
  then `wsbench freeze` merges into the tracked JSON. It refuses `limit` pilots.
- Every entry is stamped `instrument = <family prompt_version>`. `wsbench report` draws a floor
  only while the stamp equals the family's current prompt version (`lucky_guessing.floors()`).
- Covered: conjunctive, role_bound, relational, moral, user_modeling, directed_modulation, mcdm,
  brew, plus the 6 MT families. **The MT entries are stamped `mc-2026-09-16` and are no longer
  drawn**: the regex scorer has no option lists.
- Blind / described / uniform values: conjunctive 0.112/0.094/0.096; role_bound
  0.064/0.288/0.008; relational 0.000/0.03/0.004; moral 0.195/0.262/0.184; user_modeling
  0.22/0.218/0.192; DM 0.162/0.273/0.19; MCDM 0.36/0.36/0.272; brew 0.166/0.212/0.18.
  MT (stale): multihop_mt 0.658/0.746/0.126 (the options leak), typo_mt 0.134/0.206/0.166.
- Caveat in the README: a family headline is a max over the grid while the floor is one guess per
  draw. They are comparable only if `majority ≈ mean`.

### 3.2 Prompt-only (`evals/baselines/prompt_only.json`, FROZEN)
- Stock Qwen3.6-27B (no adapter) gets the prompt prefix up to the read token with the template
  stored in `_source.prompt_template`: *"Below is a prompt that was given to a language model,
  cut off at a specific token. The token in question is the final token of the text below:
  {token!r}. <prompt>{prefix}</prompt> What would a language model be thinking about on this
  token? Please write me a summary for this token."* Sampling k=1, T=1.0, top_p .95, top_k 64,
  256 new tokens. One generation per prompt position is replicated into every layer file. Each
  family's own instrument judges it at ONE layer, and `wsbench freeze kind=prompt_only` records
  `{rate, ci95, counts, judge_model, instrument, layers_judged, complete:false, saturated?}`.
- **The generator is not in this repo** (it lives in the source repo as `prompt_only_summary`).
  Only the judging and freezing are here. Re-measuring at L42 would mean regenerating the
  summaries yourself.
- Layers judged: 20 (the 13 entries from 2026-09-16), 44 (the nine from 2026-09-18), 56 (buggy,
  arithmetic). None are at 42. That is fine for a text-only floor, since the rows are identical
  across layers.
- Values: association 0.29, basic_readout 0.72, multihop 0.90, multilingual 0.76, poetry 0.71,
  typo 1.00 (saturated), DM 0.21, basic_readout_mt 0.59, multihop_mt 0.69, multilingual_mt 0.42,
  multilingual_multihop 0.449, multilingual_typo 0.66, typo_mt 0.67, chain 0.025, arithmetic
  0.153 (stale stamp `arith-free-2026-09-16`, not drawn), buggy 0.04 (stale, metric `net_S2`,
  not drawn), brew 0.958 (saturated), conjunctive 0.07, moral 0.495, relational 0.82
  (saturated), role_bound 1.00 (saturated), user_modeling 0.07. Excluded by design: mcdm (1.0),
  hallucination (prompt-only invention rate 0.53), jlens_concept_pr (precision 0.19).
- The `report` draws prompt-only beside each family, marks `SATURATED` ones, and drops
  stale-stamp ones.

### 3.3 Hallucination (family `hallucination`, plus plan 0010)
- Bank: 149 real first user turns (74 LMSYS + 75 DailyDialog) answered on-policy by Qwen3.6-27B.
  Read at 1,123 punctuation/newline sites inside the response (up to 8 per item), from the
  captured `input_ids` in `capture_rows.json`. Plan layers are FIVE. Use `layers=[42]` for the L42
  arms.
- Stage 1 (span judge, `v5c-chat`, frozen JSON prompt): one call per cell with the first **k=3**
  non-empty readouts (`HAL_K`). For token arms the bag is first summarized, then judged as k=1.
  The judge copies verbatim spans and types each one `wrong` or `off_topic`. In code: a span must
  be verbatim in its own readout or it is dropped. Any verified wrong span means the readout is
  `hallucinated`, else generic, off_topic or consistent. A hallucinated readout is *revoked* if
  all its wrong spans also appear as verified wrong spans of another readout in the same cell.
  - **Headline: `hallucination_rate` = hallucinated / specific readouts (unrevoked), lower is
    better.** CI: an item-level bootstrap of the ratio of sums (2000 draws, seed 0). Always read
    it next to `assert_share`, because a lens that never commits scores 0. Excluded from the
    macro.
- Stage 2 (plan 0010, `v5c-chat-verify-v1`): a second call per span-judged cell. It is GIVEN the
  wrong spans as established-false and lists every OTHER claim as
  true/unverifiable/disputed/off_topic. In code: unverified quotes are dropped, claims
  overlapping a false span are skipped, `disputed` counts as unverifiable, and unknown statuses
  count as unverifiable. The results go only into `extras`: `verifiable_share`,
  `false_share_of_verifiable`, `unverifiable_claims_per_readout`, `n_claims_*`, plus `by_layer`
  and `by_site_kind`. The headline is unchanged. `opts=verify=0` skips stage 2 and halves the
  cost. The two-stage design exists because a single prompt agreed with v5c at only κ≈0.44.
- Complete = pinned judge, no items/limit subset, zero missing and unjudged cells, ≤5% empty.
- Cost note from plan 0010: one full 5-layer verify pass was "≈ $60 reported / ≈ $150 actual"
  for 5.6k calls. OpenRouter `usage.cost` under-reports. The jlens_concept_pr README says the same
  thing (~2.5×).

---

## 4. Adding a custom family ("out-of-workspace": linearly decodable, not verbalizable)

Checklist (from CLAUDE.md plus what the tests enforce):
1. `evals/<fam>/items.json`: `{header..., "items": [{"name": ..., ...}]}`. `load_bank` sets
   `id = label_of(name)` (characters outside `[A-Za-z0-9._-]` become `_`). Never edit frozen banks
   in place.
2. `src/wsbench/readplan.py::plan()`: add a branch. Families that read the final token of a plain
   prompt can simply join the tuple that calls `_final_token_families(family, "plain")`. That
   honours each item's `prompt`, `eval_render` and `readout.offsets`. Otherwise return
   `ReadSpec(family, id, render, positions_rule, layers, text=..., ...)` with any positions kind
   from §1 (`positions`, `offset_from_end`, `suffix_text`, ...). `families()` lists any
   `evals/*/items.json`, so a missing branch makes `wsbench plan` raise `KeyError`.
3. `src/wsbench/evals/<fam>/__init__.py` registers an `EvalSpec(name, title, group, bank, judge=
   JudgeConfig(prompt_version=...), metric="pass_rate", higher_is_better=True, run=judge.run,
   calls_per_arm=..., sources="", scorer="regex"|"")`. `group` must be one of
   basic/basic_mt/computational/safety/association/bag_of_words/precision/logic, because
   `test_readme` maps it to a README heading.
4. `prompts.py` must export a non-empty `PROMPTS` dict and a `PROMPT_VERSION`
   (`test_prompts_in_readme` assumes every family is LLM-judged). A regex-only family re-exports
   an optional LLM judge's prompts, the way the MT families re-export the MC prompts. Each prompt
   must appear verbatim in a fenced block of `evals/<fam>/README.md` along with the version
   string. Bump the version on any edit.
5. Top-level README: an entry `**<title>** — [link](evals/<fam>/README.md)` followed by exactly the
   three lines `- *What it is:*` / `- *Example:*` / `- *Judged by:*` under the right `###` group.
   Also add a Judges-table row.
6. `capable/questions.py`: add a builder, OR an entry in `NO_QUESTION` with the reason
   (`test_no_question_families_say_what_their_gate_is`). For out-of-workspace items the natural
   gate is *inverted*: probe acc ≥ τ AND the model fails to verbalize when asked. `capable` cannot
   express that, so use `NO_QUESTION`.
7. `examples/readouts/<fam>.jsonl` (a few toy rows) and an offline `tests/test_<fam>.py`
   (monkeypatch `run_calls` if LLM-judged).
8. Optional: lucky-guessing builder (MC families only) and a prompt-only freeze.

**Minimal skeleton, regex-scored (no API key), final-token read:**

`evals/oow_probe/items.json`
```json
{
 "family": "oow-probe",
 "gate": "INVERTED: linear probe on Qwen3.6-27B L42 >= 0.90 held-out; model names the attribute <= chance when asked (10 draws, T=0.7, enable_thinking=False)",
 "frozen": "2026-10-XX",
 "items": [
  {
   "name": "oow-wordlen-0001",
   "prompt": "The committee finally approved the",
   "eval_render": "plain",
   "readout": {"kind": "final_prompt_token", "offsets": [-1]},
   "target": "eight letters",
   "units": [{"role": "attribute", "required": true,
              "match": ["eight letters", "8 letters", "8-letter", "eight-letter"]}],
   "probe": {"layer": 42, "heldout_acc": 0.97, "label": 8},
   "verbalize_rate": 0.1
  }
 ]
}
```

`src/wsbench/readplan.py` (inside `plan()`): add `"oow_probe"` to the
`_final_token_families(family, "plain")` tuple.

`src/wsbench/evals/oow_probe/__init__.py`
```python
"""Out-of-workspace probe items: attributes linearly decodable at L42 that the model cannot say."""
from pathlib import Path
from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register
from . import judge
from .prompts import PROMPT_VERSION

SPEC = register(EvalSpec(
    name="oow_probe", title="Out-of-workspace probe", group="basic",
    bank=Path("evals/oow_probe/items.json"),
    judge=JudgeConfig(prompt_version=PROMPT_VERSION),
    metric="pass_rate", higher_is_better=True, run=judge.run,
    calls_per_arm="0 (regex)", sources="", scorer="regex",
))
```

`src/wsbench/evals/oow_probe/prompts.py`
```python
"""Optional LLM judge (the shared bank judge); the scorer of record is the regex in judge.py."""
from wsbench.basic.prompts import PROMPTS, SCHEMA, SYSTEM, render_user  # noqa: F401
PROMPT_VERSION = "oow-regex-2026-10-xx"   # must also appear in evals/oow_probe/README.md
```

`src/wsbench/evals/oow_probe/judge.py`
```python
"""Regex scorer: an item passes when any unit form appears (word-bounded, folded) in one sample
of one cell; any layer, any position."""
import re
from wsbench.banks import load_bank
from wsbench.family import pass_rate_result, require_cells, tri_state
from wsbench.llm import Spend
from wsbench.mc import fold
from wsbench.mcjudge import item_scope, with_readout_count
from wsbench.readouts import load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from .prompts import PROMPT_VERSION

NAME = "oow_probe"
BANK = REPO_ROOT / "evals" / NAME / "items.json"

def _hit(forms: list[str], text: str) -> bool:
    t = fold(text)
    return any(re.search(rf"(?<!\w){re.escape(fold(f))}(?!\w)", t) for f in forms)

def run(args: JudgeArgs):
    _h, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    forms = {it["id"]: [m for u in it["units"] for m in u["match"]] for it in scope}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    seen = {(c.id, c.layer) for c in cells}
    missing = [(i, L) for i in ids for L in layers if (i, L) not in seen]
    require_cells(NAME, missing, len(ids) * len(layers), args)
    empty, hits = 0, {i: [] for i in ids}
    for c in cells:
        samples = list(c.samples) if c.samples is not None else list(c.tokens or ())
        if not any(s.strip() for s in samples):
            empty += 1
            continue
        if any(_hit(forms[c.id], s) for s in samples):   # never join samples or tokens
            hits[c.id].append(c.layer)
    rows = [{"id": i, "hitting_layers": sorted(set(hits[i])),
             "pass": tri_state(bool(hits[i]), any(k[0] == i for k in missing))} for i in ids]
    result = pass_rate_result(
        name=NAME, args=args, prompt_version=PROMPT_VERSION, rows=rows, cells=cells, rep=rep,
        n_expected=len(ids) * len(layers), n_missing=len(missing), n_unjudged=0,
        n_empty=empty, spend=Spend(), extras={"scorer": "regex"},
        chance_label="no analytic floor; compare to the probe accuracy and verbalize_rate",
    )
    return with_readout_count(result, scope, cells)
```
Design notes for this family. Token bags: single-token forms (`8`) give jlens a free win, so
either require multi-word forms the way the MT contract does (`probe_token_lens`) or summarize
first (`wsbench.summarizer.summarize`). Put the ceiling (probe accuracy) and the floor (the
model's own `verbalize_rate`, prompt-only) in `extras` or the README. Add an item-permuted null
(score each item's readouts against another item's forms), since the families that have nulls are
the trustworthy ones. Then `wsbench produce family=oow_probe method=nla out=...` works
unchanged.

---

## 5. Stored numbers to use as replication targets

There are no golden result files per method. `tests/golden/*` are option lists, prompts, regex
verdicts and capable questions. The `test_report` fixtures are synthetic (0.5 etc.). The
LessWrong results are in figures only, and its text has no numbers. What the repo does store:

**Multi-token families, regex instrument of record (2026-09-23)**: from each family README's
"Instrument history". olens/nla are prose regex; token lenses are regex-summarized. Arms: olens
`s3d-rl600` and `s3d-sft251` are any-of-11-layers (20..60). **`nla-rl-L42` is L42 only.** Token
lenses are 11 layers.

| family | s3d-rl600 | s3d-sft251 | nla-rl-L42 | jlens | rlens | logit | tuned | template-v3 | sae-l42 |
|---|---|---|---|---|---|---|---|---|---|
| basic_readout_mt | 0.71 | 0.57 | 0.53 | 0.24 | 0.27 | 0.20 | 0.19 | 0.05 | 0.07 |
| multihop_mt | 0.32 | 0.22 | 0.22 | 0.08 | 0.12 | 0.08 | 0.04 | 0.05 | 0.01 |
| multilingual_mt | 0.62 | 0.40 | 0.44 | 0.06 | 0.06 | 0.08 | 0.08 | 0.02 | 0.05 |
| typo_mt | 0.83 | 0.84 | 0.75 | 0.17 | 0.25 | 0.20 | 0.20 | 0.30 (prose regex 0.06) | 0.16 |
| multilingual_multihop | 0.40 | 0.28 | 0.34 | 0.05 | 0.10 | 0.06 | 0.13 | 0.08 | 0.09 |
| multilingual_typo | 0.78 | 0.63 | 0.60 | 0.05 | 0.00 | 0.02 | 0.03 | 0.01 | 0.06 |

- The same readouts under the old MC judge (`opts=judge=mc`): basic_readout_mt rl600 0.76 /
  sft 0.63 / nla 0.52 / jlens 0.14 / logit 0.18. multihop_mt 0.36/0.26/0.22/jlens 0.09.
  multilingual_mt 0.93/0.91/0.53. typo_mt 0.86/0.89/0.74 (template 0.39). multilingual_multihop
  0.26/0.19/0.14. multilingual_typo 1.00/0.97/0.68. Token lenses, raw-bag regex: 0.00 everywhere
  except template/sae.
- **For an L42-only replication, expect nla to land at its numbers above. olens and jlens will be
  ≤ their 11-layer numbers**, since those are a max over layers.

Other stored numbers:
- role_bound_association (summarize render, 19 sites × 6 layers, s3d-rl600): **0.63**
  (previously 0.40 on the bare render). conjunctive_association **0.14** (0.15 before), with ~83%
  "cannot tell". Per-site single-site pass rates are in `docs/read_sites.md` (best sites are `ize`,
  `.`, and the newlines after specials).
- jlens_concept_pr: Gemini precision at L44 for s3d **0.43** (hand labels 0.33), so Gemini
  over-credits.
- brew (source-repo full grid): s3d names the gold in 38% of emission cells vs 20% for an off
  colour (lift 0.18). Compute region 0.03, stir 0.02.
- jailbreak_recognition: s3d-rl600 pass_rate **0.8256** (plan 0007). That was the old 13-site
  Sonnet regime and is not comparable to the current all-token grid.
- Model-side targets: NLA held-out FVE 76.6% @ iter 400 (model card). olens joint FVE 0.487 on
  its AR.
- `capable` gate (AGENTS.md): poetry is 0.94 for Qwen on the bank. moral_rationale is 0.89.
- Floors are in §3.

---

## 6. Judge cost: all families × 3 methods at L42 (Gemini 3.8 Flash via OpenRouter)

Live OpenRouter price (fetched 2026-10-05): `google/gemini-3.8-flash` costs **$0.75/M input,
$3.75/M output** (reasoning billed as output). There is a `:batch` variant at half price, but the
repo does not use it. Judge settings: reasoning `{"effort":"minimal"}`, JSON schema output, 240
rpm default.

Per-call sizes are my estimates. Dry-run prompts are about 500–1,500 chars plus the readout;
real prose readouts are ≤256 tokens × k. That gives ~1.5k input / ~0.3k output, about $0.002
per typical call. Hallucination and arithmetic-batched calls are 5–10× bigger. Calibrated against
the repo's own report (verify stage ≈ $0.011/call reported, ~$0.027 actual).

| bucket (L42, one arm) | calls (prose arm) | calls (jlens arm) | est. reported $ | est. actual $ (×2.5) |
|---|---|---|---|---|
| bank six (association..typo) | 600 | 600 | ~1.5 | ~4 |
| MT six | 0 | ~600 summaries | 0 / ~0.3 | 0 / ~1 |
| directed_modulation + mcdm | ~1.1k | ~1.5k | ~3 | ~7 |
| chain (120) + brew (900) + buggy (98) | ~1.1k | ~1.1k | ~3 | ~7 |
| arithmetic (cells=all, 596 batched big calls; frozen is about 5× cheaper) | 596 | 596 | ~7 | ~18 |
| user_modeling (k=1) | ~2.1k | ~4.2k | ~5 | ~13 |
| conjunctive + role_bound + relational + moral | ~3.4k | ~6.5k | ~8 | ~20 |
| hallucination (span + verify; k=3 prose) | ~2.2k | ~3.4k | ~20 | ~50 |
| **subtotal per arm (Gemini families)** | **~11k** | **~18k** | **~$45–50** | **~$110–125** |
| **3 arms (jlens, olens, nla)** | | | **~$150** | **~$350** |

- Add olens at 40 and 44 and you roughly triple olens's line (another ~$100 reported). Dropping
  hallucination's verify stage (`opts=verify=0`) saves ~$10 reported per arm.
- Claude-pinned (excluded above): jailbreak ≈ **$257 per L42 arm** (repo-measured,
  $0.0089/cell × 28.8k). Agentic ≈ 51k Sonnet Stage-A calls per arm (several hundred $).
- How to measure instead of guess: `limit=3` smoke per family, then read
  `counts.spend_usd` in `results.json`, multiply out, and apply the ~2.5× under-reporting factor
  (or check the OpenRouter dashboard). The cache (`<out>/<fam>/cells.jsonl`) means re-runs only
  pay for missing cells.

---

## 7. Gotchas

- **Dependencies** (`pyproject.toml` / `uv.lock`): Python ≥ 3.12. Core: openai ≥1.40 (lock
  3.14.1), anthropic ≥1.0 (1.6.0), numpy ≥1.26 (2.5.3), pydra-config ≥0.0.17 (0.0.17.post1).
  `gpu` extra: **torch ≥2.4 (lock 2.14.0), transformers ≥5 (lock 5.17.0), peft ≥0.14 (lock
  0.21.0)**, accelerate (1.15.0), huggingface_hub (1.32.0), pyyaml (6.0.3). `qwen3_5` needs
  transformers 5.x. The olens adapter was saved with peft 0.19.1 and NLA av_base with transformers
  5.5.4, both fine. The base model's config says `transformers_version 4.57.1`; ignore that.
- **Hybrid architecture (Qwen3.6-27B = `qwen3_5`)**: 64 layers, a repeating
  [linear, linear, linear, full] pattern. That gives **48 Gated-DeltaNet linear-attention layers
  and 16 full-attention layers** (full at indices 3, 7, …, 63). **Layer 42 is a DeltaNet layer**;
  43 is full attention. d_model 5120, vocab 248,320, 24 query heads / 4 KV heads, head_dim 256,
  partial RoPE, `mamba_ssm_dtype float32`, plus an MTP head (`mtp_num_hidden_layers 1`, unused).
  transformers loads the DeltaNet kernels (`causal_conv1d`, fla `chunk_gated_delta_rule`) through
  `use_kernel_func_from_hub_with_fallback`. Without the `kernels` package or `fla` it falls back
  to torch: correct but slow, and kernel vs fallback numerics can differ slightly. The `gpu` extra
  includes neither. Pin one choice across all arms so activations are identical. Since all three
  methods share a capture, capture once.
- **`enable_thinking=False` everywhere.** Every chat render, the olens carrier and the NLA prompt
  use it, so every bank position index assumes the empty `<think>\n\n</think>\n\n` block. A
  render without it shifts the positions and opens a bare `<think>`.
- `Qwen/Qwen3.6-27B` is a VLM checkpoint. `AutoModelForCausalLM` gives a text-only
  `Qwen3_5ForCausalLM`, which is what the adapters target. Do not load
  `Qwen3_5ForConditionalGeneration`. Its blocks are at `model.language_model.layers`; the backend
  handles that, but NLA's hook search (`endswith("model.layers.1")`) is on the reader and is fine.
- **Layer convention / possible off-by-one** for NLA (see §0.5). olens's model card uses
  block-output recorders, which matches the bench.
- olens at 42 is off-contract, and `{layer}=42` is a value the carrier prompt never saw in
  training.
- olens default LoRA is **gated** (dataset, auto-approve). Use the open model-repo copy with a
  custom `bind`.
- `wsbench produce` CLI: no method kwargs. `OLens.layers=None`, so pass `layers=42` or it reads
  all 11 plan layers. NLA forces [42]. jlens without `layers=` reads the plan layers.
- Plans read more cells than some judges score: chain (3000 produced vs 120 judged), brew (2400
  vs 900), arithmetic (22k vs 596 frozen). For the verbalizers, filter positions before generating
  to save GPU (a custom loop over `readplan.plan` + `Backend.capture` + `method.read`).
- jlens_concept_pr cannot be produced: the rollout ids were stripped from the public repo. It is
  judge-only with prose arms, and the L042 reference files are shipped.
- buggy_code and arithmetic look for their record layers (56/60) unless you pass `layers=42`.
  That makes the result `complete=False`, which is expected.
- Missing cells are fatal (exit 2) unless `allow_missing=True`. A file must be all-prose or
  all-tokens. Duplicate (id, layer, pos) rows keep the first one.
- Override judges (`judge_model=` or `WSBENCH_JUDGE_MODEL`) are never "pinned", so never
  complete and never in the macro. The macro averages complete pass-rate families only:
  hallucination, jlens_concept_pr, agentic and buggy (score) are excluded.
- Floors were measured with Qwen3.6-27B and Gemini 3.8 Flash. Stale stamps (arithmetic, buggy,
  MT lucky-guess) are not drawn by `report`.
- Hallucination: k=3 readouts per cell for the in-house prose arms. With k=1 nothing can be
  revoked and `n_short_cells` goes up.
- OpenRouter `usage.cost` under-reports billed spend by about 2.5× (jlens README, plan 0010).
- jlens J file: CPU `torch.load` (3.3 GB bf16) then `.float()` (6.6 GB) before the GPU move, so
  give the Modal container ≥ 32 GB RAM.
- NLA: the in-house numbers use the producer default `iter_000400`. The model card suggests 300.
  The in-house L44 NLA numbers were off-layer and are superseded.
- `uv sync` needs network (nothing is cached offline). Tests are offline once synced.
