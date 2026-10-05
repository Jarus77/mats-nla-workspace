# Producing readouts

Judging is the contract; producing is optional. The item banks, the judges, the baselines and
the results contract are the benchmark; to score a lens you run it on the cells the plan names,
anywhere, and hand this repo one readouts file per family. `wsbench produce` (below) is a
bundled producer for anyone who would rather not write that plumbing. This page is the
interface between the two halves.

## 1. Get the read plan

```
uv run wsbench plan out=outputs/plan               # every family
uv run wsbench plan families=poetry,buggy_code     # a subset
```

One JSONL per family, one row per bank item. A row says exactly what to feed the model, how to
render it, which token positions to read, and at which layers:

```json
{"family": "poetry", "id": "couplet-ahead-head", "render": "plain",
 "positions": {"kind": "line_one_newline"}, "layers": [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
 "text": "A rhyming couplet:\nThe captain pointed at the route ahead,\nAnd told his crew to follow where he'd ",
 "note": "the newline that ends line one of the couplet"}
```

| key | meaning |
|---|---|
| `render` | how to turn the row into a token sequence; the vocabulary is `wsbench.readplan.RENDERS` (`plain`, `chat`, `chat_context`, `chat_prefill`, `chat_dm`, `chat_summarize`, `bare`, `captured`) |
| `text` / `messages` / `system` / `prefill` / `suffix` / `assistant` | the pieces the render composes; `captured` rows ship `extra.input_ids` instead |
| `positions` | a rule over the rendered tokens (below); `wsbench.readplan.resolve(rule, tokens)` turns it into indices once you have tokenized the render |
| `layers` | the decoder layers to read (residual stream after that layer) |
| `note` | the family's read site in one line; the family README has the rest |

Positions rules, all handled by `resolve`:

| kind | selects |
|---|---|
| `final_token` | the last token of the render |
| `offset_from_end {k}` | the token `k` from the end (`k=1` is the last) |
| `last_n {n}` | the last `n` tokens |
| `all` | every token |
| `all_from_end` | every token, reported as negative offsets `-n..-1` (arithmetic intermediates: the bank and judge count from the end) |
| `positions {positions}` | explicit indices (brew, hallucination, jailbreak, J-lens precision) |
| `line_one_newline` | the newline token that ends line one (poetry) |
| `from_token {token}` | from the first token with that text through the end (chain's question span) |
| `from_last_sentence_start` | from the first token of the last user turn's final sentence through the end (user modelling) |
| `suffix_text {text}` | the tokens spelling `text` at the end of the render (directed modulation's carrier) |
| `suffix_text_then_tail {text}` | those plus every token after them (the association families' summarize suffix and chat tail) |

Renders are described for Qwen3.6-27B's chat template; another model's template changes every
position index, which is why the rules are given as rules and not as numbers wherever the bank
allows it. `resolve` works on the token strings of any tokenizer.

## 2. Read the lens

For every row, every position the rule selects and every layer: run your lens on the residual
activation there and keep what it produces. A prose lens gives one or more text samples; a token
lens gives a ranked token list with scores.

## 3. Write the readouts file

One JSONL per family, one row per (item, layer, position), in the contract of the top-level
README:

```json
{"id": "couplet-ahead-head", "layer": 36, "pos": 16, "token": "\n", "samples": ["led the way"]}
{"id": "couplet-ahead-head", "layer": 36, "pos": 16, "token": "\n", "tokens": [" led", " said"], "scores": [9.1, 7.7]}
```

`id` is the row's `id` from the plan, `pos` the index into your render, `token` that position's
token string (optional, checked where a family knows it). A file is all-prose or all-tokens.
Missing cells are fatal by default: every (item, layer) the plan asks for must have a row, or the
run must be told `allow_missing=True` and the result is marked incomplete.

`wsbench convert-gen-dir` and `wsbench convert-read-json` produce this shape from the two
in-house producer layouts; anything else needs a few lines of your own.

## 4. Judge

```
uv run wsbench judge family=poetry readouts=outputs/readouts/mylens/poetry.jsonl out=outputs/judged/mylens/poetry dry_run=True
uv run wsbench run all=True readouts_root=outputs/readouts/mylens out=outputs/judged/mylens
uv run wsbench report dir=outputs/judged/mylens
```

## Or let the repo produce them

`wsbench.produce` is a small producer over Hugging Face transformers, for anyone who has a GPU
and wants readouts without writing the plumbing. Install the `gpu` extra (`uv sync --extra gpu`)
and:

```python
from wsbench.produce import Producer

# methods: logit_lens | jlens | rlens | olens | nla
p = Producer.load("Qwen/Qwen3.6-27B", "jlens")
p.read("The athlete Muhammad Ali plays the sport of", pos=-1, layer=36).readout.tokens
p.read_prompt("Sechs geteilt durch zwei ist", positions="all", layers=[20, 36, 60])
# a whole eval set, resumable; the plan picks the cells
p.run_family("poetry", "outputs/readouts/mine/poetry.jsonl", limit=10)
# swap method, keep the loaded model
p.use("olens")
```

The same from the shell:

```
wsbench produce family=poetry method=logit_lens out=outputs/readouts/mine/poetry.jsonl
wsbench produce text="The athlete Muhammad Ali plays the sport of" method=jlens positions=-1 layers=20,36,60
```

What it does: renders the prompt the way the plan says (its chat template, thinking off),
captures the residual stream after each requested decoder block in one forward pass, and hands
each (layer, position) vector to the method. The vector lenses (`logit_lens`, `jlens`, `rlens`)
return the top-10 tokens with scores; `olens` verbalizes through its LoRA with the vector placed
in the carrier prompt's `<activation>` slot as `alpha · unit(h)`, the adapter off while capturing
and on while generating; `nla` runs Karvonen's reader with the vector added norm-matched at its
marker token. Methods are plain classes behind one `read(h, layer)` protocol, so a new lens is a
few lines. Sampling for the verbalizers is `Sampling(temperature=1.0, top_p=0.95, top_k=64,
max_new_tokens=256, k=1)`, the settings the in-house arms used.

Two spellings, on purpose: the read-site `token` is `tokenizer.decode([id])`, exactly what the
banks store (`"\n"`, `" öffnen"`, `"<|im_end|>"`); the ranked `tokens` of a token lens are the
byte-level BPE strings with `Ġ`/`▁` shown as a space (`" led"`), what the regex scorers expect.
`pos` is the index the plan resolves: absolute, except the two rules that count from the end
because their banks and judges do — `offset_from_end` (some multihop items) and `all_from_end`
(arithmetic intermediates) — whose rows carry the negative offset (`-8`). A single-layer lens reads at its trained layer whatever the
plan lists: `nla` is layer 42 (not on the benchmark grid; the in-house L44 NLA numbers were
off-layer and are superseded). `nla` also loads its own copy of the reader (a second 27B on the
same device), so it wants an H200; the other methods fit one H100.

What it is not: fast. It reads one prompt at a time on one GPU, which is right for a token
position, a question or a family, and wrong for the whole benchmark at every token; the in-house
runs fan that out over Modal.

## What the plan does not carry

- **Model-dependence.** The banks were gated on Qwen3.6-27B; `wsbench capable` re-runs that gate on
  another model, and `AGENTS.md` says which families port, which need re-gating and which need
  re-generating.
- **Sampling.** The in-house arms used k=1, temperature 1.0, top-p 0.95, top-k 64, max 256 new
  tokens for prose lenses and top-10 for token lenses; the judges accept any k.
- **Two families without text.** `hallucination` and `jlens_concept_pr` ship captured token ids
  and positions, not prompts: their banks are the model's own outputs, and they are read by
  re-feeding those ids.
