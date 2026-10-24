# 12 Spontaneous recall per concept type — single layer, three readers (PILOT / EXPLORATORY)

**Question (Neel, #research-log-o-lens, 11 Oct):** "for different kinds of structural concepts check
how often each method finds it; if NLA is near 100 and OLens near 0 that's super interesting. Only
do a single layer."

**Setup.** Qwen3.6-27B, layer 42, one activation per item at the read token. 14 concept types from
exp 07 (items generated with the true value known; 160 selection items for prompted scores, 40
mining items × 8 free samples for spontaneous scores). Readers: NLA iter_000400 @42, OLens
olens_s3d_rl600 @42, J-Lens n1000 @42. Two scores per cell:

- **prompted**: AUROC of the reader's log-odds between two frozen phrasings (exp 07 selection
  split; best of mean-of-3 mined templates and the fixed label).
- **spontaneous (new, this experiment)**: Gemini 3.8 Flash judges each free readout: does it
  explicitly *state* the property (quoting text that has it does not count), which value it states,
  and whether the readout's own text *exhibits* the value (an OLens continuation written in French
  exhibits "French"). 448 calls, $2.05, 0 failures. Spot-checked by hand on bullet index, quote
  state, noun number, POS (see below).

Date 2026-10-22. Code: `recall_lib.py` (+ `test_recall_lib.py`, 7 tests), `run_judge.py`, `analyze.py`,
`combine.py`. Outputs: `results/judged/*.json`, `results/recall_table.{md,json}`, `results/combined.{md,json,png}`.

## Table

See `results/combined.md` / `.png`. Groups: position in text (the "interesting" kind), upcoming-word
syntax, controls (known in-workspace features).

## What it says

1. **OLens never states a property.** 0 of 4,480 OLens@42 free readouts state any of the 14
   properties. OLens is a continuation reader: it *exhibits* language (56%), code language (93%),
   caps (50%) and tense (16%) in its own text, and exhibits nothing detectable for any position-in-text
   or syntax concept. So on Camila's "concepts extracted from the readout" measure, every
   "NLA high / OLens zero" cell is automatic for anything the NLA states at all; the fair
   comparison is NLA-states vs OLens-exhibits vs prompted AUROC.
2. **The NLA states structural concepts unprompted, but selectively.** Bullet index 25% of
   readouts (55% of items within 8 samples, 24/25 stated values correct; e.g. "'Mop the floor' is
   the second item in an emerging list"), quote state 14% ("an unclosed quotation"), next-word POS
   31%, upcoming noun number 28% ("grammatically requires a plural noun"), gender 9%. These match
   Camila's description of the NLA's unique content ("document format and template structure",
   "mid-unit syntactic incompleteness and required completion category").
3. **Three position-in-text concepts are never stated by anyone, even when the reader can discriminate
   them when asked.** Words since sentence start: NLA prompted 0.72 vs OLens 0.51, stated 0%. Line
   position: 0.60 / 0.55, stated 0%. Indentation: 0.87 / 0.89, stated 0% (2% of items). The
   prompted/spontaneous gap is the finding: the information is in the readers' reach but not in
   their vocabulary.
4. **Prompted, OLens ≥ NLA on 9 of 14; the NLA leads only on words-since-sentence-start (0.72 vs
   0.51) and quote state (0.83 vs 0.76).** J-Lens is at or near chance on every position-in-text and
   syntax concept (0.45–0.68) and high only on language/code (0.95–0.98) and noun number (0.84).
   "What OLens finds that J-Lens doesn't" = tense, indentation, bullet index, quote, caps, gender,
   Item-N: 0.76–0.99 vs 0.49–0.68.

## Caveats

- One model, one layer, one NLA checkpoint and one OLens checkpoint. 40 items × 8 samples per
  spontaneous cell; 160 items per prompted cell. No CIs on the spontaneous rates yet (binomial
  SE ≤ 3 points at n = 320).
- The judge's "exhibits" column is trivial for the NLA on caps (its prose is sentence case
  regardless) and is only meaningful for OLens.
- Prompted scores reuse exp 07's selection split; the templates were mined on the same 40 items
  whose free readouts are judged here (mining split), so the spontaneous and prompted columns are
  on different items.
- The NLA's "states it" count depends on the judge's reading of "explicit claim"; four cells were
  spot-checked (bullet index, quote, number, POS) and the stated values matched the ground truth
  in the examples read. A kimi-k3 second-judge pass on a stratified sample is the next check.
- OLens at 40/44 was not judged (it states nothing at 42; the prompted OL@40/44 numbers in exp 07
  match 42 on every feature).

## Spot-check examples (NLA, judged "stated")

- bullet index, true early: "'Mop the floor' is the second item in an emerging list of cleaning tasks"
- quote, true inside: "The final fragment ... is an unclosed quotation, requiring ..."
- noun number, true plural: "'j'attendais des' grammatically requires a noun phrase (plural noun)"
- POS, true adjective, judged noun: "requiring his biographical credentials or role (e.g., 'prominent LDS Church leader')" — judge disagreement; the NLA names a noun phrase with an adjective head.

## OLens by input layer (Neel's second ask: "what layer X OLens can find that layer Y OLens doesn't")

Same 14 concepts, OLens fed the layer-40, 42 or 44 activation (40 and 44 are on its training grid;
42 is not). Judged 2026-10-22, 448 more calls, $1.87. Table: `results/ol_by_layer.md`.

- **Stated:** 6 of 13,420 readouts across the three layers state any property (all six are "this is
  Python" at layer 40). OLens states nothing at any layer.
- **Exhibited:** within a few points across layers (language 56 / 56 / 59%, code 87 / 93 / 94%,
  caps 44 / 50 / 38%, tense 8 / 16 / 16%); nothing exhibited for any position-in-text or syntax concept.
- **Prompted:** every concept within 0.05 across 40 / 42 / 44 (largest spread: indentation 0.86 /
  0.89 / 0.91).

On these concepts there is nothing one of these three layers finds that another doesn't. The
off-grid layer 42 behaves like its trained neighbours, which also answers the "OLens can't really
read L42" worry from the mentor follow-ups a second way.
