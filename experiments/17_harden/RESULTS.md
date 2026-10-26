# 17 — Harden the headline cells (26 Oct 2026)

Before the joint update reaches Neel: (a) re-score table column and parenthesis with the full candidate set
(Guiv's critique of two-candidate log-odds), (b) confidence intervals on exps 15 and 16 (`ci_prior.py`;
tables appended to those experiments' RESULTS), (c) a vocabulary readout of the exp 15 directions (is the
statement-built direction lexical?), (d) cross-scoring with Guiv's bank when his files arrive. Declared
criteria: `docs/superpowers/specs/2026-10-26-harden-headline-design.md`. App `b7kx`; stages in `modal17.py`.

## (a) Multi-candidate re-score: the two-class AUROC survives, the readouts are one-sided

Guiv's critique: a two-candidate log-odds oversells a reader (his 85-94% two-way fell to 12-21% with eight
candidates). Here the value-slot phrasings of each reader are scored over the full value set on the exp 13
selection items (table column: columns one to six, true column from the generator; parenthesis: inside a
parenthesis / outside / inside a quotation / inside square brackets, 11 items with an open quote or bracket
excluded). Metrics per phrasing: top-1 among C (chance 1/C), mean reciprocal rank, the two-class AUROC
re-normalised over all candidates (log-sum-exp of each class's candidates), and the argmax counts per class.
95% cluster-bootstrap CIs. Binary mined families (e.g. "Final token begins a new row") have no value slot and
are not re-scored; the two-candidate reference is the fixed-label pair from exp 13. Full table: `results/mc17.md`.

| row | reader (phrasing) | top-1 among C | class AUROC over all C | two-candidate ref | argmax by class |
|---|---|---|---|---:|---|
| table column (6) | NLA (fixed) | **0.47** [0.39, 0.54], chance 0.17 | **0.89** [0.84, 0.94] | 0.89 | first items: "first" 75/80; late items: "first" 39, "second" 39, "third" 2, columns 4-6: **0** |
| table column (6) | OLens (col_N) | 0.12 [0.07, 0.18] | 0.80 [0.73, 0.86] | 0.64 | first: mostly "2"/"3"; late: mostly "3"; Spearman +0.31 |
| table column (6) | J-lens (words) | 0.28 [0.21, 0.34] | 0.66 [0.57, 0.74] | 0.64 | |
| parenthesis (4) | NLA (state) | **0.41** [0.33, 0.49], chance 0.25 | **0.88** [0.83, 0.93] | 0.88 | inside items: "inside a parenthesis" 61/73; outside items: "inside a quotation" 49, "inside a parenthesis" 27, **"outside": 0** |
| parenthesis (4) | OLens (state) | 0.28 [0.21, 0.36] | 0.40 [0.31, 0.50] | 0.43 | spread over all four |
| parenthesis (4) | J-lens (words) | 0.23 [0.16, 0.30] | 0.45 [0.36, 0.55] | 0.47 | |

**Declared verdicts.** Table column: survives (NLA six-way top-1 lower bound 0.39 > 0.33, class AUROC lower
bound 0.84 > 0.80, both above OLens). Parenthesis: two-candidate artefact by the declared rule (NLA four-way
top-1 lower bound 0.33, below the 0.50 bar).

**What the confusion shows, which the verdicts do not.** On both rows the NLA's discrimination is unchanged by
the extra candidates (0.89 and 0.88, the same as the two-candidate numbers), but its *naming* is one-sided. It
says "first column" for 94% of first-cell items and never names a late column (columns four to six get zero
argmax hits; late items split between "first" and "second"). It says "inside a parenthesis" for 84% of inside
items and never says "outside" (outside items go to "inside a quotation"). The table-column verdict passes
because the first-cell items alone carry the top-1; the parenthesis verdict fails because "outside" is never
produced. This is the prompted-score version of the exp 13 spontaneous finding (the NLA states "a new row is
beginning" and "an open parenthetical", never a late column or "outside"): **the NLA reads the marked,
completion-relevant state and does not name its complement.** Guiv's critique therefore applies to value
naming, not to marked-state detection; our prompted cells should be quoted as "first vs not-first" and
"inside vs not-inside", and the 1.00 / 0.96 headline (mean of three binary mined phrasings) is a
marked-state discrimination, not column counting. OLens's numeric phrasing is weakly ordinal (Spearman
+0.31; late items shift from "2" to "3"), which is more ordinal information than its two-candidate 0.64
suggested, and still far from naming the column.

**Three qualifications a careful reader will ask for (added after review).**
1. *The one-sidedness is already in the two-candidate scores.* Within the multi-candidate run, the NLA's own
   fixed-label pair has "first" > "fifth" on 100% of late items and "inside" > "outside" on 100% of outside
   items (own-pair sign rate column in `results/mc17.md`); the 0.89 / 0.88 AUROCs come from the *magnitude*
   of the log-odds, not its sign. The confusion confirms the one-sidedness; it did not discover it.
2. *It is a raw-argmax statement under a strong phrase prior.* "outside any parenthesis" is about 8 nats less
   likely than "inside a parenthesis" averaged over all items, and "sixth column" about 7 nats less likely than
   "first column". With a label-free per-candidate centring (subtract each candidate's mean log-prob over
   items), outside items go to "outside" 34 of 76 and late items to columns four to six 45 of 80 ("sixth" 29),
   so the relative evidence to name the complement is present behind the prior (prior-centred column in
   `results/mc17.md`). "Does not name its complement" therefore means "under raw argmax, which is what a
   user reading the NLA gets". The declared verdicts are unchanged; the centred counts are a diagnostic.
3. *Candidate lengths are matched within every template* (`cand_lengths_{row}.json` via the `noop` stage):
   NLA table column 11 tokens for all six, OLens 3 / 5 / 5, NLA parenthesis 7 for all four, OLens 3; so the
   summed-log-prob top-1 carries no length bias here. The J-lens cell is weaker by construction: each
   candidate is a max over its single-token variants, candidates have different variant counts, and its
   "class AUROC" is a log-sum-exp of cosine scores rather than of log-probs; the J-lens is not part of the
   declared verdict and its row is indicative only.

Cost: four GPU runs of about two minutes each after model load (OL+J-lens on H100, NLA on H200), about $3.

## (c) Vocabulary readout of the exp 15 directions

Every exp 15 unit direction (eight real, 20 isotropic and 20 J-complement nulls) projected through the
unembedding, logits = W_U ((1 + w_norm) v). Statistic: peak ratio = max |logit| / sd over the 248,320-row
unembedding, placed against the 20 isotropic directions. Declared rule: lexical if z > 3 and the statement's own
words are in the top 10. Full tables: `results/vocab17.md`.

| row | direction | peak ratio | z vs iso | top positive tokens (decoded) | top negative tokens |
|---|---|---:|---:|---|---|
| table column | class mean | 5.6 | +0.1 | fragments, no theme | fragments |
| table column | AR(true spans) | 5.7 | +0.3 | 后者 ("the latter"), "\n", 前者 ("the former"), 四 ("four") | 则有 / 则以 ("then has / then by") |
| table column | **AR(statement)** | 6.2 | **+1.6** | 最后是 ("finally is"), " etc", " finally", 마지막으로 ("lastly"), 其他的 ("the others") | 首先 ("first of all") |
| parenthesis | class mean | 7.1 | +3.6 | ”**. ."] ……。 " (" — closing punctuation | 下同 ("same below"), aka |
| parenthesis | AR(true spans) | 7.1 | +3.7 | ...” .") !». '}); — closers | aka, formerly, 以下简称 ("hereinafter") |
| parenthesis | **AR(statement)** | 6.1 | **+1.3** | 您现在的位置 ("your current position"), 地址位于 ("address located at"), 請問 | fragments |

**Verdict by the declared rule: the statement direction is not lexical on either row** (z +1.6 and +1.3; no
English statement word in the top 10). The rank column in `results/vocab17.md` qualifies that: on table
column the statement's own words sit at the two extremes of the 248,320 tokens (" fifth" rank 345 from the
top, " first" rank 248,298, the second most negative), which the top-10 rule misses because the extremes are
shared with multilingual synonyms; on parenthesis the statement words are mid-ranked (" inside" 171,888,
" outside" 192,372) and the direction is carried by the "current position" frame instead. **What the tokens say anyway:** the table-column statement direction
("fifth column" minus "first column") points at the vocabulary of *last-ness* (finally, etc., lastly, the
others) and away from *first-ness* (首先), in several languages; the parenthesis statement direction ("The
current position is ...") points at translations of the statement's own frame ("your current position",
"address located at"). So the direction a statement builds carries the statement's *meaning* at the
vocabulary level, spread over many languages rather than peaked on one token, which the peak-ratio rule
does not catch. The class-mean and span directions for parenthesis are the ones the rule calls lexical
(z +3.6 / +3.7): they load on closing punctuation, which is exactly the next-token decision the model
makes there, consistent with exp 15 (the model follows them) and with the J-lens being flat on this row
despite the punctuation loading (the J-lens's prompted words were "inside / outside / parenthesis", not ")").

Reading for the write-up: the direction the J-lens and OLens follow on table column (exp 15, z +10 / +12)
is a multilingual "last vs first" meaning direction that does not move the model's newline decision. The
rule for "lexical" should be revised to a semantic-cluster test before it is used again; the revision is
not applied here post hoc.

## (d) Cross-scoring with Guiv's bank (26 Oct, his `share/cross_scoring/` files, 84 test items per row)

AUROC of the mean prompted log-odds, 95% cluster-bootstrap CI (clusters = his template families on his items,
passages on ours); OLens at 42, our frozen phrasings (fixed + 3 mined) vs his two frozen pairs. Table:
`results/xs17.md`; per-value signs below from `results/vol/xs_*.pt`.

| row | reader | ours x ours (exp 13) | his items x our phrasings | his items x his phrasings | our items x his phrasings |
|---|---|---|---|---|---|
| table column | NLA | 1.00 (fixed 0.89) | 0.85 [0.79, 0.95] | 0.56 [0.36, 0.76] | 0.64 [0.56, 0.73] |
| table column | OLens | 0.74 | 0.72 [0.59, 0.86] | 0.58 [0.50, 0.67] | 0.36 [0.28, 0.45] |
| parenthesis | NLA | 0.96 | 0.85 [0.71, 0.99] | 0.75 [0.59, 0.93] | 0.83 [0.77, 0.89] |
| parenthesis | OLens | 0.73 | 0.78 [0.69, 0.97] | 0.53 [0.41, 0.67] | 0.50 [0.41, 0.60] |

1. **His 0.57 reproduces (0.56 here) and is mostly phrasing.** With our phrasings his table-column items score
   0.85 (NLA) and 0.72 (OLens); with his phrasings our items fall to 0.64 and 0.36. Items account for the rest
   (0.85 vs 1.00 for the NLA): his "late" is column 5 exactly behind a padding preamble, ours columns 4 to 6 in a
   plain table. The two numbers are two measurements, as the joint update now says, and the phrasing is the
   larger factor. His phrasings ("The current excerpt has the first table column.") were never mined from a
   reader's own register; ours were.
2. **Parenthesis is robust for the NLA (0.75 to 0.85 in every cell) and phrasing-bound for OLens** (0.78 with our
   phrasings, 0.50 to 0.53 with his). The NLA reads the state whichever bank or phrasing; OLens's partial read
   depends on the continuation-style phrasings exp 07 mined for it.
3. **The forced-choice sign is fixed on his items too**, for both readers and both phrasing sets: on table column
   every phrasing prefers the marked value on 95 to 100% of marked items and on 93 to 100% of unmarked items
   (one mined NLA phrasing reaches 0.29); on parenthesis the fixed and his phrasings prefer "inside" on 100% of
   items while two mined continuation phrasings prefer the other side on 100%. Discrimination comes from the
   margin, not the sign, exactly as exp 17 (a) and his E27 found. So the exp 18 result that the NLA names both
   values on ordinal rows is phrasing-specific: with "the {v} argument" it flips, with "has the first table
   column" it does not. The complement rule needs a phrasing-robust form (e.g. the sign under the best of k
   phrasings) before it is pre-registered again.
4. J-lens with our word sets on his items: table column 0.80 [0.70, 0.90] (0.64 on ours), parenthesis 0.47.

