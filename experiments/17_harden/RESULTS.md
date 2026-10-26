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

Cost: four GPU runs of about two minutes each after model load (OL+J-lens on H100, NLA on H200), about $3.

## (c) Vocabulary readout of the exp 15 directions

Every exp 15 unit direction (eight real, 20 isotropic and 20 J-complement nulls) projected through the
unembedding, logits = W_U ((1 + w_norm) v). Statistic: peak ratio = max |logit| / sd over the 151,936-token
vocabulary, placed against the 20 isotropic directions. Declared rule: lexical if z > 3 and the statement's own
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
English statement word in the top 10). **What the tokens say anyway:** the table-column statement direction
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
