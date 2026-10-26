# 17 — Harden the headline cells (26 Oct 2026)

Before the joint update reaches Neel: (a) re-score table column and parenthesis with the full candidate set
(Guiv's critique of two-candidate log-odds), (b) confidence intervals on exps 15 and 16 (`ci_prior.py`;
tables appended to those experiments' RESULTS), (c) a vocabulary readout of the exp 15 directions (is the
statement-built direction lexical?), (d) cross-scoring with Guiv's bank when his files arrive. Declared
criteria: `docs/superpowers/specs/2026-10-26-harden-headline-design.md`. App `b7kx`; stages in `modal17.py`.

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
