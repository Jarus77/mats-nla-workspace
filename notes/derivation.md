# Why the readers carry structural state and do not name it: the derivation

Written 27 Oct 2026, after exps 00 to 23. This is the 24 Oct proposal's section 2b ("the theory, stated so it
predicts") redone with the assumptions explicit, one equation, the experiment that tested each step, and what
each step would have looked like if it were false. It is a derivation from the readers' training objectives, not
a theorem; the last section says why a theorem is not available and would not help.

## 1. Objects

- `h`: the layer-42 residual activation at a read token of Qwen3.6-27B (d = 5120).
- `c`: a binary structural state at that token (inside a string literal, call-argument index, JSON depth, table
  column, parenthesis, math mode). `d_c = mu_c+ - mu_c-` is the class-mean direction in activation space;
  `gap_c = ||d_c||`.
- `AR`: an activation reconstructor. A map from a short text `t` (the training crop is the last 32 tokens) to a
  predicted activation. Each reader has one: Oracle Lens's `ar_ptag_pooled` (tags for layers 20..60 step 4; output
  in a whitened space) and the NLA's own reconstructor (raw space, layer 42). Exp 14 and 20 use both.
- A reader is a policy `pi(t | h)` producing text from `h`.

## 2. Assumptions

A1 (the reconstructor is a window map). On its training distribution, `AR(t)` is close to the activation the
subject model itself produces after reading `t`: `AR(t) ~ E[h' | the text ends in t]`. Evidence: retrieval@1 0.94 on
held-out spans for the Oracle Lens AR (lit review); the NLA critic rebuilds every readable state from the true
32-token span at 0.82 to 0.95 and the Oracle Lens AR at 0.93 to 1.00 (exp 20 table, exp 14 table), while the
model's own activation on the same window carries it at 0.94 to 1.00 (exp 20, "model, window 32").

A2 (the reward runs through the reconstructor). Both readers were trained by GRPO with a reward that is a
decreasing function of `||h - AR(t)||` (the NLA in raw space with the FVE normalisation; Oracle Lens in the AR's
whitened space, AR frozen), plus a KL term toward the warm start. No term in either reward reads `t` as a
description. Evidence: the training cards (lit review, "RL: GRPO with reward -log||h - AR(z)||^2, KL toward init";
"RL then rewards bullet FVE with the AR frozen").

A3 (the register comes from the warm start, not the reward). The NLA starts from model-written summaries of the
prefix (an explanation register); Oracle Lens starts from bullets chosen to maximise FVE. The KL term keeps each
near its register. Evidence: the NLA's readouts are explanations with a `<explanation>` tag; Oracle Lens's are
`- ` bullets; neither register changed over RL (exp 07 to 13 readouts).
Exp 29 tests A3 directly on the NLA's own history: structural naming is 0.17 to 0.49 for the text-only warm start,
0.01 to 0.21 for the activation warm start, and 0.00 to 0.29 with no trend through 800 RL steps, while FVE rises
0.18 to 0.59 (Python strings) and code-language naming rises (judge 0.55 to 0.71-0.77; regex 0.46 to 0.62-0.75). The register is set before RL and RL keeps it.

## 3. The one equation

Write a readout as `t = s (+) u`: a part `s` that exhibits the text (a continuation, a quoted fragment, a
description of the genre) and an appended statement `u` of the state ("inside a string literal", "column five").
The reward's sensitivity to `c` through `u` is the fraction of the class gap that the statement moves the
reconstructor along the class direction:

    Delta_u(c) = < AR(s (+) u_c+) - AR(s (+) u_c-) , d_c / gap_c > / gap_c

This is exp 14's "gap kept" and the exp 18, 19 and 22 "statement shift" instrument (`stmt`), measured with `s`
fixed and `u` the frozen statement pair. The same quantity with `u` empty and `s` varying over spans in state
`c+` versus `c-` is the "span-sayable" instrument (`span`), reported as a frozen-probe AUROC on `AR(s)`.

## 4. Consequences

C1 (what the optimum says). Under A2, the policy's optimum emits the text whose reconstructor image is nearest
`h`. Under A1 that is text the subject model would have been reading when it produced `h`: for Oracle Lens a
sample of the context, for the NLA a description whose image lands near `h`. Nothing in the optimum refers to
what a human would want said about `h`.

C2 (a statement is paid only through Delta_u). A statement `u` of `c` raises the reward by an amount that is zero
when `Delta_u(c) = 0`, and the KL and length terms charge for it. So at the optimum the reader states `c` only
when `Delta_u(c)` is large. This is the revised NLA rule, "states iff statement-sayable", scored 8 of 8 on exp 19
and 6 of 8 on exp 22 (one miss at `stmt` 0.2976 against the 0.30 line, stated at 0.70, the positive control; one
at `stmt` 0.13 stated at 0.25 against the 0.20 cell line, JSON depth). The positive cases are language (`stmt`
0.51, stated 0.95, exp 18) and code language (0.30, stated 0.70, exp 22); the sixteen structural rows of exps 19
and 22 all have `stmt` at or under 0.23 and are stated at 0.25 or less, fourteen of them under 0.20.

Counter-rows, added 28 Oct after the theory review (`notes/draft_critique_theory.md`): four rows in the record do
not fit the 0.30 rule as stated. Table column (shift 0.07, stated 0.41) and parenthesis (-0.05, stated 0.30) from
exps 13 and 14; call argument in exp 18 (0.09, stated 0.33); bullet index (0.23, stated 0.25). Over the 30 rows
with both numbers the Spearman correlation of shift with stated rate is 0.43, and 0.21 once the three language
rows are removed: the data are two clusters (self-inducing rows stated, structural rows mostly not) with no graded
relation inside the lower cluster, and the shift is measured from one statement pair per row with no interval
(exp 14 found paraphrase moves it from 0.10 to 0.44 on parenthesis). So C2 as a threshold is not established; what
is established is the two-type picture plus a warm-start register term that lets the NLA state pending structure
("a new row is beginning") on some synthetic rows. Exp 28 measures the shift with eight paraphrase pairs per row on
every row with a stated rate and tests whether a graded relation exists at all.

C3 (which states have Delta_u = 0). Reading "column five" does not put a model at column five; reading a French
sentence does put it in French. Call `c` self-inducing when a statement of it induces it. Position-in-text states
are not self-inducing, so under A1 `Delta_u = 0` for them: exp 14 measures 0.03 (table column) and 0.10
(parenthesis, descriptive statement) through the Oracle Lens AR, 0.07 and -0.05 through the NLA's; language is
0.10 by statement through the Oracle Lens AR but 1.39 when the bullet is simply written in French, and 0.72 by
statement through the NLA's AR (whose warm start was descriptions, so a description of language is partly
in-distribution for it). Exp 18, 19 and 22 give `stmt` between -0.03 and 0.23 for every structural row.

C4 (why Oracle Lens never states and the NLA sometimes does). For Oracle Lens the register is bullets of context
(A3), and for any self-inducing `c` a bullet written in `c` earns more than a bullet stating `c` (1.39 against
0.10), so stating is dominated at the optimum for self-inducing states and worthless for the rest: 6 of 13,420
readouts state a concept in exps 07 to 13, the state is named where it holds at random read points 0 of 500 times in
exp 23 and 1 of 1,939 in exp 27, max 1 percent on every row of
exps 18, 19, 22. For the NLA the register is description (A3), so a statement is in-register and costs little KL;
it is emitted where `Delta_u` pays (language, code language) and not where it does not (every structural row).

C5 (carrying without naming). By A1, a readout `s` that exhibits text in state `c` has `AR(s)` on the `c+` side
of `d_c` whether or not `u` is present. So the state is recoverable from the readout by any reconstructor-like
decoder, to the extent that a 32-token span can carry it (the `span` instrument). Exp 20: Oracle Lens readouts
rebuild the six readable natural rows at 0.70 to 0.94, the NLA's at 0.93 to 0.99, the same tokens shuffled at
0.30 to 0.65; a bag-of-tokens classifier on the readout text alone agrees (0.72 to 0.96 on the string and depth
rows). Exp 24 repeats the decode through Oracle Lens's reconstructor, which was trained on spans and on neither
reader's output: Oracle Lens's readouts carry the state along the span's own direction (frozen probe 0.73 to 0.80 on
five of six readable rows), the NLA's explanations carry it in a direction of their own (frozen probe at chance, a
probe fit on the explanations' rebuilds 0.81 to 0.90), and shuffling either readout's tokens costs 0.2 to 0.3 on
every carrying row. So C5 holds for both readers with a distinction: Oracle Lens carries `c` as a continuation
would (A1 applies to its output directly), the NLA carries it as a description that a span-trained decoder does
not place on the span axis. Whether a decoder reads vocabulary or arrangement is the decoder's property, not the
readout's: a bag-of-tokens classifier recovers the string rows from vocabulary (exp 20), a window map needs the
order (exp 24).

C6 (use is invisible to the reward). Whether the subject model acts on `c` (the USED push) enters neither A1 nor
A2, so naming cannot track use. Exp 19: LaTeX math is the most used row (push z +25.6) and is never stated; the
exp 18 rule "states iff used" scored 5 of 10 and was dropped.

## 5. What each assumption would have looked like if false

- A1 false (the reconstructor reads statements as descriptions): `Delta_u` large for structural statements, so
  C2 would predict stating and the readers would state position freely. Observed 0.03 to 0.10. The partial
  failure of A1 is real and measured: the NLA's reconstructor moves 0.72 on a language statement, which is the
  one place the NLA states.
- A2 false (a naming term in the reward): stating would not track `Delta_u`. Hong's fact-check reward (lit
  review) shows what happens when a text-reading term is added: the fact score triples with FVE flat. A
  statement-aware critic is the retraining experiment this derivation points at, and it is the one proposal in
  the write-up.
- A3 false (register set by the reward): the NLA would drift toward continuations like Oracle Lens, since
  continuations earn more per token under A1 and A2. It did not; the register is stable across checkpoints (exp 29: steps 100 to 800).
- C5 false (readouts do not carry the state): the shuffled control and the readouts would score alike. Observed
  0.30 to 0.65 against 0.70 to 0.99.

## 6. Limits of the derivation

- C2 is weaker than its wording in section 4: see the counter-rows paragraph there. The 0.30 line was set from
  exp 14's four values, and any line between 0.24 and 0.71 would have scored the same rows the same way.
- It is about the optimum of the reward. The trained readers are not at it (NLA FVE 0.71, Oracle Lens AR FVE
  0.22 whitened), so every "iff" is a threshold on a continuous quantity: the 0.30 line on `stmt` is set by
  exp 17 and 19 and the positive control sits at 0.2976. The ordering of rows by `stmt` is what the derivation
  predicts; the line is ours.
- It does not say which descriptions the NLA's register makes cheap. "Names the complement iff both values are
  positive states" is an observed regularity (6 of 8, 6 of 8) with no derivation here.
- It treats `AR` as one map. Exp 20's reconciliation with exp 16 found the two reconstructors agree in direction
  and differ in degree (Oracle Lens's readout carries table column at 0.64 through the NLA critic, while exp 16
  put 0.98 of the position signal in the part of the vector Oracle Lens's words do not rebuild); exp 24 is the
  check.
- It says nothing about hallucination (exps 05, 06), which is what the reconstructor rewards when the context
  is underdetermined by `h`.

## 7. Why not a theorem

The only statements provable here are immediate from the definitions: "a token sequence whose reconstructor
image equals the empty one earns zero and costs KL, so the optimum omits it" (C2) and a data-processing bound
`I(c; t) <= I(c; AR(t)) + I(c; t | AR(t))`, which holds for every reader and every concept and says nothing
about which rows are exhibited. `AR` and `pi` are learned networks; which concepts are self-inducing is an
empirical property of the subject model that no derivation supplies, and exp 14 measured it. A formal bound on
what a text-trained reconstructor can keep (the 24 Oct "theory day") was dropped for this reason: it would hold
everywhere and predict nothing the `span` instrument does not already measure. What the derivation buys is the
order of the experiments: it says which quantity to measure (`Delta_u`), which direction a failure would take,
and which retraining would change the behaviour.

## Pointers

exp 14 `experiments/14_ar_blindspot/RESULTS.md` (statement shifts through both reconstructors); exp 16
`experiments/16_difference_term/RESULTS.md` (remainder decomposition); exp 18, 19, 22 `results/instr2*.json`
(`span`, `stmt`); exp 20 `experiments/20_ar_structure/RESULTS.md` (carrying, text-bag check); exp 23
`experiments/23_random_points/RESULTS.md` (naming at random points); lit review `notes/lit_review.md` (training
cards). Proposal section 2b: `notes/draft_proposal_2026-10-24.md`.
