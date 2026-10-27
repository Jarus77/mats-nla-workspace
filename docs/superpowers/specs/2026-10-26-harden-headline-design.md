# Harden the headline cells, then pre-register the prediction test

Date 2026-10-26. Decided with Suraj in chat after the exp 15 / 16 write-up. Private design note.

## Intent

Before the joint update reaches Neel, make the two headline cells (table column, parenthesis) survive the
two attacks we already know about, put confidence intervals on exps 15 and 16, and confirm the
statement-built direction is lexical. Then write the pre-registration for week two's prediction test, so
its predictions are on paper before any reader is run on new concepts.

## Goals

1. Multi-candidate re-score of table column (six column values) and parenthesis (four enclosure states) for
   NLA, Oracle Lens and the J-lens on the 160 exp 13 selection items per row, with cluster-bootstrap CIs.
2. Cluster-bootstrap CIs on the exp 15 paired shifts and the exp 16 follow-AUROCs, from the local `.pt` files.
3. Vocabulary readout of every exp 15 direction through the unembedding: is the statement direction lexical?
4. Cross-scoring with Guiv's bank (his items x our phrasings, our items x his phrasings) as soon as his files
   arrive; the code must run without them (skip, not fail).
5. A pre-registration document for the prediction test: instruments, concept list, one prediction line per
   cell, decision rule, and a hash step before readers run.

## Non-goals

No new concepts are scored in this plan. No retraining. No change to any frozen phrasing, direction, dose or
seed of exps 13-16. No outward message is sent (Neel, Agam, Guiv) by the implementer.

## Declared criteria (written before scoring)

- Table column "survives" Guiv's critique if the NLA's six-way top-1 accuracy has a 95% CI lower bound above
  twice chance (0.33) and its six-way log-sum-exp AUROC (late columns vs first) has a CI lower bound above
  0.80, and both point estimates exceed Oracle Lens's. Anything else is reported as "two-candidate artefact"
  and the claim retreats to the spontaneous columns (41% vs 0%).
- Parenthesis survives if the NLA's four-way top-1 CI lower bound is above 0.50 (twice chance).
- The statement direction is "lexical" if its vocabulary peak ratio (max |logit| / sd over the vocabulary)
  is more than 3 null-SDs above the 20 isotropic directions and its top-10 tokens contain the statement's own
  content words.

## Constraints

- Modal main workspace, profile `glitchcraft-inc`, at most two GPUs at once (`RFM_MAXC` default 1).
- Modal app and function names opaque (no person, project or stage word); CLI stage keys may be descriptive.
- Every stage skips when its output exists; nothing already on `rfm-results` is overwritten.
- Commits as Jarus77, conventional type prefix, no assistant trailers. Nothing pushed until Suraj says "push".
- API keys from `.env`, never printed. Guiv's raw files stay untracked (`experiments/17_harden/inputs/` is
  git-ignored).
- Python 3.12, the repo's `.venv`; tests with `.venv/bin/python -m pytest`; no new dependencies.

## Left to the implementer

Batch sizes within the existing readers' limits; markdown table layout; the exact 20 token ids reported per
direction beyond the ones named in the plan.
