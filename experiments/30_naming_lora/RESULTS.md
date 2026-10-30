# Exp 30: can the NLA be taught to name what it carries, without losing reconstruction?

**Question.** Exps 22, 23 and 25 found that the NLA's explanations carry the structural state (the reconstructor
rebuilds it at 0.93 to 1.00 on the readable rows) while the NLA never names it (statement rate under 0.20 on every
natural row; 0.03 in-string naming at exp 23's random read points). `notes/derivation.md` reads this from the
objective: the GRPO reward pays only through `Delta_u`, the reconstructor's own use of the text, so a true statement
that the reconstructor does not use is never paid for (A2 / C2). This experiment is the cheapest causal test of that
reading (`notes/draft_critique_intervention.md`, design (b)): add a supervised naming term to the NLA's own LoRA,
through the same injection hook it reads at inference, and ask whether naming appears on held-out items at no cost in
FVE. If the verbalizer can be taught to name from the injected activation, the silence was a property of the reward;
if it cannot even when paid for, it is a property of the verbalizer's access.

**Design** (`run30.sh`; app `j9rw`, `modal30.py`). Data (`data30.py`, CPU, `k3vd`): exp 22's selection split per row
(160 items), split 80/20 by cluster (seed 30; clusters never straddle); six training rows (py_string, py_arg,
json_depth, md_heading, md_list, code_lang) and two rows held out entirely for transfer (js_string: same template as
py_string; latex_math: a novel template). Targets: arm **S** = the item's existing k = 1 NLA explanation (exp 25's
`nla_sel_{row}.json`) with the row's fixed NLA statement for the item's TRUE value (`FIXED["nla"][row]`, exp 22)
appended as one final sentence inside the explanation tags; arm **C** (control, prices SFT itself) = the same
explanation with no statement. Items whose explanation is "(empty)" are skipped. The frozen file and its sha256:
`results/data30.json`, `results/frozen_sha256.txt`. Training (`train30.py`, H200, `b4mt`, one arm per invocation):
`load_nla(adapter=iter_000400, load_ar=False, trainable=True)`, which continues the EXISTING r64 rsLoRA adapter
(its `lora_` parameters are the only trainable ones; everything else frozen), gradient checkpointing (non-reentrant),
bf16 autocast, a cloning twin of the injection hook so autograd flows through the injection; loss = mean token NLL of
the target given the injected activation (`nla.nll_of_targets`, the gradient twin of `logprob_of_strings_batch`:
same prompt, same hook, teacher-forced, loss on the tokens after `<explanation>\n` through `</explanation>` + eos);
AdamW lr 1e-5 (20-step warmup), batch 8 x about 400 tokens, 400 steps (about 4 epochs over about 770 items), grad
clip 1.0; the adapter is saved in peft's layout every 100 steps (resumable) to `adapter_{arm}/`, the loss logged every
10 steps to `train30_{arm}.json`. Eval (H200, `l7kz`, `--arm base | S | C`, AR loaded): on the held-out clusters of
all eight rows, k = 1 readouts, FVE through the reconstructor (gate set = the held-out items of the six training rows,
one predict-the-mean baseline shared by all arms), statement accuracy (the readout names the correct value with the
row's fixed statement; the complement counts as wrong; naming rate = either), the two transfer rows, and exp 23's 500
random read points (in-string naming when the state holds and when it does not, lib23's strict regex, plus the
statement-level check). `analyze30.py` -> `results/table30.md`, `results/fig30.png`. `p5_baseline.py` scores the
design-(c) baseline from exp 22's existing k = 8 readouts if they carry per-sample FVE; they carry raw texts only, so
P5 is recorded as skipped unless a `results/p5_fve.json` is produced.

**Cost.** About $30 for one clean pass (two H200 training arms of about 45 min each including the load, three H200
eval passes of about 25 min, one CPU data stage), $100 to $150 with reruns. Never more than one GPU at a time.

## Predictions written before the run (copied from the memo, `notes/draft_critique_intervention.md` §2)

- **P1.** Gate-set FVE of the statement arm within 0.02 of the control arm, both within 0.03 of 0.761.
- **P2.** On held-out clusters of trained readable rows (py_string, json_depth, py_arg) statement accuracy >= 0.85;
  on md_list (readable 0.49) <= 0.65.
- **P3.** Transfer py_string -> js_string accuracy >= 0.70; latex_math (novel template) naming of "math" < 0.30.
- **P4.** Exp 23 in-string naming rises from 0.03 to >= 0.50 when the state holds, <= 0.15 when it does not.
- **P5** (baseline c). Within-item FVE difference, naming minus non-naming, |mean| <= 0.02.

**What falsifies the account.** P2 fails while P1 holds: the verbalizer cannot read the state through the injection
even when paid for it, so the silence is capacity, not reward, and C2's "paid only through `Delta_u`" is the wrong
explanation. P1 fails in the statement arm only (drop >= 0.05): the reconstructor charges for statements rather than
ignoring them; `Delta_u` is negative, C2 must be amended from "indifferent" to "hostile", the objective-level account
survives in changed form. P5 fails with naming samples scoring higher: the reward already pays for statements and A2
is false in effect; longer GRPO would name. F2 (the reflex: naming rate 1.0 with accuracy at chance on held-out
clusters) is why statement accuracy, not naming rate, is the primary cell, and why both values of a row sit in
training. F3 (FVE falls in the statement arm but not in the control arm) is a result, read against the control arm.

**Gate.** If the training forward/backward fails under the pinned kernels (the memo's F1: backward through the
DeltaNet kernels of flash-linear-attention 0.5.2, or the hook under gradient checkpointing, or peft not resuming the
rsLoRA cleanly), this experiment ships the frozen data file and the eval of the base adapter only (`eval30_base.json`)
and reports the failure with the traceback; the fallback named in the memo (a new r16 adapter on the attention
projections only) is a separate decision, not run silently.

## Results (27 Oct, 23:53 UTC; `results/table30.md`, `results/eval30_{base,S,C}.json`, `results/train30_{S,C}.json`, `results/fig30.png`; two training arms of 400 steps, 13 minutes each on one H200 per `pipeline.log` (13.4 min launch to done, as `table30.md` now prints; the `wall_s` of 35,699 s and 36,492 s in `train30_{S,C}.json` over-count, because `train30.py` re-added the time since the last checkpoint at every step, and are not training time); three evaluations of about 10 minutes; about $8)

Training ran: the backward pass through the injection hook and the hybrid attention kernels worked at the first
step; arm S loss 1.81 to 1.33, arm C 1.38 at the end (the control's targets have no appended sentence, so its loss
is not comparable step for step). Statement arm S = the NLA's own explanation plus one true statement of the row's
state; control arm C = the explanation alone; base = the untouched iter_000400 adapter.

**Held-out source files, 32 items per row** (naming = the readout contains a statement of either value; correct
= the statement names the true value; FVE = reconstruction score through the NLA's own critic):

| row | trained on | base naming / FVE | control naming / FVE | arm S naming | correct when it names | arm S FVE |
|---|---|---|---|---:|---|---:|
| py_string | yes | 0.00 / 0.54 | 0.00 / 0.54 | 0.59 | 15 of 19 (0.79) | 0.55 |
| py_arg | yes | 0.00 / 0.34 | 0.00 / 0.36 | 0.50 | 13 of 16 (0.81) | 0.33 |
| json_depth | yes | 0.00 / -0.02 | 0.00 / -0.02 | 0.72 | 22 of 23 (0.96) | 0.02 |
| code_lang | yes | 0.00 / 0.44 | 0.00 / 0.47 | 0.28 | 9 of 9 (1.00) | 0.46 |
| md_heading | yes | 0.00 / 0.65 | 0.00 / 0.64 | 0.81 | 15 of 26 (0.58) | 0.62 |
| md_list | yes | 0.00 / 0.50 | 0.00 / 0.50 | 0.75 | 11 of 24 (0.46) | 0.50 |
| js_string | held out | 0.00 / 0.54 | 0.00 / 0.57 | 0.53 | 10 of 17 (0.59) | 0.54 |
| latex_math | held out | 0.00 / 0.59 | 0.00 / 0.59 | 0.03 | 1 of 1 | 0.60 |

Gate-set FVE (192 held-out items of the six training rows): base 0.504, control 0.512, statement arm 0.507.

**Exp 23's 500 random read points** (the adapter's readout classified as stating "inside a string" or "outside a
string" by the row's fixed statement, lib30's substring check, against the true label; recounted from the readouts in
`eval30_{arm}.json`):

| arm | when the point is inside a string (n = 134): says inside / outside / nothing | when it is not (n = 366) |
|---|---|---|
| base | 0.00 / 0.00 / 1.00 | 0.00 / 0.00 / 1.00 |
| control | 0.00 / 0.00 / 1.00 | 0.00 / 0.00 / 1.00 |
| statement arm | 0.12 / 0.24 / 0.64 | 0.04 / 0.21 / 0.75 |

### Scoring the predictions

| prediction | result |
|---|---|
| P1 gate FVE within 0.02 of the control arm, both within 0.03 of 0.761 | partial: the first clause holds (0.507 against 0.512, base 0.504; no row moves more than 0.04); the second fails, both arms about 0.25 below 0.761 (S -0.254, C -0.249; the base adapter is at 0.504 on this gate set too) |
| P2 statement accuracy at or above 0.85 on held-out files of readable rows (accuracy counts items with no statement as wrong) | fails as defined: 0.41 to 0.69, because the adapter states on 50 to 72 percent of items; conditional on stating, 0.79 to 1.00 on the four readable code rows, 0.46 to 0.58 on the two Markdown rows (the md_list line of at most 0.65 holds) |
| P3 transfer: js_string accuracy at or above 0.70; latex_math naming under 0.30 | fails on js_string (0.31; conditional 0.59 on 17 statements); holds on latex_math (0.03) |
| P4 random points: in-string naming at or above 0.50 when the state holds, at or below 0.15 when not | fails: by lib23's strict regex 0.40 when the state holds against 0.26 when not (`table30.md`); by the statement check "inside" 0.12 against 0.04, "outside" 0.24 against 0.21 |
| P5 within-item FVE of naming versus non-naming samples | not scorable: exp 22's k = 8 readouts carry no per-sample FVE |

## Reading

1. **The reward is indifferent to the statements.** Appending a true statement to every training target leaves
   the reconstruction score where the control arm leaves it, on every row and on the gate set. This is the
   derivation's assumption A2 tested on this reader rather than cited from its training card: the critic neither
   pays for the statement nor charges for it.
2. **The verbalizer can read the state through the injection, where the state is there.** On the four readable
   code rows, the statements the adapter makes are right 79 to 100 percent of the time on source files it never
   saw; on the two Markdown rows, where the layer-42 probe is at 0.63 to 0.72, it fires the template three times
   in four and is right at chance. Accuracy follows readability, not training exposure. So the gap on the readable
   rows was never capacity.
3. **But what was learned is narrow.** A row of the same kind not in training (JavaScript strings) gets statements
   on half its items, right 59 percent of the time; a row with a template never seen gets none; and at random read
   points inside strings the adapter mostly says "outside" regardless of the truth, with a weak "inside" signal
   (0.12 against 0.04). Four hundred supervised steps on 768 items teach the trained templates at the trained
   kind of read point, not a general habit of reporting state. Three of the five pre-registered lines fail on their
   definitions, and the failures are the generalisation lines; P1 holds only in its first clause (the absolute
   FVE line at 0.761 fails for both arms alike).
4. **What this changes.** The sentence "the gap is the objective, not capacity" is supported for the trained rows
   and bounded by the transfer rows. The causal claim that fixing the reward would make the readers name what they
   carry is not shown here; what is shown is that naming costs the reward nothing and that the information is
   reachable. Exp 29's trajectory decides whether the reward or the warm start set the register.

Cost: about $8. Code paths added to `src/mats/readers/nla.py` for training are the only change outside this
directory.
