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

## Results

Not run yet.
