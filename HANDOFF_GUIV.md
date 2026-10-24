# Handoff: run exps 15 and 16 and send everything back in one zip

For Guiv (and his Claude / Codex). Written 24 Oct 2026 by Suraj's assistant. Two experiments, already
built and tested, blocked only on GPU access. Expected cost $12 to $17 on RunPod. Expected wall clock
about 2 hours on one H200, less with two cards.

**Please do not change any constant, seed, dose, template or model path. If a stage fails, send the log
and traceback as they are; do not retry with different settings.** Everything we need to continue is in
the deliverables list at the end; we cannot resume if a file is missing.

## What the experiments are

- **Exp 15, AR-mediated steering** (`experiments/15_ar_steering/`): does Qwen3.6-27B use the table-column
  and parenthesis state the NLA reports, and can the NLA's reconstructor build a steering vector for it
  from text? Eight steering directions per concept, centred edits at layer 42, 20 + 20 random-direction
  nulls, model-side continuation metrics, and NLA / Oracle Lens / J-lens prompted readouts.
- **Exp 16, difference-term swap** (`experiments/16_difference_term/`): Neel's suggestion. Freeze the part
  of the activation the Oracle Lens output reconstructs, swap the remainder from another prompt, re-run
  the readers. Four rows.

Design details: the docstrings at the top of `spec15.py` and `spec16.py`.

## Machine

- One GPU with **at least 120 GB** for `prep` (exp 15) and `s1` (exp 16): the NLA plus its reconstructor is
  about 92 GB. H200 (141 GB) or B200. The other stages fit an 80 GB card, so a second cheaper card can
  take them if you run things in parallel, but one H200 for everything is simplest.
- **220 GB of disk** for the weights (about 150 GB) plus outputs.
- CUDA-capable Linux, Python 3.12.

## Setup (once, ~1 hour, mostly the download)

```bash
git clone https://github.com/Jarus77/mats-nla-workspace.git && cd mats-nla-workspace
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.0 transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 safetensors==0.8.0 \
            numpy==2.5.3 pyyaml scipy flash-linear-attention==0.5.2 huggingface-hub==1.32.0 hf-xet==1.6.0 modal pytest
# the code reads absolute paths; create them (or symlink to your big disk)
sudo mkdir -p /results /weights && sudo chown "$USER" /results /weights
cp -R experiments/handoff_inputs/{03_pipeline_smoke,07_template_mining,13_new_rows,14_ar_blindspot} /results/
export HF_TOKEN=hf_...            # your token; only used for download rate limits
bash infra/download_weights_local.sh
pytest experiments/15_ar_steering experiments/16_difference_term -q    # 12 tests, no GPU
```

`modal` is installed only because the stage functions are declared with its decorators; nothing talks to
Modal. `run_local.py` sets `RFM_LOCAL=1`, which turns the volume handles into no-ops.

## Run (one stage at a time, in this order; each stage skips rows whose output already exists)

```bash
mkdir -p logs
cd experiments/15_ar_steering
python run_local.py --stage prep       2>&1 | tee ../../logs/15_prep.log        # H200, ~15 min
python run_local.py --stage behave     2>&1 | tee ../../logs/15_behave.log      # ~60 min, 4 jobs
python run_local.py --stage score_nla  2>&1 | tee ../../logs/15_score_nla.log   # ~20 min
python run_local.py --stage score_ol   2>&1 | tee ../../logs/15_score_ol.log    # ~15 min
python run_local.py --stage analyze    2>&1 | tee ../../logs/15_analyze.log     # CPU, 1 min
cd ../16_difference_term
python run_local.py --stage s1         2>&1 | tee ../../logs/16_s1.log          # H200, ~40 min (4 rows)
python run_local.py --stage s2         2>&1 | tee ../../logs/16_s2.log          # ~25 min
python run_local.py --stage analyze    2>&1 | tee ../../logs/16_analyze.log     # CPU
```

With two cards: after `prep`, `behave` on one and `score_nla` / `score_ol` on the other
(`CUDA_VISIBLE_DEVICES=1 python run_local.py --stage score_nla`). Exp 16 `s1` can run alongside exp 15's
`behave` once `prep` is done. Nothing else overlaps.

If you prefer Modal on your own workspace instead of RunPod: `bash experiments/handoff_inputs/upload_to_modal.sh`,
`modal secret create hf-token HF_TOKEN=...`, `modal run infra/download_weights.py --only qwen,nla,olens_ar,jlens`,
then `modal run experiments/15_ar_steering/modal15.py --stage <stage>` in the same order (`RFM_MAXC=4` to
allow four containers), same for `modal16.py`, and `modal volume get rfm-results 15_ar_steering/ ...` to fetch.

## Success checks

After exp 15 `analyze`, `/results/15_ar_steering/` holds exactly:
`spec15_{tb_tablecol,tb_paren}.pt`, `behave15_{tb_tablecol,tb_paren}_c{0,1}.pt` (4 files),
`score15_nla_{tb_tablecol,tb_paren}.pt`, `score15_ol_{tb_tablecol,tb_paren}.pt`, `analysis15_{tb_tablecol,tb_paren}.json`.
After exp 16 `analyze`, `/results/16_difference_term/` holds `s1_<row>.pt`, `s2_<row>.pt`, `analysis16_<row>.json`
for the four rows `tb_tablecol, tb_paren, ij2_lang, bk_listidx`.
`experiments/15_ar_steering/results/` and `experiments/16_difference_term/results/` hold the small JSON prints.
A stage that reports `"skipped": true` found its output already present; that is fine.

## Deliverables: one zip, sent once, nothing else needed afterwards

```bash
cd /results && zip -r ~/handoff_results_$(date +%Y%m%d).zip 15_ar_steering 16_difference_term
cd <repo> && zip -r ~/handoff_results_$(date +%Y%m%d).zip logs experiments/15_ar_steering/results experiments/16_difference_term/results
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/gpu.txt && zip -j ~/handoff_results_*.zip ~/gpu.txt
```

Roughly 300 MB. Send the zip (Drive link or Slack) with one line: GPU type, total wall clock, total
cost. Please do not push to the repo; we merge the results ourselves.

## If something breaks

- OOM in `prep` or `s1`: the card is under 120 GB. Everything else can still run; send what you have.
- A missing-file error names the path; it means one of the `handoff_inputs` folders was not copied to `/results`.
- Weights not found with `HF_HUB_OFFLINE=1`: the download did not finish; re-run the download script, it resumes.
- Anything else: send the log and the traceback as they are.
