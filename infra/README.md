# Modal infra (MATS sprint)

Workspace: `<modal-workspace>` (Modal CLI 1.5.x). Run commands from inside `infra/` (or from the
repo root with the `infra/` prefix); `modal_common.py` is shipped to containers through
`add_local_python_source`.

| File | What it does |
|---|---|
| `modal_common.py` | Volumes (`rfm-weights` at `/weights`, `rfm-results` at `/results`), the download image and the GPU image (pins match WorkspaceBench's `uv.lock`), HF env |
| `download_weights.py` | App `mats-download-weights`: one CPU container per repo, `snapshot_download` into `/weights/hf` (HF cache layout), commits the volume after each repo |
| `smoke_test.py` | App `mats-smoke-test`: H100, loads Qwen3.6-27B bf16 offline, chat prompt with `enable_thinking=False`, hooks the layer-42 residual, generates 20 tokens, reports peak memory, writes `/results/smoke_test.json` |

## 1. Download weights (~175 GB total)

```bash
cd infra
modal run --detach download_weights.py                    # all five repos in parallel
modal run --detach download_weights.py --only qwen,lenses # subset (keys: qwen, nla, olens_ar, olens_default, lenses)
```

What is fetched:

| Key | Repo | Selection | Size |
|---|---|---|---|
| `qwen` | `Qwen/Qwen3.6-27B` | everything | 55.6 GB |
| `nla` | `ceselder/qwen3.6-27b-nla-rl` | `av_base/*`, `av_rl_adapters/iter_000{300,400}/*`, `ar_reconstructor/*`, `nla_meta.yaml`, `run_config.yaml`, `data/*`, `README.md` (not `ar_sft_checkpoints`, not `ar_reconstructor_sft_init`) | ~93 GB of 402 GB |
| `olens_ar` | `agu18dec/olens_and_ar` | everything (`olens_s3d_rl600/`, `ar_ptag_pooled/`) | 1.0 GB |
| `olens_default` | dataset `agu18dec/local-workspace` | `ckpts/ao/rl/s3d.ddp600.s0/iter_000600/*` (WorkspaceBench default olens LoRA) | 0.9 GB |
| `lenses` | `camilablank/workspace-lenses` | `qwen3.6-27b/**` (j-lens, r-lens, template-lens, ...) | 24.9 GB |

Each repo is an independent `map` input with 2 retries; a failure in one does not stop the
others. `snapshot_download` resumes, so rerun the same command (or `--only <key>`) after a failure.
Downloads use hf-xet (huggingface_hub >= 1.0 no longer uses hf_transfer); auth comes from the
`hf-token` Modal secret.

**Known blocker: `olens_default` fails with 403 GatedRepoError.** `agu18dec/local-workspace` is
gated (auto-approve). The `hf-token` Modal secret authenticates as HF user `<hf-user>`, which has
not accepted the gate. Fix: log in to huggingface.co as the token's owner, open
https://huggingface.co/datasets/agu18dec/local-workspace and request access (auto-approved),
or put a token from an account that has access in the secret
(`modal secret create hf-token HF_TOKEN=hf_... --force`), then
`modal run download_weights.py --only olens_default`.
Fallback in the meantime: `agu18dec/olens_and_ar` (downloaded) contains `olens_s3d_rl600/`, which
by name is the same s3d RL step-600 olens; pass its path as the olens LoRA dir (check its
`adapter_config.json` against the WorkspaceBench default before relying on it).

Monitor:

```bash
modal app list | grep mats              # state of mats-download-weights
modal app logs <app-id>                 # per-repo "[key] done -> ... volume committed" lines
modal volume ls rfm-weights hf/hub     # one models--*/datasets--* dir per repo
modal volume ls rfm-weights hf/hub/models--Qwen--Qwen3.6-27B/snapshots
```

## 2. Use the weights from any function

```python
from modal_common import VOLUMES, gpu_image
@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100")
def f():
    # HF_HOME=/weights/hf and HF_HUB_OFFLINE=1 are set in gpu_image
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda")
    snapshot_download("agu18dec/local-workspace", repo_type="dataset",
                      allow_patterns=["ckpts/ao/rl/s3d.ddp600.s0/iter_000600/*"])  # resolves from cache
```

Sub-folder repos (`agu18dec/olens_and_ar`, `ceselder/qwen3.6-27b-nla-rl`,
`camilablank/workspace-lenses`) have no root `config.json`; load with
`snapshot_download(repo)` + a path into the snapshot, or `subfolder=`. WorkspaceBench's
`wsbench produce` works unchanged because it calls `snapshot_download` with the same patterns.
Write outputs to `/results/...` and call `VOLUMES["/results"].commit()`.

## 3. Smoke test (after the `qwen` download has finished)

```bash
cd infra
modal run smoke_test.py               # H100
modal run smoke_test.py --gpu H200
modal volume get rfm-results smoke_test.json -
```

Result of the first run (2026-10-05, app `ap-tJHdz5HKQO8UkaZGoaOHQV`, H100 80GB):
`AutoModelForCausalLM` gives `Qwen3_5ForCausalLM` (`qwen3_5_text`, text tower only), 64 decoder
blocks, load 36 s, layer-42 residual `[28, 5120]` with last-token norm 92.8 (mean 99.5),
greedy output "The capital of France is Paris, which is famous for its iconic landmarks like the
Eiffel Tower,", 20 tokens in 4.1 s, `max_memory_allocated` 54.1 GB. `fla_installed: true` means the
flash-linear-attention Gated DeltaNet kernels are in use (the torch fallback is >10x slower).
causal-conv1d is not installed (only an sdist that must compile against torch); transformers
falls back to torch for the short conv and logs a "much slower" warning. Generation runs at
~5 tok/s with HF `generate`; if decode speed matters, build causal-conv1d into the image
(`uv_pip_install("causal-conv1d", extra_options="--no-build-isolation", gpu="H100")` on a CUDA
devel base) or use the `kernels` hub path.

## GPU plan

| Workload | GPU | Why |
|---|---|---|
| logit lens / jlens / rlens | H100 80GB | one 27B bf16 copy (~54 GB) plus lens tensors |
| olens (LoRA verbalizer) | H100 80GB | the LoRA sits on the probed model; adapter off for capture, on for generation |
| nla (layer 42) | H200 141GB | base model + AV (activation verbalizer) + AR (reconstructor): a second 27B on the same device |

Image versions: torch 2.14.0 (cu130), transformers 5.17.0, peft 0.21.0, accelerate 1.15.0,
flash-linear-attention 0.5.2, huggingface-hub 1.32.0, hf-xet 1.6.0, Python 3.12.
