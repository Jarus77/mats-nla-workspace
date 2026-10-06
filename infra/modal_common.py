"""Shared Modal resources for the MATS sprint: volumes, images, env.

Every function that touches weights mounts `rfm-weights` at WEIGHTS_DIR and sets
HF_HOME=HF_HOME_DIR, so `from_pretrained("Qwen/Qwen3.6-27B")` (or a repo id plus subfolder)
resolves from the volume's HF cache. Set HF_HUB_OFFLINE=1 in GPU jobs to guarantee no network.

Pinned versions mirror WorkspaceBench's uv.lock (torch 2.14.0, transformers 5.17.0,
peft 0.21.0, accelerate 1.15.0, huggingface-hub 1.32.0, hf-xet 1.6.0).
"""

import modal

WEIGHTS_VOLUME_NAME = "rfm-weights"
RESULTS_VOLUME_NAME = "rfm-results"
WEIGHTS_DIR = "/weights"
RESULTS_DIR = "/results"
HF_HOME_DIR = f"{WEIGHTS_DIR}/hf"

BASE_MODEL = "Qwen/Qwen3.6-27B"
NLA_LAYER = 42  # NLA / olens read layer: output of decoder block 42 (d_model 5120)

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

VOLUMES = {WEIGHTS_DIR: weights_volume, RESULTS_DIR: results_volume}

# huggingface_hub >= 1.0 downloads through hf-xet (hf_transfer is no longer used by the hub
# client). Keep the xet chunk cache off the volume so it does not bloat it.
HF_ENV = {
    "HF_HOME": HF_HOME_DIR,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_XET_CACHE": "/tmp/hf_xet",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

HF_PINS = ["huggingface-hub==1.32.0", "hf-xet==1.6.0"]

# Optional secret: authenticated downloads get higher rate limits.
HF_SECRETS = [modal.Secret.from_name("hf-token")]

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*HF_PINS, "pyyaml")
    .env(HF_ENV)
    .add_local_python_source("modal_common")
)

# GPU image. flash-linear-attention supplies the Gated DeltaNet kernels transformers' qwen3_5
# model picks up automatically (the torch fallback is >10x slower). causal-conv1d ships only an
# sdist that must be compiled against torch; the torch fallback for the short conv is cheap, so
# it is left out.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "peft==0.21.0",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "numpy==2.5.3",
        "pyyaml",
        "flash-linear-attention==0.5.2",
        *HF_PINS,
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("modal_common")
)
