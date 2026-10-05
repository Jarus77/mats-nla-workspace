"""Modal image for 03_pipeline_smoke: infra pins (modal_common.gpu_image cannot be extended after
its local mount, so the same pins are rebuilt here) + scipy, with src/mats and this directory's
helpers shipped in. Mirrors experiments/00_nla_gate/env.py."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/03_pipeline_smoke"

app_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "peft==0.21.0",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "numpy==2.5.3",
        "pyyaml",
        "scipy",
        "flash-linear-attention==0.5.2",
        *HF_PINS,
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/root/src"})
    .add_local_python_source("modal_common", "env", "smoke_items", "normrerun_lib")
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)
