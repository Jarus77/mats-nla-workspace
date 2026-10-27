"""Modal image for 20_ar_structure: exp 18/19's GPU image with this directory's helper."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/20_ar_structure"
OUT19 = "/results/19_natural_bank"
OUT13 = "/results/13_new_rows"
_LOCAL = ("modal_common", "env", "lib20")
_PINS = ("torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
         "safetensors==0.8.0", "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_PINS, *HF_PINS)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)
