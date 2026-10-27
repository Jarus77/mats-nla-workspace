"""Modal image for 25_public_exhibits: exp 22's GPU image (pinned torch/transformers + src/mats) with this directory's
helper and exp 22's row spec. `feature_spec22` / `feature_spec` live in experiments/22_public_bank; that directory is
appended to sys.path here (appended, so this directory's own env.py and lib25 stay first) and
`add_local_python_source` resolves them by import, as it resolves any module."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_EXP22 = _REPO / "experiments" / "22_public_bank"
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
if _EXP22.is_dir() and str(_EXP22) not in sys.path:
    sys.path.append(str(_EXP22))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/25_public_exhibits"
OUT22 = "/results/22_public_bank"
_LOCAL = ("modal_common", "env", "lib25", "feature_spec22", "feature_spec")
_PINS = ("torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
         "safetensors==0.8.0", "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_PINS, *HF_PINS)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)
