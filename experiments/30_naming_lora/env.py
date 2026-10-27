"""Modal image for 30_naming_lora: exp 25's pattern (pinned torch 2.14 / transformers 5.17 / peft 0.21 /
flash-linear-attention 0.5.2 + src/mats) with this directory's helpers, exp 22's row spec (`feature_spec22`,
`feature_spec`: the fixed statement templates) and exp 23's `lib23` (the strict naming regex for the random-point
eval). Both sibling directories are appended to sys.path (appended, so this directory's env.py and lib30 stay first)
and `add_local_python_source` resolves them by import."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_EXP22 = _REPO / "experiments" / "22_public_bank"
_EXP23 = _REPO / "experiments" / "23_random_points"
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
for _d in (_EXP22, _EXP23):
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.append(str(_d))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/30_naming_lora"
OUT22 = "/results/22_public_bank"
OUT23 = "/results/23_random_points"
OUT25 = "/results/25_public_exhibits"
_LOCAL = ("modal_common", "env", "lib30", "data30", "train30", "feature_spec22", "feature_spec", "lib23")
_PINS = ("torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
         "safetensors==0.8.0", "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_PINS, *HF_PINS)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)
