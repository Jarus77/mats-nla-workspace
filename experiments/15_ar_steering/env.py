"""Modal image for 15_ar_steering: exp 13/14's GPU image (pinned torch/transformers/peft + src/mats)
with this directory's helpers shipped in. Inputs come from exp 13 (items, activations, frozen
templates) and exp 14 (NLA-AR reconstructions) on the results volume."""
import os
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

# Local (non-Modal) mode for a bare GPU machine: RFM_LOCAL=1 makes the volume handles no-ops; the
# code then reads/writes the same absolute paths (/results, /weights/hf) on the local filesystem.
if os.environ.get("RFM_LOCAL") == "1":
    modal.Volume.reload = lambda self, *a, **k: None  # type: ignore[method-assign]
    modal.Volume.commit = lambda self, *a, **k: None  # type: ignore[method-assign]

OUT = "/results/15_ar_steering"
SRC13 = "/results/13_new_rows"
SRC14 = "/results/14_ar_blindspot"
BASIS = "/results/03_pipeline_smoke/jspace_basis_L42.pt"
_LOCAL = ("modal_common", "env", "spec15", "lib15", "analyze15")

_PINS = ("torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
         "safetensors==0.8.0", "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_PINS, *HF_PINS)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)
