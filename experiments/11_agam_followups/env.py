"""Modal image for 11_agam_followups: the 10 image recipe, src/mats, this directory's helpers and the
pilot item banks (data/items.json: 60 col + 120 tense items with token ids; data/pilot_L42.pt: the 08
round-5 L42 activations, directions and doses for the k=8 cell)."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/11_agam_followups"
BASIS_DIR = "/results/03_pipeline_smoke"
ACTS_05 = "/results/05_planted_intermediates/readers_base.pt"
ACTS_10 = "/results/10_operand_gate/acts_hs43.pt"
ITEMS_REMOTE = "/root/data/items.json"
PILOT_REMOTE = "/root/data/pilot_L42.pt"
_LOCAL = ("modal_common", "env", "a11_modal_lib")

app_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0", "safetensors==0.8.0",
        "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2", *HF_PINS,
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
    .add_local_file(str(_HERE / "data" / "items.json"), ITEMS_REMOTE)
    .add_local_file(str(_HERE / "data" / "pilot_L42.pt"), PILOT_REMOTE)
)
