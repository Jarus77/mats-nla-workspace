"""Modal image for 17_harden: the exp 16 GPU image with this directory's helpers."""
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

OUT = "/results/17_harden"
SRC13 = "/results/13_new_rows"
SRC15 = "/results/15_ar_steering"
SRC14 = "/results/14_ar_blindspot"
_LOCAL = ("modal_common", "env", "spec17", "lib17")

# Frozen phrasing templates (repo files, not on the volume): shipped into the image, read by templates_path().
_TEMPLATES = {k: _REPO / "experiments" / k / "templates_frozen.json" for k in ("13_new_rows", "07_template_mining")}


def templates_path(exp: str) -> Path:
    """Repo copy on the laptop / RunPod checkout; the image copy inside a Modal container."""
    p = _TEMPLATES[exp]
    return p if p.exists() else Path(f"/root/templates/{exp}.json")

_PINS = ("torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
         "safetensors==0.8.0", "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_PINS, *HF_PINS)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
    .add_local_file(str(_TEMPLATES["13_new_rows"]), "/root/templates/13_new_rows.json")
    .add_local_file(str(_TEMPLATES["07_template_mining"]), "/root/templates/07_template_mining.json")
)
