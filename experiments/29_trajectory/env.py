"""Modal images for 29_trajectory: exp 22's GPU image (pinned torch/transformers + src/mats) with this directory's
helper and the exp 22/23/25/26 modules lib29 imports (`feature_spec22` / `feature_spec` in 22_public_bank, `lib23` in
23_random_points, `lib25` in 25_public_exhibits, `lib26` in 26_layer_sweep). Those directories are appended to sys.path
here (appended, so this directory's own env.py and lib29 stay first) and `add_local_python_source` resolves them by
import, as it resolves any module (exp 25/26's pattern). lib19 (the text bag) is laptop-only and not shipped.
The fetch stage uses modal_common's download_image (no GPU, network on)."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_EXPS = _REPO / "experiments"
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
for _d in (_EXPS / "22_public_bank", _EXPS / "23_random_points", _EXPS / "25_public_exhibits", _EXPS / "26_layer_sweep"):
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.append(str(_d))
from modal_common import HF_ENV, HF_PINS, VOLUMES, download_image  # noqa: E402,F401

OUT = "/results/29_trajectory"
OUT22 = "/results/22_public_bank"
NLA_SNAPSHOT = "/weights/hf/hub/models--ceselder--qwen3.6-27b-nla-rl/snapshots/5a13b7ec21a69fcdd0fb24d5edbe96a92aef4b9f"
NLA_REVISION = "5a13b7ec21a69fcdd0fb24d5edbe96a92aef4b9f"     # the commit the volume snapshot mirrors (= HF HEAD, 28 Oct)
_LOCAL = ("modal_common", "env", "lib29", "lib25", "lib26", "lib23", "feature_spec22", "feature_spec")
_PINS = ("torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
         "safetensors==0.8.0", "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_PINS, *HF_PINS)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)
