"""Modal images for 07_template_mining. GPU image = the 03_pipeline_smoke pins (rebuilt here, as
03 does, because modal_common.gpu_image cannot be extended after its local mount), with src/mats
and this directory's helpers shipped in. The CPU build image adds spaCy/datasets (as 02's
nonj_common does) plus read-only copies of 02's generator modules (s1_data, s1b_data, features)."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/07_template_mining"
_E02 = _REPO / "experiments" / "02_nonj_features"
_LOCAL = ("modal_common", "env", "feature_spec", "items_build")

_PINS = (
    "torch==2.14.0",
    "transformers==5.17.0",
    "peft==0.21.0",
    "accelerate==1.15.0",
    "safetensors==0.8.0",
    "numpy==2.5.3",
    "pyyaml",
    "scipy",
    "flash-linear-attention==0.5.2",
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_PINS, *HF_PINS)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)

build_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.5.3", "transformers==5.17.0", *HF_PINS)
    .uv_pip_install("spacy==3.8.7", "datasets", "pandas", "pyarrow")
    .run_commands(
        "uv pip install --system https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "0", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source(*_LOCAL)
    .add_local_file(str(_E02 / "s1_data.py"), "/root/ext02/s1_data.py")
    .add_local_file(str(_E02 / "s1b_data.py"), "/root/ext02/s1b_data.py")
    .add_local_file(str(_E02 / "features.py"), "/root/ext02/features.py")
)
