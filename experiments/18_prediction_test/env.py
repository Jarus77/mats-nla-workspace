"""Modal images for 18_prediction_test: exp 13's two images (GPU: pinned torch/transformers + src/mats;
CPU build: spaCy/datasets + 02's paragraph cleaner) with this directory's helpers, exp 07's
items_build (for the fresh col bank) and exp 08's pilot_items (for the exclusion list) shipped in."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/18_prediction_test"
OUT13 = "/results/13_new_rows"
OUT07 = "/results/07_template_mining"
OUT08 = "/results/08_col_pilot"
_E02 = _REPO / "experiments" / "02_nonj_features"
_E07 = _REPO / "experiments" / "07_template_mining"
_LOCAL = ("modal_common", "env", "feature_spec18", "feature_spec", "items_build18", "lib18")

_PINS = ("torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0",
         "safetensors==0.8.0", "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_PINS, *HF_PINS)
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)

build_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.5.3", "transformers==5.17.0", *HF_PINS)
    .uv_pip_install("spacy==3.8.7", "datasets", "pandas", "pyarrow")
    .env({**HF_ENV, "HF_HUB_OFFLINE": "0", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source(*_LOCAL)
    .add_local_file(str(_E02 / "s1_data.py"), "/root/ext02/s1_data.py")
    .add_local_file(str(_E02 / "s1b_data.py"), "/root/ext02/s1b_data.py")
    .add_local_file(str(_E02 / "features.py"), "/root/ext02/features.py")
    .add_local_file(str(_E07 / "items_build.py"), "/root/ext07/items_build.py")
    .add_local_file(str(_E07 / "feature_spec.py"), "/root/ext07/feature_spec.py")
)

BASIS = "/results/03_pipeline_smoke/jspace_basis_L42.pt"
