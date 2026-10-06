"""Modal images for 09_ol_mechanism: the 08 col-pilot pins (same image recipe), src/mats, this directory's
helpers, and read-only copies of 08 (pilot_items, calib_items, pilot_lib, r5_lib) and 02 (s1_data, features)."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/09_ol_mechanism"
OUT02 = "/results/02_nonj_features"
OUT04 = "/results/04_backup_features"
OUT07 = "/results/07_template_mining"
OUT08 = "/results/08_col_pilot"
_E02 = _REPO / "experiments" / "02_nonj_features"
_E08 = _REPO / "experiments" / "08_col_pilot"
_LOCAL = ("modal_common", "env", "o9_templates", "o9_items", "o9_lib")


def _ext(img: modal.Image) -> modal.Image:
    for f in ("pilot_items.py", "calib_items.py", "pilot_lib.py", "r5_lib.py"):
        img = img.add_local_file(str(_E08 / f), f"/root/ext08/{f}")
    for f in ("s1_data.py", "features.py"):
        img = img.add_local_file(str(_E02 / f), f"/root/ext02/{f}")
    return img


app_image = _ext(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0", "safetensors==0.8.0",
        "numpy==2.5.3", "pyyaml", "scipy", "pandas", "pyarrow", "flash-linear-attention==0.5.2", *HF_PINS,
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
)

build_image = _ext(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.5.3", "transformers==5.17.0", *HF_PINS)
    .uv_pip_install("spacy==3.8.7", "datasets", "pandas", "pyarrow")
    .run_commands(
        "uv pip install --system https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "0", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source(*_LOCAL)
)
