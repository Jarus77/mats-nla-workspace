"""Modal image for 08_col_pilot: the 03/07 GPU pins (rebuilt, as 03 does), plus pandas/pyarrow for
item building, src/mats, this directory's helpers, and read-only copies of 02's s1_data.py
(paragraph cleaner + build_wrap) and features.py (col row selection for the dose bank)."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/08_col_pilot"
OUT02 = "/results/02_nonj_features"
OUT03 = "/results/03_pipeline_smoke"
OUT07 = "/results/07_template_mining"
_E02 = _REPO / "experiments" / "02_nonj_features"

app_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "peft==0.21.0",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "numpy==2.5.3",
        "pyyaml",
        "scipy",
        "pandas",
        "pyarrow",
        "flash-linear-attention==0.5.2",
        *HF_PINS,
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/root/src"})
    .add_local_python_source("modal_common", "env", "pilot_items", "pilot_lib", "templates")
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
    .add_local_file(str(_E02 / "s1_data.py"), "/root/ext02/s1_data.py")
    .add_local_file(str(_E02 / "features.py"), "/root/ext02/features.py")
)
