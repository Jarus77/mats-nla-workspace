"""Modal images for 04_backup_features: the same pins as experiments/03_pipeline_smoke/env.py
(infra/modal_common.py pins), plus scipy / datasets / pandas for the data stage. src/mats and this
directory's helpers are shipped in (src/mats is imported, never edited)."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/04_backup_features"
LOCAL_MODULES = ("modal_common", "env", "bf_items", "bf_feats", "bf_gate_lib", "bf_vocab", "bf_modal")


def _image(offline: bool) -> modal.Image:
    return (
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
            "flash-linear-attention==0.5.2",
            *HF_PINS,
        )
        .uv_pip_install("datasets", "pandas", "pyarrow", "spacy==3.8.7")
        .run_commands("uv pip install --system https://github.com/explosion/spacy-models/releases/download/"
                      "fr_core_news_md-3.8.0/fr_core_news_md-3.8.0-py3-none-any.whl")
        .env({**HF_ENV, "HF_HUB_OFFLINE": "1" if offline else "0", "TOKENIZERS_PARALLELISM": "false",
              "PYTHONPATH": "/root/src"})
        .add_local_python_source(*LOCAL_MODULES)
        .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
    )


gpu_image = _image(offline=True)
net_image = _image(offline=False)
