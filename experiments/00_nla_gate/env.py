"""Modal image for 00_nla_gate: infra's gpu_image pins + pyarrow, with src/mats shipped in.

infra/modal_common.gpu_image ends with add_local_python_source, so it cannot be extended;
this rebuilds the same pins. Import as `from env import app_image, VOLUMES` from scripts in
this directory (run from infra/ or anywhere; paths resolve relative to this file).
"""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

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
        "pyarrow",
        "flash-linear-attention==0.5.2",
        *HF_PINS,
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/root/src"})
    .add_local_python_source("modal_common", "env")
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats",
                   ignore=["**/__pycache__/**"])
)

NLA_PARQUET = ("/weights/hf/hub/models--ceselder--qwen3.6-27b-nla-rl/snapshots/"
               "5a13b7ec21a69fcdd0fb24d5edbe96a92aef4b9f/data/example_activations.parquet")
OUT = "/results/00_nla_gate"


def load_examples():
    """(acts [64, 5120] fp32, texts, meta rows) from the NLA repo's example parquet."""
    import pyarrow.parquet as pq
    import torch
    t = pq.read_table(NLA_PARQUET).to_pylist()
    acts = torch.tensor([r["activation_vector"] for r in t], dtype=torch.float32)
    return acts, [r["detokenized_text_truncated"] for r in t], t
