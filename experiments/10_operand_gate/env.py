"""Modal image for 10_operand_gate: the 09 image recipe, src/mats, this directory's helpers, a read-only copy
of 04's bf_gate_lib (probes, enrichment, J-Lens readout, permutation nulls) and the 600-prompt data file."""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if (_REPO / "infra").is_dir():
    sys.path.insert(0, str(_REPO / "infra"))
from modal_common import HF_ENV, HF_PINS, VOLUMES  # noqa: E402,F401

OUT = "/results/10_operand_gate"
BASIS_L42 = "/results/03_pipeline_smoke/jspace_basis_L42.pt"
DATA_REMOTE = "/root/data/arithmetic.jsonl"
_E04 = _REPO / "experiments" / "04_backup_features"
_LOCAL = ("modal_common", "env", "o10_lib", "o10s2_lib", "o10g4_lib")

app_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0", "safetensors==0.8.0",
        "numpy==2.5.3", "pyyaml", "scipy", "flash-linear-attention==0.5.2", *HF_PINS,
    )
    .env({**HF_ENV, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src:/root/ext04"})
    .add_local_python_source(*_LOCAL)
    .add_local_dir(str(_REPO / "src" / "mats"), "/root/src/mats", ignore=["**/__pycache__/**"])
    .add_local_file(str(_E04 / "bf_gate_lib.py"), "/root/ext04/bf_gate_lib.py")
    .add_local_file(str(_HERE / "data" / "arithmetic.jsonl"), DATA_REMOTE)
)
