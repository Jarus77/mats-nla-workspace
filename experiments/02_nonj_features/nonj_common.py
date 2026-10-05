"""Shared Modal resources for experiments/02_nonj_features (extends infra/modal_common.py).

Layer convention used everywhere in this directory:
  ``hs[i]`` = HF ``output_hidden_states[i]`` = output of decoder block ``i-1``.
  WorkspaceBench / J-lens "layer L" = output of block L = ``hs[L+1]``.
We capture hs[40..45] (= block outputs 39..44) so that both candidate NLA conventions
(hs[42] = block 41, hs[43] = block 42) and Oracle Lens L40/L44 (block outputs 40/44 = hs[41]/hs[45])
are covered.
"""

import sys
from pathlib import Path

_PARENTS = Path(__file__).resolve().parents
if len(_PARENTS) > 2 and (_PARENTS[2] / "infra").is_dir() and str(_PARENTS[2] / "infra") not in sys.path:
    sys.path.insert(0, str(_PARENTS[2] / "infra"))

import modal  # noqa: E402

from modal_common import (  # noqa: E402,F401
    BASE_MODEL,
    HF_ENV,
    HF_PINS,
    HF_SECRETS,
    RESULTS_DIR,
    VOLUMES,
    gpu_image,
)

OUT = f"{RESULTS_DIR}/02_nonj_features"
HS_LAYERS = [40, 41, 42, 43, 44, 45]  # hidden_states indices captured
MAIN_HS = 43  # = output of block 42 = WorkspaceBench L42 (J-lens index 42)
ALT_HS = 42  # = output of block 41 (HF hidden_states[42])

LENS_REPO_CB = "camilablank/workspace-lenses"
LENS_FILE_CB = "qwen3.6-27b/j-lens/lens.pt"
LENS_REPO_NP = "neuronpedia/jacobian-lens"
LENS_FILE_NP = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"

def _image(offline: bool) -> modal.Image:
    """GPU image with the same pins as modal_common.gpu_image, plus spaCy/sklearn/datasets."""
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
            "flash-linear-attention==0.5.2",
            *HF_PINS,
        )
        .uv_pip_install("spacy==3.8.7", "scikit-learn", "datasets", "pandas", "pyarrow", "scipy")
        .run_commands(
            "uv pip install --system https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
        )
        .env({**HF_ENV, "HF_HUB_OFFLINE": "1" if offline else "0", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_python_source("modal_common", "nonj_common", "nonj_lib", "features", "s3_probe")
    )


nonj_image = _image(offline=True)
net_image = _image(offline=False)  # network access for HF dataset / lens downloads
