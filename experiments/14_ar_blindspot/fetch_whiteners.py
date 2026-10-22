"""Download the two OLens-AR whitener files (layers 40, 44) into the HF cache on the weights volume."""
import modal

from env import VOLUMES
from modal_common import HF_ENV, HF_PINS, HF_SECRETS

app = modal.App("kq2-dl")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*HF_PINS).env({**HF_ENV, "HF_HUB_OFFLINE": "0"}).add_local_python_source("modal_common", "env")
FILES = [f"ckpts/ar/ptag/whiteners/whitening_iolens_pooled_L{L}.safetensors" for L in (40, 44)] + ["ckpts/ar/ptag/whiteners/whitening_iolens_pooled_provenance.json"]


@app.function(image=image, volumes=VOLUMES, timeout=1800, secrets=HF_SECRETS)
def fetch() -> list[str]:
    from huggingface_hub import hf_hub_download

    out = [hf_hub_download("agu18dec/local-workspace", f, repo_type="dataset") for f in FILES]
    VOLUMES["/weights"].commit()
    return out


@app.local_entrypoint()
def main() -> None:
    print("\n".join(fetch.remote()))
