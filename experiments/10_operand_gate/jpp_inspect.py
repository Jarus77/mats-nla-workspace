"""Ad-hoc: print the top-level structure of the J++ lens file on rfm-weights (CPU, mmap)."""
import glob

import modal

from env import VOLUMES, app_image

app = modal.App("rfm-inspect")


@app.function(image=app_image, volumes=VOLUMES, timeout=900, memory=32768, cpu=4)
def inspect() -> dict:
    import torch

    p = glob.glob("/weights/hf/hub/models--koayon--jpp-lenses/snapshots/*/qwen3.6-27b/lens.pt")
    if not p:
        return {"error": "no file"}
    try:
        obj = torch.load(p[0], map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:  # noqa: BLE001
        return {"path": p[0], "load_error": repr(e)[:500]}

    def desc(x, depth=0):  # noqa: ANN001, ANN202
        if torch.is_tensor(x):
            return f"tensor{tuple(x.shape)} {x.dtype}"
        if isinstance(x, dict):
            ks = list(x.keys())
            return {str(k): (desc(x[k], depth + 1) if depth < 2 else type(x[k]).__name__) for k in ks[:12]} | ({"...": f"{len(ks)} keys"} if len(ks) > 12 else {})
        if isinstance(x, (list, tuple)):
            return f"{type(x).__name__}[{len(x)}]: " + (desc(x[0], depth + 1) if x and depth < 2 else "")
        return f"{type(x).__name__}: {str(x)[:200]}"

    return {"path": p[0], "type": type(obj).__name__, "structure": desc(obj)}


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(inspect.remote(), indent=1, default=str))
