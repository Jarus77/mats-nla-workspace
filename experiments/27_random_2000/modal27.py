"""Exp 27 (app `f3qn`): exp 23's two stages on the 2,000-point bank, both readers once each, plus the layer-42 activation.
  u8yd (H100): capture layer 42 at the read token for all items; Oracle Lens free readout k=1 -> acts27.pt, ol27.json
  c2hk (H200): NLA raw readout k=1 -> nla27.json
Each stage skips itself when its output is already on the volume and commits after writing.
    cd experiments/27_random_2000 && modal run --detach modal27.py --stage ol
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, gpu_image

app = modal.App("f3qn")
LAYER = 42
GPU = dict(volumes=VOLUMES, timeout=3 * 3600, memory=131072, single_use_containers=True, max_containers=1)


def _items() -> list[dict]:
    return json.load(open(f"{OUT}/items_base.json"))


@app.function(image=gpu_image, gpu="H100", **GPU)
def u8yd() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mats import capture as C
    from mats.readers.olens import WSBENCH_SAMPLING, load_olens

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    if Path(f"{OUT}/ol27.json").exists() and Path(f"{OUT}/acts27.pt").exists():
        return {"skipped": "ol27.json and acts27.pt already on the volume"}
    items = _items()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    ol = load_olens(model, tok)
    seqs = [it["ids"] for it in items]
    with ol.base():
        acts = C.capture(model, seqs, [LAYER], [[-1]] * len(seqs), batch_size=8)[LAYER][:, 0].contiguous().cpu()
    torch.save({"ids": [it["id"] for it in items], "acts": acts}, f"{OUT}/acts27.pt")
    g = ol.verbalize(acts.float(), LAYER, k=1, sampling=WSBENCH_SAMPLING, batch=32)
    Path(f"{OUT}/ol27.json").write_text(json.dumps({"ol": {it["id"]: g[r][0] for r, it in enumerate(items)}}))
    VOLUMES["/results"].commit()
    return {"n": len(items), "wall_s": round(time.time() - t0, 1)}


@app.function(image=gpu_image, gpu="H200", **GPU)
def c2hk() -> dict:
    import torch

    from mats.readers import nla as NL

    torch.manual_seed(0)
    t0 = time.time()
    VOLUMES["/results"].reload()
    if Path(f"{OUT}/nla27.json").exists():
        return {"skipped": "nla27.json already on the volume"}
    items = _items()
    blob = torch.load(f"{OUT}/acts27.pt", weights_only=False)
    assert blob["ids"] == [it["id"] for it in items]
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    raw = nla.verbalize_raw(blob["acts"].float(), k=1, batch=64)
    Path(f"{OUT}/nla27.json").write_text(json.dumps({"nla": {it["id"]: raw[r][0] for r, it in enumerate(items)}}))
    VOLUMES["/results"].commit()
    return {"n": len(items), "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(stage: str = "ol") -> None:
    r = {"ol": u8yd, "nla": c2hk}[stage].remote()
    print(stage, json.dumps(r))
