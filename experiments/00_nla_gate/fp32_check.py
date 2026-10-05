"""Residual mismatch on the worst layer_check examples: is it bf16 noise? Capture block-42
output in fp32 (H200) for the 6 worst + 4 typical examples and compare to the parquet gold."""
import json
from pathlib import Path

import modal
from env import OUT, VOLUMES, app_image, load_examples

app = modal.App("mats-00-fp32-check")
IDX = [14, 10, 29, 23, 15, 27, 0, 1, 2, 3]


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3600)
def run() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    acts, texts, _ = load_examples()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.float32, device_map="cuda").eval()
    got = {}
    model.model.layers[42].register_forward_hook(
        lambda m, i, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
    out = []
    for i in IDX:
        ids = tok(texts[i], add_special_tokens=True, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            model(ids, use_cache=False, logits_to_keep=1)
        h, g = got["h"], acts[i]
        out.append({"i": i, "cos": float(torch.nn.functional.cosine_similarity(h, g, dim=0)),
                    "rel": float((h - g).norm() / g.norm())})
        print(out[-1], flush=True)
    Path(f"{OUT}/fp32_check.json").write_text(json.dumps(out, indent=1))
    VOLUMES["/results"].commit()
    return {"rows": out}


@app.local_entrypoint()
def main():
    res = run.remote()
    (Path(__file__).parent / "results" / "fp32_check.json").write_text(json.dumps(res, indent=1))
