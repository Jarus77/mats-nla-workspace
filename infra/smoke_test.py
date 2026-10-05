"""GPU smoke test: load Qwen/Qwen3.6-27B (bf16) from the mats-weights volume, run one chat prompt
with thinking disabled, capture the layer-42 residual (output of decoder block 42), generate 20
tokens, and report peak GPU memory.

    modal run infra/smoke_test.py
    modal run infra/smoke_test.py --gpu H200   # override GPU type

Run only after `download_weights.py` has finished the `qwen` repo (HF_HUB_OFFLINE=1 in the image).
"""

import modal

from modal_common import BASE_MODEL, NLA_LAYER, RESULTS_DIR, VOLUMES, gpu_image

app = modal.App("mats-smoke-test")

PROMPT = "In one sentence, what is the capital of France and why is it famous?"
MAX_NEW_TOKENS = 20


def _decoder_blocks(model):  # noqa: ANN001, ANN202
    for path in ("model.layers", "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("decoder blocks not found")


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=60 * 60, memory=131072)
def smoke(layer: int = NLA_LAYER) -> dict:
    import importlib.util
    import json
    import time

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info: dict = {
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "fla_installed": importlib.util.find_spec("fla") is not None,
        "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(info, flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    info["load_s"] = round(time.time() - t0, 1)
    info["model_type"] = model.config.model_type
    info["model_class"] = type(model).__name__
    blocks = _decoder_blocks(model)
    info["n_layers"] = len(blocks)
    print(f"loaded {info['model_class']} ({info['model_type']}), {len(blocks)} blocks in {info['load_s']}s", flush=True)

    messages = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = tok(text, return_tensors="pt").to("cuda")
    info["prompt_tokens"] = int(enc.input_ids.shape[1])

    captured: dict = {}

    def hook(_mod, _inp, out):  # noqa: ANN001, ANN202
        hs = out[0] if isinstance(out, tuple) else out
        captured.setdefault("h", hs.detach().float().cpu())

    handle = blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        handle.remove()

    h = captured["h"][0]  # [seq, d_model]
    info["resid_shape"] = list(h.shape)
    info["resid_last_token_norm"] = round(h[-1].norm().item(), 3)
    info["resid_mean_token_norm"] = round(h.norm(dim=-1).mean().item(), 3)
    print(f"layer {layer} residual {tuple(h.shape)} last-token norm {info['resid_last_token_norm']}", flush=True)

    t1 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    info["gen_s"] = round(time.time() - t1, 2)
    info["generation"] = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    info["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps(info, indent=2), flush=True)

    with open(f"{RESULTS_DIR}/smoke_test.json", "w") as f:
        json.dump(info, f, indent=2)
    VOLUMES[RESULTS_DIR].commit()
    return info


@app.local_entrypoint()
def main(gpu: str = "") -> None:
    fn = smoke.with_options(gpu=gpu) if gpu else smoke
    print(fn.remote())
