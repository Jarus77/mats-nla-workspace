"""Prompt-only floor verification: re-generate the frozen wsbench prompt-only template
(evals/baselines/prompt_only.json _source) with base Qwen3.6-27B, k=1, under three renderings:
  think_off  chat template, enable_thinking=False   (what T_wsb used)
  think_on   chat template, enable_thinking=True    (Qwen default)
  raw        no chat template (plain completion of the template text)
Same seed (0) per variant; sampling T=1, top_p 0.95, top_k 64, 256 new tokens, skip_special_tokens.
Raw decoded text is kept (the regex scorer strips <think> tags but keeps their content).

    PYTHONPATH=../../infra modal run --detach floor_modal.py   # -> data/floor_gen.json
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, app_image

app = modal.App("mats-06-floor-verify")
HERE = Path(__file__).resolve().parent
TOKEN_BUDGET = 24000
MAX_PROMPTS = 48
VARIANTS = ("think_off", "think_on", "raw")


_MODEL = {}


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=2 * 3600, memory=65536)
def generate_all(jobs: list[dict]) -> dict:
    return {v: generate(jobs, v) for v in VARIANTS}


def generate(jobs: list[dict], variant: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    if not _MODEL:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
        tok.padding_side = "left"
        _MODEL["tok"], _MODEL["model"] = tok, AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    tok, model = _MODEL["tok"], _MODEL["model"]

    def render(user: str) -> str:
        if variant == "raw":
            return user
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=(variant == "think_on"))

    texts = [render(j["user"]) for j in jobs]
    ntok = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    order = sorted(range(len(jobs)), key=lambda i: ntok[i])
    batches, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * ntok[i] > TOKEN_BUDGET or len(cur) >= MAX_PROMPTS):
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    torch.manual_seed(0)
    outs = {}
    for idx in batches:
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=True, temperature=1.0,
                                 top_p=0.95, top_k=64, pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for i, d, g in zip(idx, dec, gen):
            outs[jobs[i]["key"]] = {"text": d, "n_new": int((g != tok.pad_token_id).sum())}
        print(f"[{variant}] {len(outs)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    res = {"variant": variant, "outs": outs, "t_total": time.time() - t0,
           "example_render": texts[0][-400:]}
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/floor_{variant}.json").write_text(json.dumps(res, ensure_ascii=False))
    VOLUMES["/results"].commit()
    return res


@app.local_entrypoint()
def main():
    jobs = [j for j in json.loads((HERE / "data" / "jobs.json").read_text(encoding="utf-8"))
            if j["arm"] == "prompt_wsb"]
    merged = generate_all.remote(jobs)
    (HERE / "data" / "floor_gen.json").write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print({v: round(r["t_total"]) for v, r in merged.items()})
