"""TextNLA: a text-only "re-description" baseline for the NLA.

Qwen3.6-27B (base, no activation access) is given each example's source text with the exact
instruction prompt Sonnet used to write the NLA warm-start explanations (av_base/nla_meta.yaml
api_summaries.instruction_prompt). Its <analysis> becomes the explanation fed to the NLA AR in
score_gate.py. This is the "prompt re-solving" ceiling for FVE: everything recoverable from the
text alone, in the NLA's own register. k=4 samples, T=1.0, enable_thinking=False.
"""
import json
import re
from pathlib import Path

import modal
from env import OUT, VOLUMES, app_image, load_examples

app = modal.App("mats-00-textnla")
K = 4


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600)
def run() -> dict:
    import torch
    import yaml
    from mats.readers.nla import snapshot_dir
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    _, texts, _ = load_examples()
    meta = yaml.safe_load((snapshot_dir() / "av_base" / "nla_meta.yaml").read_text())
    instr = meta["api_summaries"]["instruction_prompt"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    prompts = [tok.apply_chat_template([{"role": "user", "content": instr.format(text=t)}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for t in texts]
    raw = []
    B = 4
    for s in range(0, len(prompts), B):
        enc = tok(prompts[s:s + B], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=300, num_return_sequences=K,
                                 pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        raw += [dec[i * K:(i + 1) * K] for i in range(len(dec) // K)]
        print(s, repr(dec[0][:300]), flush=True)

    def grab(t):
        m = re.search(r"<analysis>(.*?)(</analysis>|$)", t, re.DOTALL)
        return m.group(1).strip() if m else t.strip()
    res = {"raw": raw, "explanations": [[grab(t) for t in row] for row in raw],
           "n_closed": sum("</analysis>" in t for row in raw for t in row)}
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/textnla.json").write_text(json.dumps(res, indent=1))
    VOLUMES["/results"].commit()
    return res


@app.local_entrypoint()
def main():
    res = run.remote()
    (Path(__file__).parent / "results" / "textnla.json").write_text(json.dumps(res, indent=1))
    print("closed", res["n_closed"])
