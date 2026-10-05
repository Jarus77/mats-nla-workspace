"""TextNLA with base Qwen3.6-27B (no activation access) on Modal.

Arms (each a chat prompt, enable_thinking=False; sampling = the readers' own: T=1, top_p 0.95,
top_k 64, 256 new tokens; k=3 samples per prompt, sample 0 is the k-matched primary):
  qwen_prompt_nla   NLA warm-start instruction (av_base/nla_meta.yaml) on the item prompt
  qwen_prompt_wsb   wsbench prompt-only floor template on the item prompt
  qwen_resolve_nla  re-solving ceiling: RESOLVE_TEMPLATE on the NLA@42 readout
  qwen_resolve_ol42 re-solving ceiling: RESOLVE_TEMPLATE on the OL@42 readout

    cd experiments/06_resolving_decomposition
    modal run qwen_modal.py::instruction      # writes data/nla_instruction.txt
    python build_jobs.py                      # writes data/jobs.json
    modal run qwen_modal.py                   # -> data/qwen_gen.json
"""
import json
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, app_image

app = modal.App("mats-06-resolving-decomposition")
HERE = Path(__file__).resolve().parent
K = 3
TOKEN_BUDGET = 40000  # prompt tokens x K per generate() call (32 prompts OOMed an H100)
MAX_PROMPTS = 32
N_SHARDS = 6


@app.function(image=app_image, volumes=VOLUMES, cpu=2, timeout=600)
def get_instruction() -> str:
    from mats.textnla import nla_instruction
    return nla_instruction()


def _strip(t: str) -> str:
    import re
    m = re.search(r"<analysis>(.*?)(</analysis>|$)", t, re.DOTALL)
    return m.group(1).strip() if m else t.strip()


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=3 * 3600, memory=65536)
def generate(jobs: list[dict], shard: int, check: bool = False) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(1000 + shard)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    t_load = time.time() - t0

    def chat(user: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def gen(prompts: list[str], k: int, greedy: bool = False, max_new: int = 256) -> list[list[str]]:
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0,
                                                       top_p=0.95, top_k=64)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, num_return_sequences=k,
                                 pad_token_id=tok.pad_token_id, **kw)
        dec = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return [dec[i * k:(i + 1) * k] for i in range(len(prompts))]

    res: dict = {"shard": shard, "t_load": t_load}
    if check:  # left-padding sanity: greedy batched (mixed lengths) vs one prompt at a time
        ps = [chat(j["user"]) for j in sorted(jobs, key=lambda j: len(j["user"]))[::max(1, len(jobs) // 4)][:4]]
        batched = [r[0] for r in gen(ps, 1, greedy=True, max_new=48)]
        single = [gen([p], 1, greedy=True, max_new=48)[0][0] for p in ps]
        res["pad_check"] = [{"batched": b, "single": s, "same": b == s}
                            for b, s in zip(batched, single)]
        print("pad_check same:", [x["same"] for x in res["pad_check"]], flush=True)

    chats = [chat(j["user"]) for j in jobs]
    ntok = [len(tok(c, add_special_tokens=False).input_ids) for c in chats]
    order = sorted(range(len(jobs)), key=lambda i: ntok[i])
    batches, cur = [], []
    for i in order:  # sorted ascending, so the batch max is the newest prompt
        if cur and ((len(cur) + 1) * ntok[i] * K > TOKEN_BUDGET or len(cur) >= MAX_PROMPTS):
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    outs: dict[str, dict] = {}
    done = 0
    for idx in batches:
        samples = gen([chats[i] for i in idx], K)
        for i, smp in zip(idx, samples):
            outs[jobs[i]["key"]] = {"raw": smp, "samples": [_strip(x) for x in smp]}
        done += len(idx)
        print(f"[shard {shard}] {done}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    res.update(outs=outs, t_total=time.time() - t0)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/qwen_gen_shard{shard}.json").write_text(json.dumps(res, ensure_ascii=False))
    VOLUMES["/results"].commit()
    return res


@app.local_entrypoint()
def instruction():
    txt = get_instruction.remote()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "nla_instruction.txt").write_text(txt, encoding="utf-8")
    print(txt)


@app.local_entrypoint()
def main(limit: int = 0, n_shards: int = N_SHARDS):
    jobs = json.loads((HERE / "data" / "jobs_qwen.json").read_text(encoding="utf-8"))
    if limit:
        jobs = jobs[:limit]
    shards = [jobs[i::n_shards] for i in range(n_shards)]
    merged: dict = {"outs": {}, "meta": []}
    for r in generate.map(shards, range(n_shards), [True] + [False] * (n_shards - 1)):
        merged["outs"].update(r.pop("outs"))
        merged["meta"].append(r)
    out = HERE / "data" / ("qwen_gen.json" if not limit else "qwen_gen_smoke.json")
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=0), encoding="utf-8")
    print("wrote", out, len(merged["outs"]), [m.get("t_total") for m in merged["meta"]])
