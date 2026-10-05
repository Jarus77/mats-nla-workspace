"""Task 1: which residual-stream convention are the NLA example activations in?

Recompute the 64 example activations with Qwen3.6-27B and compare against
 (a) HF output_hidden_states[42] = output of decoder block 41,
 (b) output_hidden_states[43] = output of decoder block 42 (WorkspaceBench "layer 42"),
also via explicit forward hooks on blocks 41/42 (the wsbench Backend.capture path), and at
neighbouring token positions. Saves recomputed vectors to /results/00_nla_gate/recomputed.pt.
"""
import json

import modal
from env import OUT, VOLUMES, app_image, load_examples

app = modal.App("mats-00-layer-check")


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3600)
def run() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    acts, texts, rows = load_examples()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    blocks = model.model.layers
    hooked = {}

    def mk(i):
        def h(_m, _i, out):
            hooked[i] = (out[0] if isinstance(out, tuple) else out)[0].float().cpu()
        return h

    handles = [blocks[i].register_forward_hook(mk(i)) for i in (40, 41, 42, 43)]

    def cmp(x, y):
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        rel = (x - y).norm(dim=-1) / y.norm(dim=-1)
        return cos, rel

    per = []
    store = {k: [] for k in ("hs42", "hs43", "hook41", "hook42")}
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=True)["input_ids"]
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            out = model(x, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hs = [h[0].float().cpu() for h in out.hidden_states]
        gold = acts[i]
        rec = {"i": i, "n_tok": len(ids), "n_raw_tokens": rows[i]["n_raw_tokens"],
               "gold_norm": float(gold.norm())}
        for name, mat in (("hs41", hs[41]), ("hs42", hs[42]), ("hs43", hs[43]),
                          ("hs44", hs[44]), ("hook41", hooked[41]), ("hook42", hooked[42])):
            c, r = cmp(mat[-1], gold)
            rec[f"{name}_cos"], rec[f"{name}_rel"] = float(c), float(r)
            rec[f"{name}_norm"] = float(mat[-1].norm())
        # best-matching position anywhere in the sequence, for the two candidate conventions
        for name, mat in (("hs42", hs[42]), ("hs43", hs[43])):
            c, _ = cmp(mat, gold.expand_as(mat))
            rec[f"{name}_bestpos_offset"] = int(c.argmax()) - (len(ids) - 1)
            rec[f"{name}_bestpos_cos"] = float(c.max())
        rec["hook42_eq_hs43"] = float((hooked[42][-1] - hs[43][-1]).abs().max())
        rec["hook41_eq_hs42"] = float((hooked[41][-1] - hs[42][-1]).abs().max())
        for k, mat in (("hs42", hs[42]), ("hs43", hs[43]), ("hook41", hooked[41]),
                       ("hook42", hooked[42])):
            store[k].append(mat[-1])
        per.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}),
              flush=True)
    for h in handles:
        h.remove()

    def agg(key):
        v = torch.tensor([r[key] for r in per])
        return {"mean": float(v.mean()), "median": float(v.median()), "min": float(v.min()),
                "max": float(v.max())}

    summary = {k: agg(k) for k in per[0] if k.endswith(("_cos", "_rel", "_norm"))}
    summary["n"] = len(per)
    summary["n_tok_equals_n_raw_tokens"] = sum(r["n_tok"] == r["n_raw_tokens"] for r in per)
    summary["hs43_bestpos_offset_counts"] = {
        str(o): sum(r["hs43_bestpos_offset"] == o for r in per)
        for o in sorted({r["hs43_bestpos_offset"] for r in per})}
    summary["hs42_bestpos_offset_counts"] = {
        str(o): sum(r["hs42_bestpos_offset"] == o for r in per)
        for o in sorted({r["hs42_bestpos_offset"] for r in per})}
    torch.save({k: torch.stack(v) for k, v in store.items()} | {"gold": acts},
               f"{OUT}/recomputed.pt")
    res = {"summary": summary, "per_example": per}
    with open(f"{OUT}/layer_check.json", "w") as f:
        json.dump(res, f, indent=1)
    VOLUMES["/results"].commit()
    return res


@app.local_entrypoint()
def main():
    res = run.remote()
    out = __import__("pathlib").Path(__file__).parent / "results" / "layer_check.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
