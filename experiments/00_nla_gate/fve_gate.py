"""Task 2: NLA FVE reproduction + first prompt-re-description controls (H200, AV + AR).

Conditions (k samples each over the 64 example activations, baseline = predict-the-mean of the
64 rescaled gold vectors, EasyNLA paper definition):
  iter_000400 / iter_000300 / av_base(no adapter): enable_thinking=False, training-eval sampling
  iter_000400 enable_thinking=True (EasyNLA build_prompt_text default leaves '<think>\\n' open)
  iter_000400 wsbench sampling (top_p .95, top_k 64)
Controls: shuffled activation (explanation of act[perm[i]] scored against act[i]);
source-text tails as the "explanation"; empty explanation.
"""
import json
import time
from pathlib import Path

import modal
from env import OUT, VOLUMES, app_image, load_examples

app = modal.App("mats-00-fve-gate")
K = 4


def summarize(name, res, extra=None):
    import torch
    pe = res["per_example"]
    ok = pe[~torch.isnan(pe)]
    ps = res["per_sample"]
    valid = ps[~torch.isnan(ps)]
    d = {"cond": name, "fve_mean_of_samples": float(valid.mean()) if len(valid) else None,
         "fve_mean_of_examples": float(ok.mean()) if len(ok) else None,
         "fve_sd_examples": float(ok.std()) if len(ok) > 1 else None,
         "n_valid_samples": int(len(valid)), "n_failed": res["n_failed"],
         "baseline": res["baseline"], "per_example": [round(float(x), 4) for x in pe]}
    d.update(extra or {})
    print(json.dumps({k: v for k, v in d.items() if k != "per_example"}), flush=True)
    return d


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=4 * 3600)
def run() -> dict:
    import torch
    from mats.readers.nla import Sampling, fve, load_nla, predict_mean_baseline

    torch.manual_seed(0)
    acts, texts, _ = load_examples()
    t0 = time.time()
    nla = load_nla(adapter=["av_rl_adapters/iter_000400", "av_rl_adapters/iter_000300"])
    print(f"loaded in {time.time() - t0:.0f}s; prompt len {len(nla.prompt_ids)} marker {nla.marker}",
          flush=True)
    base = predict_mean_baseline(acts)
    out = {"baseline": base, "conds": [], "explanations": {}, "raw_examples": {}}

    def cond(name, sampling=Sampling(), thinking=False, adapter="iter_000400", none=False):
        nla.set_prompt(thinking)
        nla.set_adapter(adapter)
        t = time.time()
        if none:
            with nla.no_adapter():
                raw = nla.verbalize_raw(acts, K, sampling)
        else:
            raw = nla.verbalize_raw(acts, K, sampling)
        from mats.readers.nla import extract_explanation
        expl = [[extract_explanation(x) for x in row] for row in raw]
        res = nla.fve_on(acts, baseline=base, explanations=expl)
        lens = [len(nla.tok.encode(x)) for row in raw for x in row]
        d = summarize(name, res, {"gen_s": round(time.time() - t), "mean_resp_tokens":
                                  sum(lens) / len(lens)})
        out["conds"].append(d)
        out["explanations"][name] = expl
        out["raw_examples"][name] = [raw[i][0] for i in range(4)]
        return expl, res

    e400, r400 = cond("iter400_nothink")
    cond("iter300_nothink", adapter="iter_000300")
    cond("avbase_nothink", none=True)
    cond("iter400_think", thinking=True)
    cond("iter400_nothink_wsbench_sampling", sampling=Sampling(top_p=0.95, top_k=64))
    nla.set_prompt(False)

    # shuffled activation: explanations generated from act[perm[i]], scored against act[i]
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(64, generator=g)
    while (perm == torch.arange(64)).any():
        perm = torch.randperm(64, generator=g)
    shuf = [e400[int(perm[i])] for i in range(64)]
    out["conds"].append(summarize("iter400_shuffled_act",
                                  nla.fve_on(acts, baseline=base, explanations=shuf)))

    # text-only baselines: the source text itself (its tail) as the explanation
    def tail(t, n_tok):
        ids = nla.tok.encode(t, add_special_tokens=False)
        return nla.tok.decode(ids[-n_tok:])
    for n in (64, 256, 900):
        ex = [[tail(t, n)] for t in texts]
        out["conds"].append(summarize(f"source_tail_{n}tok",
                                      nla.fve_on(acts, baseline=base, explanations=ex)))
    shuf_src = [[tail(texts[int(perm[i])], 256)] for i in range(64)]
    out["conds"].append(summarize("source_tail_256tok_shuffled",
                                  nla.fve_on(acts, baseline=base, explanations=shuf_src)))
    # constant predictions
    p_empty = nla.reconstruct([""] * 1).expand(64, -1)
    out["conds"].append({"cond": "empty_explanation",
                         "fve_mean_of_samples": float(fve(p_empty, acts, base).mean())})
    out["conds"].append({"cond": "oracle_mean_vector (≈0 by construction)",
                         "fve_mean_of_samples": float(fve(acts.mean(0, keepdim=True).expand(64, -1),
                                                          acts, base).mean())})
    out["perm"] = perm.tolist()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    with open(f"{OUT}/fve_gate.json", "w") as f:
        json.dump(out, f, indent=1)
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main():
    res = run.remote()
    p = Path(__file__).parent / "results"
    (p / "fve_gate.json").write_text(json.dumps(res, indent=1))
    for c in res["conds"]:
        print({k: v for k, v in c.items() if k != "per_example"})
