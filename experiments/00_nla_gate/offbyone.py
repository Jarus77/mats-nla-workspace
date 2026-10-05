"""What does the layer convention cost? NLA FVE (iter_000400, k=4) on our recomputed vectors:
block-42 output (correct, = hidden_states[43]) vs block-41 output (hidden_states[42]).
Each is scored against itself with its own predict-the-mean baseline, and also against gold."""
import json
from pathlib import Path

import modal
from env import OUT, VOLUMES, app_image

app = modal.App("mats-00-offbyone")


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=2 * 3600)
def run() -> dict:
    import torch
    from mats.readers.nla import fve, load_nla, predict_mean_baseline

    torch.manual_seed(0)
    rec = torch.load(f"{OUT}/recomputed.pt")
    gold = rec["gold"]
    nla = load_nla(adapter="av_rl_adapters/iter_000400")
    gbase = predict_mean_baseline(gold)
    out = {}
    for name in ("hs43", "hs42"):
        acts = rec[name].float()
        r = nla.fve_on(acts, k=4)
        flat = [(i, e) for i, row in enumerate(r["explanations"]) for e in row if e]
        pred = nla.reconstruct([e for _, e in flat])
        vs_gold = fve(pred, gold[[i for i, _ in flat]], gbase)
        out[name] = {"fve_self": float(torch.nanmean(r["per_sample"])), "self_baseline": r["baseline"],
                     "fve_vs_gold": float(vs_gold.mean()), "n_failed": r["n_failed"]}
        print(name, out[name], flush=True)
    Path(f"{OUT}/offbyone.json").write_text(json.dumps(out, indent=1))
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main():
    res = run.remote()
    (Path(__file__).parent / "results" / "offbyone.json").write_text(json.dumps(res, indent=1))
