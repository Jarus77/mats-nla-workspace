"""Step 0: download the neuronpedia n1000 J-lens + datasets, inspect/compare the two J-lenses.

    cd experiments/02_nonj_features && modal run s0_setup.py
"""

import modal

from nonj_common import HF_SECRETS, LENS_FILE_CB, LENS_FILE_NP, LENS_REPO_CB, LENS_REPO_NP, OUT, VOLUMES, net_image

app = modal.App("nonj-s0-setup")


@app.function(image=net_image, volumes=VOLUMES, secrets=HF_SECRETS, timeout=3600, memory=65536, cpu=8)
def download() -> dict:
    import os

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    os.makedirs(f"{OUT}/lenses", exist_ok=True)
    p = hf_hub_download(LENS_REPO_NP, LENS_FILE_NP, local_dir=f"{OUT}/lenses/np")
    info = {"np_lens": p, "np_size": os.path.getsize(p)}
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    wt.to_parquet(f"{OUT}/data/raw_wikitext103_train.parquet")
    ag = load_dataset("fancyzhx/ag_news", split="train")
    ag.to_parquet(f"{OUT}/data/raw_ag_news_train.parquet")
    info.update(wikitext_rows=len(wt), ag_rows=len(ag))
    VOLUMES["/results"].commit()
    print(info)
    return info


@app.function(image=net_image, volumes=VOLUMES, gpu="H100", timeout=3600, memory=131072)
def inspect() -> dict:
    import glob
    import json

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from nonj_lib import jlens_atoms, save_json

    VOLUMES["/results"].reload()
    res: dict = {}
    cb = hf_hub_download(LENS_REPO_CB, LENS_FILE_CB)
    np_ = glob.glob(f"{OUT}/lenses/np/**/*.pt", recursive=True)[0]
    lenses = {}
    for name, path in (("cb", cb), ("np", np_)):
        obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        meta = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("J", "jacobians"):
                    if isinstance(v, dict):
                        ks = sorted(v.keys(), key=lambda x: int(x))
                        meta[k] = {"type": "dict", "n": len(ks), "keys_head": [str(x) for x in ks[:3]],
                                   "keys_tail": [str(x) for x in ks[-3:]],
                                   "shape": list(v[ks[0]].shape), "dtype": str(v[ks[0]].dtype)}
                    else:
                        meta[k] = {"type": "tensor", "shape": list(v.shape), "dtype": str(v.dtype)}
                elif torch.is_tensor(v):
                    meta[k] = {"tensor": list(v.shape), "dtype": str(v.dtype)}
                else:
                    s = repr(v)
                    meta[k] = s[:800]
        res[name] = {"path": path, "meta": meta}
        lenses[name] = obj
        print(name, json.dumps(meta, indent=1)[:3000], flush=True)

    # unembedding + final norm from safetensors
    snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    cand_u = [k for k in idx if k.endswith("lm_head.weight")]
    cand_n = [k for k in idx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k]
    res["lm_head_key"], res["norm_keys"] = cand_u, cand_n
    print(cand_u, cand_n, flush=True)

    def get(key: str) -> torch.Tensor:
        with safe_open(f"{snap}/{idx[key]}", framework="pt") as f:
            return f.get_tensor(key)

    W_U = get(cand_u[0]).cuda()
    nw = get(cand_n[0]).float().cuda()
    res["norm_weight_stats"] = {"mean": nw.mean().item(), "std": nw.std().item()}
    # Qwen3.5/3.6 RMSNorm uses (1 + weight) if weights are centered near 0
    gain = 1 + nw  # Qwen3_5RMSNorm uses (1 + weight)
    res["gain_mean"] = gain.mean().item()

    def getJ(obj: dict, L: int) -> torch.Tensor:
        jac = obj["J"] if "J" in obj else obj["jacobians"]
        if isinstance(jac, dict):
            return (jac[L] if L in jac else jac[str(L)]).float().cuda()
        return jac[L].float().cuda()

    torch.manual_seed(0)
    toks = torch.randint(0, W_U.shape[0], (20000,), device="cuda")
    for L in (41, 42):
        Jc, Jn = getJ(lenses["cb"], L), getJ(lenses["np"], L)
        rel = ((Jc - Jn).norm() / Jn.norm()).item()
        cos_mat = torch.nn.functional.cosine_similarity(Jc.flatten(), Jn.flatten(), dim=0).item()
        ac = (W_U[toks].float() * gain) @ Jc
        an = (W_U[toks].float() * gain) @ Jn
        cos_atoms = torch.nn.functional.cosine_similarity(ac, an, dim=1)
        # readout agreement on random gaussian-ish residual-like vectors in the column space
        x = torch.randn(64, Jc.shape[0], device="cuda")
        sc = (x @ Jc.T * gain) @ W_U.float().T
        sn = (x @ Jn.T * gain) @ W_U.float().T
        tc, tn = sc.topk(25, dim=1).indices, sn.topk(25, dim=1).indices
        overlap = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 25 for a, b in zip(tc, tn)]).mean().item()
        res[f"L{L}"] = {
            "rel_frob_diff": rel, "matrix_cos": cos_mat,
            "atom_cos_mean": cos_atoms.mean().item(), "atom_cos_p10": cos_atoms.quantile(0.1).item(),
            "top25_overlap_random_inputs": overlap,
            "norm_cb": Jc.norm().item(), "norm_np": Jn.norm().item(),
        }
        print(L, res[f"L{L}"], flush=True)
    save_json(res, f"{OUT}/lens_inspect.json")
    VOLUMES["/results"].commit()
    return res


@app.function(image=net_image, volumes=VOLUMES, secrets=HF_SECRETS, timeout=3600, memory=16384, cpu=4)
def cache_np_lens() -> str:
    """Put the neuronpedia n1000 lens into the shared HF cache on mats-weights (lead's request)."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(LENS_REPO_NP, LENS_FILE_NP)  # HF_HOME=/weights/hf
    VOLUMES["/weights"].commit()
    return p


@app.local_entrypoint()
def main(step: str = "all") -> None:
    if step in ("all", "download"):
        print(download.remote())
    if step == "cache":
        print(cache_np_lens.remote())
    if step in ("all", "inspect"):
        print(inspect.remote())
