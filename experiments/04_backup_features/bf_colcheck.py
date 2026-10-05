"""Comparability check: apply the EMITTED-FORM functional test (Fable round 4) to 02's `col`
(chars since last newline; 02's only clean NON-J feature), using 02's own captures (read-only).

02 tested col with number-word/digit tokens (paired value test, AUROC 0.500). The emitted form of a
line-position feature is the newline token: does the J-Lens (n1000, L42 = hs[43]) put more mass on
newline tokens / more newline tokens in its top-10 as col grows? Also within-width: col / width.
Rows: 02's rule (pos >= 4, current token has no newline, col >= 1), 20k random rows.

    modal run bf_colcheck.py
"""

from __future__ import annotations

import modal

from env import OUT, VOLUMES, gpu_image

app = modal.App("mats-04-colcheck")
O2 = "/results/02_nonj_features"


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=3600, memory=131072)
def colcheck(n_rows: int = 20000) -> dict:
    import json
    import os
    import pickle

    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer

    import bf_gate_lib as G
    from bf_modal import _load_lens_and_unembed

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    items = pickle.load(open(f"{O2}/data/wrap.pkl", "rb"))
    idx = np.load(f"{O2}/acts/wrap/index.npz")
    it, ps = idx["item"], idx["pos"]
    keep = [r for r in range(len(it)) if ps[r] >= 4 and not items[it[r]]["has_nl"][ps[r]] and items[it[r]]["col"][ps[r]] >= 1]
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(keep, min(n_rows, len(keep)), replace=False))
    col = np.array([items[it[r]]["col"][ps[r]] for r in rows], float)
    width = np.array([items[it[r]]["width"] for r in rows], float)
    X = torch.from_numpy(np.load(f"{O2}/acts/wrap/hs43.npy", mmap_mode="r")[rows].astype(np.float32)).cuda()
    Js, W_U, gain, _ = _load_lens_and_unembed()
    vocab = [tok.decode([i]) for i in range(W_U.shape[0])]
    nl = [i for i, s in enumerate(vocab) if "\n" in s]
    rd = G.jlens_readout(X, Js[42], W_U, gain, [[nl[0]], [nl[0]]], ([], nl))
    out: dict = {"n_rows": int(len(rows)), "n_newline_tokens": len(nl)}
    frac = col / width
    for key in ("soft", "top10"):
        sc = rd[key]
        null = [spearmanr(sc, rng.permutation(col)).statistic for _ in range(200)]
        out[key] = {
            "spearman_col": float(spearmanr(sc, col).statistic),
            "spearman_col_over_width": float(spearmanr(sc, frac).statistic),
            "null_band95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
            "auroc_col_high_vs_low": G.emitted_stat(sc[(col >= 45) | (col <= 25)], (col[(col >= 45) | (col <= 25)] >= 45).astype(int)),
            "auroc_frac_high_vs_low": G.emitted_stat(sc[(frac >= 0.8) | (frac <= 0.4)], (frac[(frac >= 0.8) | (frac <= 0.4)] >= 0.8).astype(int)),
            "top10_rate_any_newline": float((rd["top10"] > 0).mean()) if key == "top10" else None,
        }
    print(json.dumps(out, indent=1), flush=True)
    os.makedirs(f"{OUT}/colcheck", exist_ok=True)
    json.dump(out, open(f"{OUT}/colcheck/col_emitted.json", "w"), indent=1)
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main() -> None:
    print(colcheck.remote())
