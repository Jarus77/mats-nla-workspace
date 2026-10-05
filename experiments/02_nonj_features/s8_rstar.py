"""Step 8: r* = smallest rank r such that the J-lens rebuilt from the top-r J-space keeps >= 90%
top-10 readout agreement on WorkspaceBench basic_readout (100 items, final prompt token).

J-space at rank r (lead's definition, after Viswanath): span of the top-r right singular vectors
V_r of the lens map M_l = W_U diag(g) J_l (neuronpedia n1000 lens). Rebuilt lens:
lens_r(h) = softmax(W_U norm(J_l V_r V_r^T h)). Agreement = |top10(lens(h)) & top10(lens_r(h))|/10.
Computed at J-lens layers 38, 40, 41, 42, 44 (block outputs; hs index = L+1).

    modal run s8_rstar.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

from pathlib import Path

app = modal.App("nonj-s8-rstar")
_P = Path(__file__).resolve().parents
_BR = (_P[2] / "third_party/workspace-bench/evals/basic_readout/items.json") if len(_P) > 2 else Path("/nonexistent")
image = nonj_image.add_local_file(str(_BR), "/root/basic_readout_items.json") if _BR.exists() else nonj_image

RS = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
JLAYERS = [38, 40, 41, 42, 44]
ANSWER_STEM = "My one-word answer is:"


@app.function(image=image, volumes=VOLUMES, gpu="H100", timeout=2 * 3600, memory=196608)
def rstar() -> dict:
    import glob
    import json

    import numpy as np
    import torch

    from nonj_lib import decoder_blocks, final_norm_module, load_jlens_layers, load_model, norm_gain, save_json

    VOLUMES["/results"].reload()
    tok, model = load_model()
    blocks = decoder_blocks(model)
    fnorm = final_norm_module(model)
    W_U = model.lm_head.weight
    gain = norm_gain(model)
    lens_path = glob.glob(f"{OUT}/lenses/np/**/*.pt", recursive=True)[0]
    Js = {L: J.cuda() for L, J in load_jlens_layers(lens_path, JLAYERS).items()}
    items = json.load(open("/root/basic_readout_items.json"))["items"]
    cap: dict = {}
    hooks = [blocks[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].detach().float()))
             for L in JLAYERS]
    H = {L: [] for L in JLAYERS}
    for it in items:
        if it["eval_render"] == "chat":
            text = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) + ANSWER_STEM
        else:
            text = it["prompt"]
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            model.model(input_ids=ids)
        for L in JLAYERS:
            H[L].append(cap[L])
    for h in hooks:
        h.remove()
    targets = [it["target"] for it in items]
    res: dict = {"r_grid": RS, "layers": {}}
    for L in JLAYERS:
        X = torch.stack(H[L])  # [100, d]
        J = Js[L]
        Mg = torch.zeros(J.shape[0], J.shape[0], device="cuda")
        Jg = gain[:, None] * J
        for i in range(0, W_U.shape[0], 32768):
            a_ = W_U[i:i + 32768].float() @ Jg
            Mg += a_.T @ a_
        evals, evecs = torch.linalg.eigh(Mg)
        evecs = evecs.flip(1)

        def lens(Z: torch.Tensor) -> torch.Tensor:
            return (fnorm((Z @ J.T).to(torch.bfloat16)) @ W_U.T).float()

        base = lens(X)
        top_b = base.topk(10, dim=1).indices
        tgt_ids = [tok.encode(" " + t, add_special_tokens=False)[0] for t in targets]
        tgt_in_base = np.mean([tgt_ids[i] in top_b[i].tolist() for i in range(len(items))])
        rows = {}
        for r in RS:
            Q = evecs[:, :r]
            for name, Xr in (("topr", (X @ Q) @ Q.T), ("resid", X - (X @ Q) @ Q.T)):
                sc = lens(Xr)
                top = sc.topk(10, dim=1).indices
                agree = [len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(top, top_b)]
                rows.setdefault(r, {})[name] = {"agree_mean": float(np.mean(agree)),
                                                "agree_by_subfamily": {sf: float(np.mean([a for a, it in zip(agree, items) if it["subfamily"] == sf]))
                                                                       for sf in ("entity", "processing", "implicit")},
                                                "target_first_tok_in_top10": float(np.mean([tgt_ids[i] in top[i].tolist() for i in range(len(items))])),
                                                "energy_frac_of_h": float(((Xr ** 2).sum(1) / (X ** 2).sum(1)).mean())}
        rstar_ = next((r for r in RS if rows[r]["topr"]["agree_mean"] >= 0.9), None)
        res["layers"][L] = {"hs_index": L + 1, "rows": rows, "r_star": rstar_,
                            "target_first_tok_in_top10_full_lens": float(tgt_in_base),
                            "spectrum_energy_frac": {r: float(evals.flip(0)[:r].sum() / evals.sum()) for r in RS}}
        print(L, "r*", rstar_, {r: round(rows[r]["topr"]["agree_mean"], 3) for r in RS}, flush=True)
        if L in (38, 42):
            torch.save({"layer_jlens": L, "hs_index": L + 1, "evecs_top4096": evecs[:, :4096].half().cpu(),
                        "evals_top4096": evals.flip(0)[:4096].cpu(),
                        "def": "top right singular vectors of W_U diag(g) J_l (neuronpedia n1000)"},
                       f"{OUT}/jspace_basis_L{L}.pt")
    save_json(res, f"{OUT}/rstar.json")
    VOLUMES["/results"].commit()
    return {L: res["layers"][L]["r_star"] for L in JLAYERS}


@app.local_entrypoint()
def main() -> None:
    print(rstar.remote())
