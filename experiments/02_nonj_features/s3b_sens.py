"""Step 3b: layer sensitivity of G2/G3 at L40 and L44 (block outputs 40/44 = hs[41]/hs[45]),
complementing s3 (which does G2/G3 at L42 = hs[43] and L41 = hs[42]).

Per feature and layer: probe (3 seeds), J-space removal curves (top-r right singular vectors of
M_L = W_U diag(g) J_L, neuronpedia n1000) retrained + frozen + equal-rank random, steering-direction
enrichment, and J-lens (standard readout) readability computed from the cached activations.

    modal run s3b_sens.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s3b-sens")

SENS = {41: 40, 45: 44}  # hs index -> J-lens layer
RS = [256, 384, 1024, 1536, 4000]


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=196608)
def sens(fname: str) -> dict:
    import glob
    import json
    import pickle

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from features import FEATURES_BY_NAME, nexttok_classes, rows_for
    from nonj_lib import load_jlens_layers, load_model, save_json  # noqa: F401
    from s3_probe import LABEL_GROUPS, LABEL_GROUPS_ALIAS, NUM_WORDS, TENS, Prober, _auroc, _value_words

    VOLUMES["/results"].reload()
    feat = FEATURES_BY_NAME[fname]
    d = f"{OUT}/acts/{feat.dataset}"
    items = pickle.load(open(f"{OUT}/data/{feat.dataset}.pkl", "rb"))
    index = dict(np.load(f"{d}/index.npz"))
    meta = json.load(open(f"{d}/meta.json"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    if fname == "nexttok":
        index["nexttok_classes"] = nexttok_classes(tok, index["ntid"])
    if fname == "pred_pos":
        index["class_mass"] = np.load(f"{d}/pred.npz")["class_mass"]
    R = rows_for(feat, items, index, np.random.default_rng(0))
    rows, y, groups, types, classes = R["rows"], R["y"], R["groups"], R["types"], R["classes"]
    K = len(classes) if classes else None

    from safetensors import safe_open

    snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    widx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    ukey = [k for k in widx if k.endswith("lm_head.weight")][0]
    nkey = [k for k in widx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k and "mtp" not in k][0]

    def get(key: str):  # noqa: ANN202
        with safe_open(f"{snap}/{widx[key]}", framework="pt") as f:
            return f.get_tensor(key)

    W_U = get(ukey).cuda()
    gain = 1 + get(nkey).float().cuda()  # Qwen3_5RMSNorm: x * rsqrt(mean x^2 + eps) * (1 + weight)
    lens_path = glob.glob(f"{OUT}/lenses/np/**/*.pt", recursive=True)[0]
    Js = load_jlens_layers(lens_path, list(SENS.values()))
    key = "acc" if feat.kind == "cls" else "r2"
    out: dict = {"feature": fname, "layers": {}}
    for h, L in SENS.items():
        X = torch.from_numpy(np.load(f"{d}/hs{h}.npy", mmap_mode="r")[rows].astype(np.float32)).cuda()
        pr = Prober(feat.kind, fname, K, y, groups, types)
        raw = pr.run(X, keep_models=True)
        models = raw.pop("_models")
        res: dict = {"hs_index": h, "jlens_layer": L, "raw": raw}
        tr = pr.splits[0][0]
        Xtr, ytr = X[tr], torch.as_tensor(y[tr], device="cuda")
        if feat.kind == "reg":
            yc = ytr.float() - ytr.float().mean()
            u = ((Xtr - Xtr.mean(0)) * yc[:, None]).mean(0)
        else:
            if feat.steer_pair:
                a, b = classes.index(feat.steer_pair[0]), classes.index(feat.steer_pair[1])
                u = Xtr[ytr == a].mean(0) - Xtr[ytr == b].mean(0)
            else:
                a = int(np.bincount(y).argmax())
                u = Xtr[ytr == a].mean(0) - Xtr[ytr != a].mean(0)
        u = u / u.norm()
        J = Js[L].cuda()
        Jg = gain[:, None] * J
        Mg = torch.zeros(J.shape[0], J.shape[0], device="cuda")
        for i in range(0, W_U.shape[0], 32768):
            a_ = W_U[i:i + 32768].float() @ Jg
            Mg += a_.T @ a_
        _, evecs = torch.linalg.eigh(Mg)
        evecs = evecs.flip(1)
        g = torch.Generator("cuda").manual_seed(2)
        Xc = X - X.mean(0)
        for r in RS:
            Q = evecs[:, :r]
            Xp = X - (X @ Q) @ Q.T
            Qr, _ = torch.linalg.qr(torch.randn(X.shape[1], r, device="cuda", generator=g))
            Xr = X - (X @ Qr) @ Qr.T
            fu = float(((u @ Q) ** 2).sum())
            fx = float(((Xc @ Q) ** 2).sum() / (Xc ** 2).sum())
            res[r] = {"retrain": pr.run(Xp)[key], "frozen": pr.run(X, frozen_from=models, Xeval=Xp)[key],
                      "rand_retrain": pr.run(Xr)[key], "rand_frozen": pr.run(X, frozen_from=models, Xeval=Xr)[key],
                      "steer_enrichment": fu / (r / X.shape[1]), "acts_enrichment": fx / (r / X.shape[1])}
        # lens readability (standard readout, gain-folded RMSNorm, eps ignored)
        tops = []
        for i in range(0, X.shape[0], 1024):
            Z = X[i:i + 1024] @ J.T
            Z = Z / Z.pow(2).mean(1, keepdim=True).add(1e-6).sqrt() * gain
            tops.append((Z.to(torch.bfloat16) @ W_U.T).topk(50, dim=1).indices.cpu().numpy())
        top = np.concatenate(tops)
        lr: dict = {}
        lg = LABEL_GROUPS_ALIAS.get(fname, fname)
        if lg in LABEL_GROUPS:
            ids = {c: [x[1] for x in meta["labels"] if x[0] in ws] for c, ws in LABEL_GROUPS[lg].items()}
            for kk in (10, 50):
                own = np.array([np.isin(top[i, :kk], ids.get(str(classes[k]), [])).any() for i, k in enumerate(y)])
                lr[f"own_label_top{kk}"] = float(own.mean())
        elif fname in ("col", "depth", "col_report"):
            nums = set(NUM_WORDS) | set(TENS.values()) | {str(i) for i in range(10)} | {"hundred"}
            nid = [x[1] for x in meta["labels"] if x[0] in nums]
            for kk in (10, 50):
                lr[f"any_number_top{kk}"] = float(np.isin(top[:, :kk], nid).any(1).mean())
                cid = [[x[1] for x in meta["labels"] if x[0] in _value_words(int(v))] for v in y]
                lr[f"correct_value_top{kk}"] = float(np.mean([np.isin(top[i, :kk], c).any() for i, c in enumerate(cid)]))
        elif fname == "nexttok":
            nt = index["ntid"][rows]
            lr["next_token_top10"] = float((top[:, :10] == nt[:, None]).any(1).mean())
        res["lens"] = lr
        out["layers"][h] = res
        print(fname, h, {k_: v for k_, v in res.items() if k_ != "raw"}, flush=True)
        del X, Xp, Xr
        torch.cuda.empty_cache()
    _ = _auroc
    save_json(out, f"{OUT}/probe_results_sens/{fname}.json")
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main(features: str = "col,depth,pred_pos,tense,lang,col_report,topic,nexttok") -> None:
    for r in sens.map(features.split(","), return_exceptions=True):
        print(r if not isinstance(r, dict) else r["feature"])
