"""Step 3c: revised G2 (lead, after the 03_pipeline_smoke finding that strong features survive ANY
equal-rank removal). For every feature at L42 (hs[43]; L40/L44 = hs[41]/hs[45] as sensitivity):

 (a) J-ONLY probe: probe trained/evaluated on the coordinates of P_J h (projection ONTO the top-r
     J-space, r = r* = 384 and 4r* = 1536) vs probes on projections onto K=10 random equal-rank
     subspaces (distribution + 95% band).
 (b) enrichment ||Q_r^T d||^2 / (r/d) of: the mean-diff/covariance steering direction, the probe
     direction (ridge on raw centred h), and the top-8 INLP directions (iterative ridge-probe
     nullspace projection on the binary steer pair / the regression target). Random = 1.0.
 (c) old survival test, now relative: retention_J(r) - mean retention over 10 random equal-rank
     removals (retention = score after removal / score before; retrained probes).
J-space = top-r right singular vectors of M_L = W_U diag(1+w_norm) J_L (neuronpedia n1000).

    modal run s3c_jonly.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s3c-jonly")

LAYERS = {43: 42, 41: 40, 45: 44}  # hs index -> J-lens layer
RS = [384, 1536]
N_RAND = 10


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=196608)
def jonly(fname: str) -> dict:
    import glob
    import json
    import pickle

    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from features import FEATURES_BY_NAME, nexttok_classes, rows_for
    from nonj_lib import load_jlens_layers, save_json
    from s3_probe import Prober

    VOLUMES["/results"].reload()
    feat = FEATURES_BY_NAME[fname]
    d = f"{OUT}/acts/{feat.dataset}"
    items = pickle.load(open(f"{OUT}/data/{feat.dataset}.pkl", "rb"))
    index = dict(np.load(f"{d}/index.npz"))
    if fname == "nexttok":
        index["nexttok_classes"] = nexttok_classes(AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B"), index["ntid"])
    if fname == "pred_pos":
        index["class_mass"] = np.load(f"{d}/pred.npz")["class_mass"]
    R = rows_for(feat, items, index, np.random.default_rng(0))
    rows, y, groups, types, classes = R["rows"], R["y"], R["groups"], R["types"], R["classes"]
    K = len(classes) if classes else None
    key = "acc" if feat.kind == "cls" else "r2"

    snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    widx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    ukey = [k for k in widx if k.endswith("lm_head.weight")][0]
    nkey = [k for k in widx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k and "mtp" not in k][0]

    def get(k_: str):  # noqa: ANN202
        with safe_open(f"{snap}/{widx[k_]}", framework="pt") as f:
            return f.get_tensor(k_)

    W_U = get(ukey).cuda()
    gain = 1 + get(nkey).float().cuda()  # Qwen3_5RMSNorm uses (1 + weight)
    lens_path = glob.glob(f"{OUT}/lenses/np/**/*.pt", recursive=True)[0]
    Js = load_jlens_layers(lens_path, list(LAYERS.values()))
    out: dict = {"feature": fname, "kind": feat.kind, "metric": key, "chance": (1.0 / K) if K else 0.0, "layers": {}}

    for h, L in LAYERS.items():
        X = torch.from_numpy(np.load(f"{d}/hs{h}.npy", mmap_mode="r")[rows].astype(np.float32)).cuda()
        D = X.shape[1]
        J = Js[L].cuda()
        Jg = gain[:, None] * J
        Mg = torch.zeros(D, D, device="cuda")
        for i in range(0, W_U.shape[0], 32768):
            a_ = W_U[i:i + 32768].float() @ Jg
            Mg += a_.T @ a_
        _, ev = torch.linalg.eigh(Mg)
        ev = ev.flip(1)
        del Mg
        pr = Prober(feat.kind, fname, K, y, groups, types)
        base = pr.run(X)[key]
        res: dict = {"hs_index": h, "jlens_layer": L, "base": base}
        tr = pr.splits[0][0]
        Xc = X - X[tr].mean(0)
        # directions: steering (mean diff / covariance), ridge probe, INLP
        ytr = torch.as_tensor(y[tr], device="cuda")
        if feat.kind == "reg":
            t = (ytr.float() - ytr.float().mean())
            sel = torch.arange(len(tr), device="cuda")
        else:
            if feat.steer_pair:
                a, b = classes.index(feat.steer_pair[0]), classes.index(feat.steer_pair[1])
            else:
                a = int(np.bincount(y).argmax())
                b = None
            m = (ytr == a) | ((ytr == b) if b is not None else torch.ones_like(ytr, dtype=torch.bool))
            sel = torch.nonzero(m).squeeze(1)
            t = torch.where(ytr[sel] == a, 1.0, -1.0)
        Xs = Xc[tr][sel]
        steer = (Xs * (t - t.mean())[:, None]).mean(0)
        dirs = {"steer": steer / steer.norm()}
        lam = 1e-2 * (Xs ** 2).sum() / Xs.shape[0]
        P = torch.eye(D, device="cuda")
        inlp = []
        for _ in range(8):
            Z = Xs @ P
            w = torch.linalg.solve(Z.T @ Z + lam * torch.eye(D, device="cuda"), Z.T @ (t - t.mean()))
            w = P @ w
            w = w / w.norm()
            inlp.append(w)
            P = P - torch.outer(w, w)
        dirs["ridge_probe"] = inlp[0]
        for k_, w in enumerate(inlp):
            dirs[f"inlp{k_}"] = w
        g = torch.Generator("cuda").manual_seed(17)
        for r in RS:
            Q = ev[:, :r]
            rr: dict = {}
            # (b) enrichment
            rr["enrichment"] = {n_: float(((v @ Q) ** 2).sum() / (r / D)) for n_, v in dirs.items()}
            rr["enrichment_acts"] = float(((Xc @ Q) ** 2).sum() / (Xc ** 2).sum() / (r / D))
            # (a) J-only probe vs random-subspace probes; (c) relative survival
            rr["jonly"] = pr.run(X @ Q)[key]
            rr["jremoved"] = pr.run(X - (X @ Q) @ Q.T)[key]
            rnd_only, rnd_rem = [], []
            for _ in range(N_RAND):
                Qr, _ = torch.linalg.qr(torch.randn(D, r, device="cuda", generator=g))
                rnd_only.append(pr.run(X @ Qr)[key])
                rnd_rem.append(pr.run(X - (X @ Qr) @ Qr.T)[key])
            ro = np.array(rnd_only)
            rr["rand_only"] = ro.tolist()
            rr["rand_only_mean"], rr["rand_only_band95"] = float(ro.mean()), [float(np.percentile(ro, 2.5)), float(np.percentile(ro, 97.5))]
            rr["jonly_z_vs_rand"] = float((rr["jonly"] - ro.mean()) / max(ro.std(), 1e-6))
            rr["rand_removed"] = rnd_rem
            rr["retention_J"] = rr["jremoved"] / base
            rr["retention_rand_mean"] = float(np.mean(rnd_rem) / base)
            rr["retention_J_minus_rand"] = rr["retention_J"] - rr["retention_rand_mean"]
            res[r] = rr
            print(fname, h, r, {k_: v for k_, v in rr.items() if k_ not in ("rand_only", "rand_removed", "enrichment")},
                  {k_: round(v, 2) for k_, v in rr["enrichment"].items()}, flush=True)
        out["layers"][h] = res
        del X, Xc
        torch.cuda.empty_cache()
    save_json(out, f"{OUT}/jonly/{fname}.json")
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main(features: str = "col,depth,pred_pos,tense,lang,col_report,topic,nexttok") -> None:
    for r in jonly.map(features.split(","), return_exceptions=True):
        print(r if not isinstance(r, dict) else r["feature"])
