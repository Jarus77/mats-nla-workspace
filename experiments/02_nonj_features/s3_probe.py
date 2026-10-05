"""Step 3: probes (i), J-space removal (ii), J-lens readability (iii), dose statistics, probe export.

Per feature (one H100 container each, no model load needed):
  (i)  linear probe at hs[40..45], 3 grouped (by document) 80/20 splits, Hewitt&Liang control task
       (random label per token type), cluster-bootstrap 95% CI on seed 0.
  (ii) at hs[43] (J-lens L42) and hs[42] (J-lens L41):
       R1  per-token J-space component removal: h - Pi_S h, S = k=25 NN-gradient-pursuit atoms
           (paper's "non-J-space component"); retrained probe and frozen raw probe.
       R1r same with 25 random J-lens atoms per token (control).
       R2a remove top-r PCs of the dataset's J-space components (r=25,100,500) + random r=500 control.
       R2v remove top-r eigvecs of the J-lens atom Gram matrix (r=200, 2000) + random r=2000 control.
       R3  paper's J-orthogonalised mean-difference probe: probe direction minus its own top-25
           pursuit J-component; AUROC / Spearman of the projection before vs after.
  (iii) J-lens readability of the feature value (label tokens in top-k; lens-as-probe AUROC).
  dose: natural projection percentiles along the unit steering direction + residual norms.
  export: /results/02_nonj_features/probes/<feature>_hs<i>.pt

    modal run s3_probe.py [--features col,tense]
"""

import modal

from nonj_common import LENS_FILE_CB, LENS_REPO_CB, OUT, VOLUMES, nonj_image

app = modal.App("nonj-s3-probe")

LABEL_GROUPS = {
    "next_pos": {"NOUN": ["noun", "nouns", "名词"], "VERB": ["verb", "verbs", "动词"],
                 "ADJ": ["adjective", "adjectives", "adj", "形容词"], "ADV": ["adverb", "adverbs", "副词"],
                 "ADP": ["preposition", "prepositions", "介词"], "DET": ["determiner", "determiners", "article"],
                 "PRON": ["pronoun", "pronouns"], "PROPN": ["proper"]},
    "lang": {"Arabic": ["Arabic", "阿拉伯语", "阿拉伯"], "Greek": ["Greek", "希腊语", "希腊"],
             "Polish": ["Polish", "波兰语", "波兰"], "Russian": ["Russian", "俄语", "俄罗斯"],
             "Turkish": ["Turkish", "土耳其语", "土耳其"], "Vietnamese": ["Vietnamese", "越南语", "越南"]},
    "tense": {"Past": ["past", "preterite", "过去", "historical", "ago"], "Pres": ["present", "现在"]},
    "topic": {"World": ["world", "politics", "international", "war", "government"],
              "Sports": ["sports", "sport", "game", "football", "baseball"],
              "Business": ["business", "economy", "finance", "market", "markets", "stocks"],
              "SciTech": ["technology", "tech", "science", "scientific", "computer", "software", "internet"]},
}
LABEL_GROUPS_ALIAS = {"pred_pos": "next_pos"}
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


def _value_words(v: int) -> list[str]:
    """Label words that would express count v in a single-token lens readout."""
    out = [str(v)[0]]  # leading digit (Qwen splits numbers into digits)
    if v <= 20:
        out.append(NUM_WORDS[v])
    elif v < 100:
        out.append(TENS[v // 10])
    return out


# ------------------------------------------------------------------------------------------ torch


def _std(Xtr, *Xs):  # noqa: ANN001, ANN202
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-4)
    return mu, sd, [(X - mu) / sd for X in (Xtr, *Xs)]


def fit_logreg(X, y, K: int, lam: float, iters: int = 150):  # noqa: ANN001, ANN201
    import torch

    W = torch.zeros(X.shape[1], K, device=X.device, requires_grad=True)
    b = torch.zeros(K, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():  # noqa: ANN202
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + lam * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def fit_ridge(X, y, lam: float):  # noqa: ANN001, ANN201
    import torch

    ym = y.mean()
    A = X.T @ X + lam * X.shape[0] * torch.eye(X.shape[1], device=X.device)
    w = torch.linalg.solve(A, X.T @ (y - ym))
    return w, ym


def _reg_metrics(pred, y, feat: str) -> dict:  # noqa: ANN001
    import numpy as np
    from scipy.stats import spearmanr

    p, t = pred, y
    r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    if feat in ("col", "col_report"):
        pb, tb = np.floor(p / 10), np.floor(t / 10)
    else:
        pb, tb = np.round(p), t
    return {"r2": float(r2), "spearman": float(spearmanr(p, t).statistic), "mae": float(np.abs(p - t).mean()),
            "ord_acc": float((pb == tb).mean()), "ord_within1": float((np.abs(pb - tb) <= 1).mean())}


def _cluster_ci(groups, fn, n_boot: int = 500, seed: int = 0) -> list[float]:  # noqa: ANN001
    import numpy as np

    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    pos = {g: np.nonzero(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([pos[g] for g in samp])
        vals.append(fn(idx))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


class Prober:
    """Fits and scores one probe type on (X, y) with grouped splits."""

    def __init__(self, kind: str, feat: str, K: int | None, y, groups, types, n_seeds: int = 3):  # noqa: ANN001
        import numpy as np
        from sklearn.model_selection import GroupShuffleSplit

        self.kind, self.feat, self.K = kind, feat, K
        self.y, self.groups, self.types = y, groups, types
        self.splits = [next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(y, y, groups))
                       for s in range(n_seeds)]
        rng = np.random.default_rng(123)
        ut = np.unique(types)
        vals = rng.choice(y, len(ut), replace=True)  # control task: label sampled per token type
        lut = dict(zip(ut.tolist(), vals.tolist()))
        self.y_ctrl = np.array([lut[t] for t in types.tolist()], dtype=y.dtype)
        self.lam = None

    def _fit(self, Xtr, ytr, lam):  # noqa: ANN001, ANN202
        import torch

        if self.kind == "cls":
            return fit_logreg(Xtr, torch.as_tensor(ytr, device=Xtr.device), self.K, lam)
        return fit_ridge(Xtr, torch.as_tensor(ytr, device=Xtr.device, dtype=torch.float32), lam)

    def _pred(self, model, X):  # noqa: ANN001, ANN202
        if self.kind == "cls":
            W, b = model
            return (X @ W + b).argmax(1).cpu().numpy()
        w, ym = model
        return (X @ w + ym).cpu().numpy()

    def _score(self, pred, y) -> dict:  # noqa: ANN001
        if self.kind == "cls":
            return {"acc": float((pred == y).mean())}
        return _reg_metrics(pred, y, self.feat)

    def select_lam(self, X) -> float:  # noqa: ANN001
        import numpy as np

        grid = [1e-4, 1e-3, 1e-2] if self.kind == "cls" else [1e-3, 1e-2, 1e-1, 1.0]
        tr, _ = self.splits[0]
        rng = np.random.default_rng(7)
        ug = np.unique(self.groups[tr])
        val_g = set(rng.choice(ug, max(1, len(ug) // 5), replace=False).tolist())
        is_val = np.array([g in val_g for g in self.groups[tr]])
        a, b = tr[~is_val], tr[is_val]
        _, _, (Xa, Xb) = _std(X[a], X[b])
        best, best_s = grid[0], -1e9
        for lam in grid:
            m = self._fit(Xa, self.y[a], lam)
            s = self._score(self._pred(m, Xb), self.y[b])
            s = s["acc"] if self.kind == "cls" else s["r2"]
            if s > best_s:
                best, best_s = lam, s
        self.lam = best
        return best

    def run(self, X, control: bool = False, frozen_from: list | None = None, Xeval=None,  # noqa: ANN001
            keep_models: bool = False, ci: bool = False) -> dict:
        """Train on X (train split), test on X (or Xeval). frozen_from: reuse models (no retraining)."""
        import numpy as np

        if self.lam is None:
            self.select_lam(X)
        y = self.y_ctrl if control else self.y
        res, models, preds0 = [], [], None
        for s, (tr, te) in enumerate(self.splits):
            if frozen_from is None:
                mu, sd, (Xtr, Xte) = _std(X[tr], (Xeval if Xeval is not None else X)[te])
                m = self._fit(Xtr, y[tr], self.lam)
            else:
                m, mu, sd = frozen_from[s]
                Xte = ((Xeval if Xeval is not None else X)[te] - mu) / sd
            p = self._pred(m, Xte)
            res.append(self._score(p, y[te]))
            if keep_models:
                models.append((m, mu, sd))
            if s == 0:
                preds0 = (p, te)
        out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        out.update({k + "_sd": float(np.std([r[k] for r in res])) for k in res[0]})
        if ci and preds0 is not None:
            p, te = preds0
            yt, gt = y[te], self.groups[te]
            key = "acc" if self.kind == "cls" else "r2"
            out[key + "_ci95"] = _cluster_ci(gt, lambda idx: self._score(p[idx], yt[idx])[key])
        if keep_models:
            out["_models"] = models
        return out


def _auroc(score, pos) -> float:  # noqa: ANN001
    from sklearn.metrics import roc_auc_score

    if pos.all() or (~pos).all():
        return float("nan")
    return float(roc_auc_score(pos, score))


# ------------------------------------------------------------------------------------------ main


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=6 * 3600, memory=196608)
def probe_feature(fname: str) -> dict:
    import glob
    import json
    import os
    import pickle
    import time

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from features import FEATURES_BY_NAME, nexttok_classes, rows_for
    from nonj_lib import HS_LAYERS, jlens_atoms, load_jlens_layers, nn_pursuit, project_out_span, save_json

    t0 = time.time()
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
    rng = np.random.default_rng(0)
    R = rows_for(feat, items, index, rng)
    rows, y, groups, types, classes = R["rows"], R["y"], R["groups"], R["types"], R["classes"]
    K = len(classes) if classes else None
    print(f"{fname}: {len(rows)} rows, classes={classes}", flush=True)
    counts = {str(c): int((y == i).sum()) for i, c in enumerate(classes)} if classes else None
    result: dict = {"feature": fname, "dataset": feat.dataset, "kind": feat.kind, "n_rows": int(len(rows)),
                    "n_docs": int(len(np.unique(groups))), "class_counts": counts,
                    "classes": [str(c) if fname != "nexttok" else tok.decode([c]) for c in classes] if classes else None,
                    "chance": (1.0 / K) if K else None, "layers": {}, "removal": {}, "lens": {}, "dose": {}}
    if feat.kind == "reg":
        result["y_stats"] = {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std())}

    # J-lens atoms (same lens as capture)
    snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    widx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    ukey = [k for k in widx if k.endswith("lm_head.weight")][0]
    nkey = [k for k in widx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k
            and "mtp" not in k][0]

    def get(key: str):  # noqa: ANN202
        with safe_open(f"{snap}/{widx[key]}", framework="pt") as f:
            return f.get_tensor(key)

    W_U = get(ukey).cuda()
    nw = get(nkey).float().cuda()
    gain = 1 + nw  # Qwen3_5RMSNorm: x * rsqrt(mean x^2 + eps) * (1 + weight)
    lens_path = meta["lens_path"]
    lens_hs = {43: 42, 42: 41}
    Js = dict(zip(lens_hs, load_jlens_layers(lens_path, list(lens_hs.values())).values()))
    lab_words = [x[0] for x in meta["labels"]]

    prober = Prober(feat.kind, fname, K, y, groups, types)
    rstar_path = f"{OUT}/rstar.json"
    r_export: list[int] = [256, 1024]
    if os.path.exists(rstar_path):
        rs_ = json.load(open(rstar_path))["layers"]
        rstar_ = rs_.get("42", {}).get("r_star")
        if rstar_:
            r_export = sorted(set(r_export + [int(rstar_), 4 * int(rstar_)]))
    r_grid = sorted(set([25, 200, 500, 1000, 2000, 4000] + r_export))
    result["r_export"] = r_export
    os.makedirs(f"{OUT}/probes", exist_ok=True)

    for h in HS_LAYERS:
        X = torch.from_numpy(np.load(f"{d}/hs{h}.npy", mmap_mode="r")[rows].astype(np.float32)).cuda()
        prober.lam = None
        raw = prober.run(X, keep_models=True, ci=True)
        models = raw.pop("_models")
        ctrl = prober.run(X, control=True)
        key = "acc" if feat.kind == "cls" else "r2"
        lr = {"raw": raw, "control_task": ctrl, "selectivity": raw[key] - ctrl[key], "lam": prober.lam}
        result["layers"][h] = lr
        print(f"[{fname}] hs{h} raw {key}={raw[key]:.3f} ctrl={ctrl[key]:.3f} t={time.time()-t0:.0f}s", flush=True)

        # ---------------- steering direction + dose statistics (train split of seed 0)
        tr = prober.splits[0][0]
        Xtr = X[tr]
        ytr = torch.as_tensor(y[tr], device=X.device)
        if feat.kind == "reg":
            yc = ytr.float() - ytr.float().mean()
            cdir = ((Xtr - Xtr.mean(0)) * yc[:, None]).mean(0) / yc.var()
            per_unit = cdir.norm().item()  # activation norm per unit of y
            u = cdir / cdir.norm()
            ovr = None
        else:
            means = torch.stack([Xtr[ytr == k].mean(0) for k in range(K)])
            if feat.steer_pair:
                a, b = classes.index(feat.steer_pair[0]), classes.index(feat.steer_pair[1])
                diff = means[a] - means[b]
            else:
                a = int(np.bincount(y).argmax())
                rest = Xtr[ytr != a].mean(0)
                diff = means[a] - rest
            per_unit = diff.norm().item()  # distance between the two class means
            u = diff / diff.norm()
            ovr = torch.stack([means[k] - Xtr[ytr != k].mean(0) for k in range(K)])
        proj = (X @ u).cpu().numpy()
        dev = np.abs(proj - np.median(proj))
        norms = X.norm(dim=1).cpu().numpy()
        dose = {"proj_pct": {p: float(np.percentile(proj, p)) for p in (1, 10, 50, 90, 99)},
                "absdev_pct": {p: float(np.percentile(dev, p)) for p in (50, 90, 99)},
                "proj_std": float(proj.std()), "resid_norm_pct": {p: float(np.percentile(norms, p)) for p in (10, 50, 90)},
                "steer_unit_norm": per_unit}
        dose["doses"] = {"p90": dose["absdev_pct"][90], "p99": dose["absdev_pct"][99], "2xp99": 2 * dose["absdev_pct"][99]}
        dose["dose_over_median_norm"] = {k: v / dose["resid_norm_pct"][50] for k, v in dose["doses"].items()}
        if feat.kind == "cls":
            proj_by = {str(classes[k] if fname != "nexttok" else tok.decode([classes[k]])):
                       float(proj[y == k].mean()) for k in range(K)}
            dose["class_mean_proj"] = proj_by
        result["dose"][h] = dose

        # frozen raw probe weights in raw space (seed 0)
        m0, mu0, sd0 = models[0]
        if feat.kind == "cls":
            Wm, bm = m0
            w_raw = (Wm / sd0.T).cpu()
            b_raw = (bm - (mu0 / sd0) @ Wm).cpu()
        else:
            wv, ym = m0
            w_raw = (wv / sd0[0]).cpu()
            b_raw = (ym - ((mu0 / sd0)[0] * wv).sum()).cpu()
        export = {"feature": fname, "hs_index": h, "block_output": h - 1, "jlens_layer": h - 1,
                  "convention": "hs[i] = HF output_hidden_states[i] = output of decoder block i-1; "
                                "WorkspaceBench/J-lens layer L = hs[L+1]",
                  "kind": feat.kind, "classes": result["classes"],
                  "steer_dir_unit": u.cpu(), "steer_per_unit_norm": per_unit,
                  "steer_def": ("covariance direction Cov(h,y)/Var(y), unit-normalised; per_unit_norm = |shift| per +1 of y"
                                if feat.kind == "reg" else
                                f"class mean difference {feat.steer_pair or 'top class vs rest'}, unit-normalised"),
                  "ovr_mean_diffs": ovr.cpu() if ovr is not None else None,
                  "probe_w_raw": w_raw, "probe_b_raw": b_raw,
                  "probe_def": "logistic (multinomial) or ridge on standardised h; weights folded back to raw h",
                  "probe_lam": prober.lam, "dose": dose, "metrics": {"raw": raw, "control": ctrl}}

        if h in lens_hs:
            atoms, _ = jlens_atoms(Js[h].cuda(), W_U, gain)
            jsp = dict(np.load(f"{d}/jsp_hs{h}.npz"))
            idx = torch.from_numpy(jsp["idx"][rows].astype(np.int64)).cuda()
            coef = torch.from_numpy(jsp["coef"][rows].astype(np.float32)).cuda()
            rem: dict = {"jspace_fve_mean": float(jsp["fve"][rows].mean())}

            def chunked(fn, *args, bs: int = 2048):  # noqa: ANN001, ANN202
                return torch.cat([fn(*[a[i:i + bs] for a in args]) for i in range(0, X.shape[0], bs)])

            X_r1 = chunked(lambda x, ix: project_out_span(x, atoms, ix), X, idx)
            rem["R1_retrain"] = prober.run(X_r1)
            rem["R1_frozen"] = prober.run(X, frozen_from=models, Xeval=X_r1)
            rem["R1_norm_removed_frac"] = float(1 - (X_r1.norm(dim=1) ** 2 / X.norm(dim=1) ** 2).mean())
            ridx = torch.randint(0, atoms.shape[0], idx.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(1))
            X_r1r = chunked(lambda x, ix: project_out_span(x, atoms, ix), X, ridx)
            rem["R1rand_retrain"] = prober.run(X_r1r)
            rem["R1rand_frozen"] = prober.run(X, frozen_from=models, Xeval=X_r1r)
            del X_r1, X_r1r
            # R2a: PCA of the J-space components of these activations
            Jc = chunked(lambda ix, cf: torch.einsum("bk,bkd->bd", cf, atoms[ix].float()), idx, coef)
            Jc = Jc - Jc.mean(0)
            _, _, Vh = torch.linalg.svd(Jc[torch.randperm(Jc.shape[0], device="cuda")[:20000]], full_matrices=False)
            del Jc
            for r in (25, 100, 500):
                Q = Vh[:r].T
                Xp = X - (X @ Q) @ Q.T
                rem[f"R2a_r{r}_retrain"] = prober.run(Xp)
                rem[f"R2a_r{r}_frozen"] = prober.run(X, frozen_from=models, Xeval=Xp)
            g = torch.Generator("cuda").manual_seed(2)
            Qr, _ = torch.linalg.qr(torch.randn(X.shape[1], 500, device="cuda", generator=g))
            rem["R2a_rand500_retrain"] = prober.run(X - (X @ Qr) @ Qr.T)
            # R2v (lead's primary definition, after Viswanath): J-space at rank r = span of the top-r
            # right singular vectors of the lens map M = W_U diag(g) J_l. Curves over r with an
            # equal-rank random-subspace control at every r; enrichment = energy in top-r / (r/d).
            Mg = torch.zeros(X.shape[1], X.shape[1], device="cuda")
            Jg = (gain[:, None] * Js[h].cuda())
            for i in range(0, W_U.shape[0], 32768):
                a_ = W_U[i:i + 32768].float() @ Jg
                Mg += a_.T @ a_
            evals, evecs = torch.linalg.eigh(Mg)
            evals, evecs = evals.flip(0), evecs.flip(1)
            _, Sj, Vjh = torch.linalg.svd(Js[h].cuda(), full_matrices=False)  # J alone (secondary)
            del Mg
            dd = X.shape[1]
            enrich_dirs = {"steer": u[None]}
            if ovr is not None:
                enrich_dirs["ovr_mean_diffs"] = ovr
            Wd = (w_raw.T if feat.kind == "cls" else w_raw[None]).cuda().float()
            enrich_dirs["probe_weights"] = Wd
            rem["R2v_spectrum_energy_frac"] = {}
            rem["enrichment"] = {}
            rgen = torch.Generator("cuda").manual_seed(2)
            for r in r_grid:
                rem["R2v_spectrum_energy_frac"][r] = float(evals[:r].sum() / evals.sum())
                if r in r_export:  # J-removed steering direction at this rank (for G4 / leak checks)
                    Qe = evecs[:, :r]
                    ue = u - Qe @ (Qe.T @ u)
                    export[f"steer_dir_unit_jrem_r{r}"] = (ue / ue.norm()).cpu()
                    export[f"steer_dir_jrem_r{r}_kept_frac"] = float((ue.norm() ** 2).item())
                Q = evecs[:, :r]
                Xp = X - (X @ Q) @ Q.T
                rem[f"R2v_r{r}_retrain"] = prober.run(Xp)
                rem[f"R2v_r{r}_frozen"] = prober.run(X, frozen_from=models, Xeval=Xp)
                Qr, _ = torch.linalg.qr(torch.randn(dd, r, device="cuda", generator=rgen))
                Xr = X - (X @ Qr) @ Qr.T
                rem[f"R2v_rand{r}_retrain"] = prober.run(Xr)
                rem[f"R2v_rand{r}_frozen"] = prober.run(X, frozen_from=models, Xeval=Xr)
                del Xp, Xr
                Qj = Vjh[:r].T
                en = {}
                for nm, D in enrich_dirs.items():
                    Dn = D / D.norm(dim=1, keepdim=True)
                    fm = ((Dn @ Q) ** 2).sum(1)
                    fj = ((Dn @ Qj) ** 2).sum(1)
                    en[nm] = {"M_energy": fm.cpu().tolist(), "M_enrichment": (fm / (r / dd)).cpu().tolist(),
                              "Jonly_energy": fj.cpu().tolist(), "Jonly_enrichment": (fj / (r / dd)).cpu().tolist()}
                # natural activations' own enrichment, for reference
                Xc = X - X.mean(0)
                fx = ((Xc @ Q) ** 2).sum() / (Xc ** 2).sum()
                en["activations_centered"] = {"M_energy": float(fx), "M_enrichment": float(fx / (r / dd))}
                rem["enrichment"][r] = en
            del evecs, Vjh
            # R3: paper's J-orthogonalised mean-difference probe
            te = prober.splits[0][1]
            dirs = (u[None] if ovr is None else ovr)
            pi, pc, rec = nn_pursuit(dirs.float(), atoms, k=25)
            dirs_o = project_out_span(dirs.float(), atoms, pi)
            r3: dict = {"dir_jspace_frac": float(((rec ** 2).sum(1) / (dirs ** 2).sum(1)).mean()),
                        "dir_top_atoms": [[tok.decode([int(t)]) for t in pi[j, :10].tolist()] for j in range(pi.shape[0])]}
            P = (X[te] @ dirs.T).cpu().numpy()
            Po = (X[te] @ dirs_o.T).cpu().numpy()
            yt = y[te]
            if feat.kind == "reg":
                from scipy.stats import spearmanr

                r3["spearman_raw"] = float(spearmanr(P[:, 0], yt).statistic)
                r3["spearman_orth"] = float(spearmanr(Po[:, 0], yt).statistic)
            else:
                au = [_auroc(P[:, k], yt == k) for k in range(K)]
                auo = [_auroc(Po[:, k], yt == k) for k in range(K)]
                zs = [float((P[yt == k, k].mean() - P[yt != k, k].mean()) / P[yt != k, k].std()) for k in range(K)]
                zo = [float((Po[yt == k, k].mean() - Po[yt != k, k].mean()) / Po[yt != k, k].std()) for k in range(K)]
                r3.update(auroc_raw=au, auroc_orth=auo, z_raw=zs, z_orth=zo,
                          auroc_raw_mean=float(np.nanmean(au)), auroc_orth_mean=float(np.nanmean(auo)),
                          z_raw_mean=float(np.mean(zs)), z_orth_mean=float(np.mean(zo)))
            rem["R3"] = r3
            # J-orth steering direction export (steer direction minus its top-25 J-lens component)
            pi1, _, _ = nn_pursuit(u[None].float(), atoms, k=25)
            u_o = project_out_span(u[None].float(), atoms, pi1)[0]
            export["steer_dir_unit_jorth"] = (u_o / u_o.norm()).cpu()
            export["steer_dir_jspace_frac"] = float(1 - (u_o.norm() ** 2).item())
            result["removal"][h] = rem
            k_ = "acc" if feat.kind == "cls" else "r2"
            print(f"[{fname}] hs{h} R1 retrain {rem['R1_retrain'][k_]:.3f} frozen {rem['R1_frozen'][k_]:.3f} "
                  f"R2v2000 {rem['R2v_r2000_retrain'][k_]:.3f} t={time.time()-t0:.0f}s", flush=True)
            del atoms

            # ---------------- (iii) J-lens readability
            L = dict(np.load(f"{d}/lens_hs{h}.npz"))
            lens_res: dict = {}
            for ro in ("std", "cos"):
                top = L[f"top_{ro}"][rows]
                lr_: dict = {}
                lg_name = LABEL_GROUPS_ALIAS.get(fname, fname)
                if lg_name in LABEL_GROUPS:
                    groups_ = LABEL_GROUPS[lg_name]
                    lab_ids = {c: [x[1] for x in meta["labels"] if x[0] in ws] for c, ws in groups_.items()}
                    for kk in (10, 25, 50):
                        hit = {c: np.isin(top[:, :kk], ids).any(1) for c, ids in lab_ids.items()}
                        own = np.array([hit[str(classes[k])][i] if str(classes[k]) in hit else False
                                        for i, k in enumerate(y)])
                        anyl = np.zeros(len(y), bool)
                        for c in hit:
                            anyl |= hit[c]
                        lr_[f"own_label_top{kk}"] = float(own.mean())
                        lr_[f"any_label_top{kk}"] = float(anyl.mean())
                    if ro == "std":
                        ranks = L["lab_rank"][rows]
                        au = []
                        for k, c in enumerate(classes):
                            cols = [j for j, w in enumerate(lab_words) if w in groups_.get(str(c), [])]
                            if cols:
                                au.append(_auroc(-ranks[:, cols].min(1), y == k))
                        lr_["lens_as_probe_auroc_mean"] = float(np.nanmean(au)) if au else None
                elif fname in ("col", "depth", "col_report"):
                    num_words = set(NUM_WORDS) | set(TENS.values()) | {str(i) for i in range(10)} | {"hundred"}
                    num_ids = [x[1] for x in meta["labels"] if x[0] in num_words]
                    yv = y.astype(int)
                    rng2 = np.random.default_rng(5)
                    yperm = rng2.permutation(yv)
                    for kk in (10, 25, 50):
                        lr_[f"any_number_top{kk}"] = float(np.isin(top[:, :kk], num_ids).any(1).mean())
                        corr_ids = [[x[1] for x in meta["labels"] if x[0] in _value_words(int(v))] for v in yv]
                        perm_ids = [[x[1] for x in meta["labels"] if x[0] in _value_words(int(v))] for v in yperm]
                        lr_[f"correct_value_top{kk}"] = float(np.mean([np.isin(top[i, :kk], c).any() for i, c in enumerate(corr_ids)]))
                        lr_[f"permuted_value_top{kk}"] = float(np.mean([np.isin(top[i, :kk], c).any() for i, c in enumerate(perm_ids)]))
                elif fname == "nexttok":
                    nt = index["ntid"][rows]
                    for kk in (1, 10, 25, 50):
                        lr_[f"next_token_top{kk}"] = float((top[:, :kk] == nt[:, None]).any(1).mean())
                lens_res[ro] = lr_
            if fname == "nexttok":
                nr = L["next_rank"][rows]
                lens_res["std"]["next_rank_median"] = float(np.median(nr))
            # most frequent lens tokens at these positions (sanity)
            vals, cnts = np.unique(L["top_std"][rows][:, :10], return_counts=True)
            lens_res["frequent_top10_tokens"] = [(tok.decode([int(v)]), int(c)) for v, c in
                                                 sorted(zip(vals, cnts), key=lambda z: -z[1])[:30]]
            result["lens"][h] = lens_res
        torch.save(export, f"{OUT}/probes/{fname}_hs{h}.pt")
        del X
        torch.cuda.empty_cache()
    save_json(result, f"{OUT}/probe_results/{fname}.json")
    VOLUMES["/results"].commit()
    print(f"[{fname}] done in {time.time()-t0:.0f}s", flush=True)
    return {"feature": fname, "secs": time.time() - t0}


@app.local_entrypoint()
def main(features: str = "col,next_pos,tense,depth,topic,nexttok") -> None:
    for r in probe_feature.map(features.split(","), return_exceptions=True):
        print(r)
