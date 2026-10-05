"""Step 7: is the feature value readable from the J-lens top-k LIST (what a human/judge would see),
beyond what local text gives? And how much of each feature is explained by token identity?

For each feature (same rows as s3), grouped 80/20 splits x 3 seeds:
  lens_bag     bag of the top-25 J-lens tokens (neuronpedia n1000, standard readout, hs43=J L42)
  cur_tok      one-hot of the current token
  local4       bag of the last 4 tokens (t-3..t)
  local4+lens  both
Logistic regression (classification) or ridge (regression) on sparse features.

    modal run s7_lensbag.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s7-lensbag")


@app.function(image=nonj_image, volumes=VOLUMES, timeout=3 * 3600, memory=65536, cpu=16)
def lensbag(fname: str, hs: int = 43, k: int = 25) -> dict:
    import pickle

    import numpy as np
    import scipy.sparse as sp
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import GroupShuffleSplit
    from transformers import AutoTokenizer

    from features import FEATURES_BY_NAME, nexttok_classes, rows_for
    from nonj_lib import save_json

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
    rows, y, groups = R["rows"], R["y"], R["groups"]
    top = np.load(f"{d}/lens_hs{hs}.npz")["top_std"][rows, :k]
    V = int(max(top.max(), index["tid"].max())) + 1
    n = len(rows)

    def bag(cols: np.ndarray) -> sp.csr_matrix:
        r = np.repeat(np.arange(n), cols.shape[1])
        c = cols.reshape(-1)
        ok = c >= 0
        return sp.csr_matrix((np.ones(ok.sum()), (r[ok], c[ok])), shape=(n, V)).sign()

    tid, item, pos = index["tid"], index["item"], index["pos"]
    loc = np.stack([np.where((pos[rows] - o) >= 0, tid[np.clip(rows - o, 0, None)], -1) for o in range(4)], 1)
    feats = {"lens_bag": bag(top), "cur_tok": bag(loc[:, :1]), "local4": bag(loc)}
    feats["local4+lens"] = sp.hstack([feats["local4"], feats["lens_bag"]]).tocsr()
    out: dict = {"feature": fname, "n": int(n), "k": k, "hs": hs}
    splits = [next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(y, y, groups)) for s in range(3)]
    for name, X in feats.items():
        scores = []
        for tr, te in splits:
            if feat.kind == "cls":
                m = LogisticRegression(C=1.0, max_iter=300, solver="saga").fit(X[tr], y[tr])
                scores.append({"acc": float((m.predict(X[te]) == y[te]).mean())})
            else:
                m = RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0, 10000.0]).fit(X[tr], y[tr])
                p = m.predict(X[te])
                r2 = 1 - ((p - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
                scores.append({"r2": float(r2), "spearman": float(spearmanr(p, y[te]).statistic)})
        if name == "lens_bag":  # which lens tokens carry the signal (last fitted split)
            from transformers import AutoTokenizer as _AT

            _tok = _AT.from_pretrained("Qwen/Qwen3.6-27B")
            coef = m.coef_ if feat.kind == "reg" else np.abs(m.coef_).max(0)
            topc = np.argsort(-np.abs(coef))[:25]
            out["lens_bag_top_tokens"] = [(_tok.decode([int(t)]), float(coef[t])) for t in topc]
        out[name] = {kk: float(np.mean([s[kk] for s in scores])) for kk in scores[0]}
        out[name].update({kk + "_sd": float(np.std([s[kk] for s in scores])) for kk in scores[0]})
        print(fname, name, out[name], flush=True)
    save_json(out, f"{OUT}/lensbag/{fname}_hs{hs}.json")
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main(features: str = "col,next_pos,tense,depth,topic,nexttok") -> None:
    for r in lensbag.map(features.split(","), return_exceptions=True):
        print(r)
