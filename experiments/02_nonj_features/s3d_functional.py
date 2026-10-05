"""Step 3d: FUNCTIONAL non-J criterion (lead, G2 final): does the FULL J-lens readout (neuronpedia
n1000, standard readout, L42 = hs[43]) track the feature on natural items?

Score of value v at a row = -log10(best rank among the J-lens tokens that name v).
  classes (pred_pos, tense, lang, topic): one-vs-rest AUROC of the class's score, averaged over
           classes; null band = same statistic with labels permuted (200x).
  counts (col, depth, col_report): paired AUROC of score(row, true value) vs score(row, value of
           a random other row); null = the same with two independent permutations (200x).
  nexttok: paired AUROC using the position of the token in the top-50 list (51 if absent).
A feature is J-readable if the statistic is above the null 95% band.

    modal run s3d_functional.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s3d-functional")


@app.function(image=nonj_image, volumes=VOLUMES, timeout=3600, memory=65536, cpu=8, max_containers=16)
def functional(fname: str, hs: int = 43, n_perm: int = 200) -> dict:
    import json
    import pickle

    import numpy as np
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer

    from features import FEATURES_BY_NAME, nexttok_classes, rows_for
    from nonj_lib import save_json
    from s3_probe import LABEL_GROUPS, LABEL_GROUPS_ALIAS, _value_words

    VOLUMES["/results"].reload()
    feat = FEATURES_BY_NAME[fname]
    d = f"{OUT}/acts/{feat.dataset}"
    items = pickle.load(open(f"{OUT}/data/{feat.dataset}.pkl", "rb"))
    index = dict(np.load(f"{d}/index.npz"))
    meta = json.load(open(f"{d}/meta.json"))
    if fname == "nexttok":
        index["nexttok_classes"] = nexttok_classes(AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B"), index["ntid"])
    if fname == "pred_pos":
        index["class_mass"] = np.load(f"{d}/pred.npz")["class_mass"]
    R = rows_for(feat, items, index, np.random.default_rng(0))
    rows, y, classes = R["rows"], R["y"], R["classes"]
    L = np.load(f"{d}/lens_hs{hs}.npz")
    words = [x[0] for x in meta["labels"]]
    rng = np.random.default_rng(3)
    out: dict = {"feature": fname, "hs": hs, "n": int(len(rows))}

    def band(v: list[float]) -> list[float]:
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    if fname == "nexttok":
        top = L["top_std"][rows]
        nt = index["ntid"][rows]

        def pos_of(tokens: np.ndarray) -> np.ndarray:
            hit = top == tokens[:, None]
            return np.where(hit.any(1), hit.argmax(1), 51)

        s_true = -pos_of(nt)

        def paired(a: np.ndarray, b: np.ndarray) -> float:
            return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))

        out["stat"] = paired(s_true, -pos_of(rng.permutation(nt)))
        out["null"] = [paired(-pos_of(rng.permutation(nt)), -pos_of(rng.permutation(nt))) for _ in range(n_perm)]
    else:
        ranks = L["lab_rank"][rows].astype(np.float64)
        lg = LABEL_GROUPS_ALIAS.get(fname, fname)
        if lg in LABEL_GROUPS:
            sc = []
            for c in classes:
                cols = [j for j, w in enumerate(words) if w in LABEL_GROUPS[lg].get(str(c), [])]
                sc.append(-np.log10(ranks[:, cols].min(1) + 1) if cols else np.zeros(len(rows)))
            S = np.stack(sc, 1)

            def ovr(yy: np.ndarray) -> float:
                return float(np.nanmean([roc_auc_score(yy == k, S[:, k]) for k in range(len(classes)) if 0 < (yy == k).sum() < len(yy)]))

            out["stat"] = ovr(y)
            out["null"] = [ovr(rng.permutation(y)) for _ in range(n_perm)]
            out["per_class_auroc"] = {str(c): float(roc_auc_score(y == k, S[:, k])) for k, c in enumerate(classes)}
        else:
            vals = sorted(set(int(v) for v in y))
            vcols = {v: [j for j, w in enumerate(words) if w in _value_words(v)] for v in vals}
            Sv = {v: -np.log10(ranks[:, vcols[v]].min(1) + 1) for v in vals if vcols[v]}
            yi = y.astype(int)

            def score(vv: np.ndarray) -> np.ndarray:
                return np.array([Sv[v][i] if v in Sv else 0.0 for i, v in enumerate(vv)])

            def paired(a: np.ndarray, b: np.ndarray) -> float:
                keep = a != b
                return float(roc_auc_score(np.r_[np.ones(keep.sum()), np.zeros(keep.sum())], np.r_[a[keep], b[keep]]))

            # compare score at the true value with the score at a permuted value (rows where values differ)
            def paired_vals(v1: np.ndarray, v2: np.ndarray) -> float:
                keep = v1 != v2
                s1, s2 = score(v1)[keep], score(v2)[keep]
                return float(roc_auc_score(np.r_[np.ones(len(s1)), np.zeros(len(s2))], np.r_[s1, s2]))

            out["stat"] = paired_vals(yi, rng.permutation(yi))
            out["null"] = [paired_vals(rng.permutation(yi), rng.permutation(yi)) for _ in range(n_perm)]
            _ = paired
    out["null_band95"] = band(out["null"])
    out["null_mean"] = float(np.mean(out["null"]))
    out["j_readable"] = bool(out["stat"] > out["null_band95"][1])
    out.pop("null")
    print(out, flush=True)
    save_json(out, f"{OUT}/functional/{fname}_hs{hs}.json")
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main(features: str = "col,depth,pred_pos,tense,lang,col_report,topic,nexttok") -> None:
    for r in functional.map(features.split(","), return_exceptions=True):
        print(r)
