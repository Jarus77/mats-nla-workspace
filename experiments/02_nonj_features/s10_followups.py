"""Step 10 (lead, round 4 follow-ups), one container per (feature, lens):

 1. Generalised functional gate: "the lens readout contains the feature's value in the form the lens
    emits". For pred_pos the J-lens emits words, so the top-10 J-lens tokens at the read position are
    classified by POS (corpus lexicon: word-initial tokens whose next-word UPOS is >= 80% one class,
    >= 3 occurrences) and the per-class fraction is scored by one-vs-rest AUROC against the predicted-POS
    label (null: 200 label permutations). Same token-class test run for tense (past vs present verb forms).
 2. Lens robustness: gate col, col_report, tense, lang (and pred_pos) under the camilablank n25 j-lens
    as well as neuronpedia n1000: diff-in-means enrichment at matched r (384, 1536) and the label-token
    functional AUROC (as in s3d) computed from that lens's readout at L42 (hs[43]).

    modal run s10_followups.py
"""

import modal

from nonj_common import LENS_FILE_CB, LENS_REPO_CB, OUT, VOLUMES, nonj_image

app = modal.App("nonj-s10-followups")
RS = [384, 1536]
PAST = ["was", "were", "had", "did", "said", "became", "made", "went", "took", "came", "gave", "found",
        "began", "remained", "included", "served", "played", "led", "won", "received", "used", "called",
        "left", "saw", "held", "wrote", "moved", "died", "lived", "stated", "appeared", "showed", "seemed"]
PRES = ["is", "are", "has", "does", "says", "becomes", "makes", "goes", "takes", "comes", "gives", "finds",
        "begins", "remains", "includes", "serves", "plays", "leads", "wins", "receives", "uses", "calls",
        "leaves", "sees", "holds", "writes", "moves", "dies", "lives", "states", "appears", "shows", "seems"]


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=2 * 3600, memory=196608, max_containers=16)
def followup(fname: str, lens: str) -> dict:
    import collections
    import glob
    import json
    import pickle

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer

    from features import FEATURES_BY_NAME, POS_CLASSES, rows_for
    from nonj_lib import load_jlens_layers, save_json
    from s3_probe import LABEL_GROUPS, LABEL_GROUPS_ALIAS, _value_words

    VOLUMES["/results"].reload()
    feat = FEATURES_BY_NAME[fname]
    d = f"{OUT}/acts/{feat.dataset}"
    items = pickle.load(open(f"{OUT}/data/{feat.dataset}.pkl", "rb"))
    index = dict(np.load(f"{d}/index.npz"))
    meta = json.load(open(f"{d}/meta.json"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    if fname == "pred_pos":
        index["class_mass"] = np.load(f"{d}/pred.npz")["class_mass"]
    R = rows_for(feat, items, index, np.random.default_rng(0))
    rows, y, classes = R["rows"], R["y"], R["classes"]
    rng = np.random.default_rng(3)

    snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    widx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    ukey = [k for k in widx if k.endswith("lm_head.weight")][0]
    nkey = [k for k in widx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k and "mtp" not in k][0]

    def get(k_: str):  # noqa: ANN202
        with safe_open(f"{snap}/{widx[k_]}", framework="pt") as f:
            return f.get_tensor(k_)

    W_U = get(ukey).cuda()
    gain = 1 + get(nkey).float().cuda()  # Qwen3_5RMSNorm: (1 + weight)
    path = (glob.glob(f"{OUT}/lenses/np/**/*.pt", recursive=True)[0] if lens == "np"
            else hf_hub_download(LENS_REPO_CB, LENS_FILE_CB))
    J = load_jlens_layers(path, [42])[42].cuda()
    X = torch.from_numpy(np.load(f"{d}/hs43.npy", mmap_mode="r")[rows].astype(np.float32)).cuda()
    D = X.shape[1]
    out: dict = {"feature": fname, "lens": lens, "n": int(len(rows)), "hs": 43, "jlens_layer": 42}

    # ---- enrichment of the diff-in-means / covariance direction (train split = all rows; direction only)
    yt = torch.as_tensor(y, device="cuda")
    if feat.kind == "reg":
        yc = yt.float() - yt.float().mean()
        u = ((X - X.mean(0)) * yc[:, None]).mean(0)
    else:
        a, b = classes.index(feat.steer_pair[0]), classes.index(feat.steer_pair[1])
        u = X[yt == a].mean(0) - X[yt == b].mean(0)
    u = u / u.norm()
    Mg = torch.zeros(D, D, device="cuda")
    Jg = gain[:, None] * J
    for i in range(0, W_U.shape[0], 32768):
        a_ = W_U[i:i + 32768].float() @ Jg
        Mg += a_.T @ a_
    ev_, evec = torch.linalg.eigh(Mg)
    evec = evec.flip(1)
    Xc = X - X.mean(0)
    out["enrichment"] = {r: float(((u @ evec[:, :r]) ** 2).sum() / (r / D)) for r in RS}
    out["enrichment_acts"] = {r: float(((Xc @ evec[:, :r]) ** 2).sum() / (Xc ** 2).sum() / (r / D)) for r in RS}
    del Mg

    # ---- lens readout at the rows: top-10 tokens + ranks of label tokens
    lab = meta["labels"]
    lab_ids = torch.tensor([x[1] for x in lab], device="cuda")
    words = [x[0] for x in lab]
    tops, ranks = [], []
    for i in range(0, X.shape[0], 512):
        Z = X[i:i + 512] @ J.T
        Z = Z / Z.pow(2).mean(1, keepdim=True).add(1e-6).sqrt() * gain
        sc = (Z.to(torch.bfloat16) @ W_U.T).float()
        tops.append(sc.topk(10, dim=1).indices.cpu().numpy())
        srt = sc.sort(dim=1).values
        ranks.append((sc.shape[1] - torch.searchsorted(srt, sc[:, lab_ids].contiguous(), right=True)).cpu().numpy())
    top = np.concatenate(tops)
    rk = np.concatenate(ranks).astype(np.float64)

    def band(v: list[float]) -> list[float]:
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def ovr_auc(S: np.ndarray, yy: np.ndarray, ks: list[int]) -> float:
        return float(np.nanmean([roc_auc_score(yy == k, S[:, k]) for k in ks if 0 < (yy == k).sum() < len(yy) and S[:, k].std() > 0]))

    # ---- (A) label-token functional test (as s3d)
    lg = LABEL_GROUPS_ALIAS.get(fname, fname)
    if lg in LABEL_GROUPS:
        S = np.stack([(-np.log10(rk[:, [j for j, w in enumerate(words) if w in LABEL_GROUPS[lg].get(str(c), [])]].min(1) + 1)
                       if any(w in LABEL_GROUPS[lg].get(str(c), []) for w in words) else np.zeros(len(rows)))
                      for c in classes], 1)
        ks = list(range(len(classes)))
        stat = ovr_auc(S, y, ks)
        null = [ovr_auc(S, rng.permutation(y), ks) for _ in range(200)]
    else:
        vals = sorted(set(int(v) for v in y))
        Sv = {v: -np.log10(rk[:, [j for j, w in enumerate(words) if w in _value_words(v)]].min(1) + 1)
              for v in vals if any(w in _value_words(v) for w in words)}
        yi = y.astype(int)

        def sc_(vv: np.ndarray) -> np.ndarray:
            return np.array([Sv[v][i] if v in Sv else 0.0 for i, v in enumerate(vv)])

        def paired(v1: np.ndarray, v2: np.ndarray) -> float:
            keep = v1 != v2
            s1, s2 = sc_(v1)[keep], sc_(v2)[keep]
            return float(roc_auc_score(np.r_[np.ones(len(s1)), np.zeros(len(s2))], np.r_[s1, s2]))

        stat = paired(yi, rng.permutation(yi))
        null = [paired(rng.permutation(yi), rng.permutation(yi)) for _ in range(200)]
    out["label_token_functional"] = {"stat": stat, "null_band95": band(null), "j_readable": bool(stat > band(null)[1])}

    # ---- (B) token-class functional test: the value in the form the lens emits (words)
    if fname in ("pred_pos", "tense"):
        if fname == "pred_pos":
            cnt = collections.defaultdict(collections.Counter)
            for it in pickle.load(open(f"{OUT}/data/prose.pkl", "rb")):
                for j, labv in enumerate(it["next_pos"]):
                    if labv and j + 1 < len(it["ids"]):
                        cnt[it["ids"][j + 1]][labv] += 1
            lex = {}
            for t, c in cnt.items():
                tot = sum(c.values())
                lb, k = c.most_common(1)[0]
                if tot >= 3 and k / tot >= 0.8 and lb in POS_CLASSES:
                    lex[t] = POS_CLASSES.index(lb)
            cls_names = POS_CLASSES
            lab_to_k = {c: POS_CLASSES.index(c) for c in classes}
        else:
            lex = {}
            for k, ws in enumerate((PAST, PRES)):
                for w in ws:
                    for s in (" " + w, w, " " + w.capitalize()):
                        e = tok.encode(s, add_special_tokens=False)
                        if len(e) == 1:
                            lex[e[0]] = k
            cls_names = ["Past", "Pres"]
            lab_to_k = {"Past": 0, "Pres": 1}
        lut = np.full(max(int(top.max()), max(lex)) + 1, -1)
        for t, k in lex.items():
            lut[t] = k
        tc = lut[top]  # [n,10]
        F = np.stack([(tc == k).mean(1) for k in range(len(cls_names))], 1)
        yk = np.array([lab_to_k[classes[v]] for v in y])
        ks = sorted(set(yk.tolist()))
        stat2 = ovr_auc(F, yk, ks)
        null2 = [ovr_auc(F, rng.permutation(yk), ks) for _ in range(200)]
        per = {cls_names[k]: float(roc_auc_score(yk == k, F[:, k])) for k in ks if F[:, k].std() > 0 and 0 < (yk == k).sum() < len(yk)}
        cover = float((tc >= 0).mean())
        bin_ = None
        if fname == "pred_pos":
            m = np.isin(yk, [POS_CLASSES.index("ADJ"), POS_CLASSES.index("NOUN")])
            sc2 = F[m, POS_CLASSES.index("ADJ")] - F[m, POS_CLASSES.index("NOUN")]
            yb = yk[m] == POS_CLASSES.index("ADJ")
            bin_ = {"adj_vs_noun_auroc": float(roc_auc_score(yb, sc2)),
                    "null_band95": band([float(roc_auc_score(rng.permutation(yb), sc2)) for _ in range(200)])}
        out["token_class_functional"] = {"stat_ovr_auroc": stat2, "null_band95": band(null2),
                                         "j_readable": bool(stat2 > band(null2)[1]), "per_class": per,
                                         "frac_top10_tokens_in_lexicon": cover, "adj_vs_noun": bin_,
                                         "lexicon_size": len(lex)}
    print(json.dumps(out, default=float)[:2000], flush=True)
    save_json(out, f"{OUT}/followups/{fname}_{lens}.json")
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main(features: str = "col,col_report,tense,lang,pred_pos", lenses: str = "np,cb") -> None:
    jobs = [(f, l) for f in features.split(",") for l in lenses.split(",")]
    for r in followup.starmap(jobs, return_exceptions=True):
        print(r if not isinstance(r, dict) else (r["feature"], r["lens"], r["enrichment"], r["label_token_functional"], r.get("token_class_functional")))
