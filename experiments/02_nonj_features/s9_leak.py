"""Step 9: leak pre-check. Steer along the J-REMOVED feature direction at J-lens layer 38 (output of
block 38 = hs[39]) and ask whether the J-lens readout at layers 40/42/44 starts to show the
feature value (label tokens / number tokens move up), vs the unsteered run and equal-norm random
directions. Also reports how much the top-10 readout changes at all.

Direction at L38 is computed here (activations at hs[39] are not cached): covariance direction
(counts) or class-mean difference (classes) on 250 training items, then the top-r J-space of
M_38 = W_U diag(g) J_38 (neuronpedia n1000) is projected out, r in {r*, 4r*} from rstar.json.
Dose = p99 and 2x p99 of the natural |projection - median| at hs[39].

    modal run s9_leak.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s9-leak")

READ_L = [40, 42, 44]
STEER_L = 38
WORDS = {
    "pred_pos": (["adjective", "adjectives", "adj", "形容词"], ["noun", "nouns", "名词"]),
    "tense": (["past", "preterite", "过去"], ["present", "现在"]),
    "lang": (["Russian", "俄语", "俄罗斯"], ["Polish", "波兰语", "波兰"]),
}
NUMS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
        "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"] + [str(i) for i in range(10)]


def _ids(tok, words: list[str]) -> list[int]:  # noqa: ANN001
    out = set()
    for w in words:
        for s in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                out.add(e[0])
    return sorted(out)


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=196608)
def leak(fname: str, n_train: int = 250, n_eval: int = 60) -> dict:
    import glob
    import json
    import pickle

    import numpy as np
    import torch

    from features import FEATURES_BY_NAME, rows_for
    from nonj_lib import decoder_blocks, final_norm_module, load_jlens_layers, load_model, save_json

    VOLUMES["/results"].reload()
    feat = FEATURES_BY_NAME[fname]
    items = pickle.load(open(f"{OUT}/data/{feat.dataset}.pkl", "rb"))
    index = dict(np.load(f"{OUT}/acts/{feat.dataset}/index.npz"))
    if fname == "pred_pos":
        index["class_mass"] = np.load(f"{OUT}/acts/prose/pred.npz")["class_mass"]
    R = rows_for(feat, items, index, np.random.default_rng(0), cap_per_class=10**9, cap_reg=10**9)
    lab_of = {(int(index["item"][r]), int(index["pos"][r])): y for r, y in zip(R["rows"], R["y"])}
    item_ids = sorted({k[0] for k in lab_of})
    rng = np.random.default_rng(1)
    rng.shuffle(item_ids)
    tr_items, ev_items = item_ids[:n_train], item_ids[n_train:n_train + n_eval]

    tok, model = load_model()
    blocks = decoder_blocks(model)
    fnorm = final_norm_module(model)
    W_U = model.lm_head.weight
    lens_path = glob.glob(f"{OUT}/lenses/np/**/*.pt", recursive=True)[0]
    Js = {L: J.cuda() for L, J in load_jlens_layers(lens_path, READ_L).items()}
    basis = torch.load(f"{OUT}/jspace_basis_L{STEER_L}.pt")["evecs_top4096"].float().cuda()
    rstar = json.load(open(f"{OUT}/rstar.json"))["layers"]["42"]["r_star"] or 256
    r_list = [int(rstar), int(4 * rstar)]

    cap: dict = {}
    state: dict = {"vec": None}

    def steer_hook(_m, _i, out):  # noqa: ANN001, ANN202
        cap[STEER_L] = (out[0] if isinstance(out, tuple) else out).detach()
        if state["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["vec"].to(out[0].dtype),) + tuple(out[1:])
        return out + state["vec"].to(out.dtype)

    hooks = [blocks[STEER_L].register_forward_hook(steer_hook)]
    for L in READ_L:
        hooks.append(blocks[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())))

    # ---- direction at hs[39]
    Xs, ys = [], []
    for i in tr_items:
        ids = torch.tensor(items[i]["ids"], device="cuda")[None]
        with torch.no_grad():
            model.model(input_ids=ids)
        ps = [p for p in range(len(items[i]["ids"])) if (i, p) in lab_of]
        Xs.append(cap[STEER_L][0, ps].float())
        ys += [lab_of[(i, p)] for p in ps]
    X = torch.cat(Xs)
    y = torch.tensor(np.array(ys), device="cuda")
    if feat.kind == "reg":
        yc = y.float() - y.float().mean()
        u = ((X - X.mean(0)) * yc[:, None]).mean(0)
    else:
        a, b = R["classes"].index(feat.steer_pair[0]), R["classes"].index(feat.steer_pair[1])
        u = X[y == a].mean(0) - X[y == b].mean(0)
    u = u / u.norm()
    proj = (X @ u).cpu().numpy()
    p99 = float(np.percentile(np.abs(proj - np.median(proj)), 99))
    med_norm = float(X.norm(dim=1).median())
    dirs = {}
    for r in r_list:
        Q = basis[:, :r]
        ur = u - Q @ (Q.T @ u)
        dirs[f"jrem{r}"] = ur / ur.norm()
    dirs["full"] = u
    g = torch.Generator("cuda").manual_seed(9)
    for k in range(2):
        v = torch.randn(u.shape[0], device="cuda", generator=g)
        dirs[f"rand{k}"] = v / v.norm()

    pos_ids = torch.tensor(_ids(tok, WORDS[fname][0]) if fname in WORDS else _ids(tok, NUMS), device="cuda")
    neg_ids = torch.tensor(_ids(tok, WORDS[fname][1]), device="cuda") if fname in WORDS else None

    def readout(ps: list[int]) -> dict:
        out = {}
        for L in READ_L:
            Z = fnorm((cap[L][0, ps].float() @ Js[L].T).to(torch.bfloat16))
            sc = (Z @ W_U.T).float()
            srt = sc.sort(dim=1).values
            V = sc.shape[1]

            def rank(ids_):  # noqa: ANN001, ANN202
                return (V - torch.searchsorted(srt, sc[:, ids_].contiguous(), right=True)).float()

            rp = rank(pos_ids)
            out[L] = {"pos_logrank": torch.log10(rp.min(1).values + 1), "pos_top10": (rp.min(1).values < 10).float(),
                      "top10": sc.topk(10, dim=1).indices}
            if neg_ids is not None:
                out[L]["neg_logrank"] = torch.log10(rank(neg_ids).min(1).values + 1)
        return out

    conds = [("none", None, 0.0)]
    for dn, mult in (("p99", 1.0), ("2xp99", 2.0)):
        for nm, d in dirs.items():
            for sgn in (+1, -1):
                conds.append((f"{nm}{'+' if sgn > 0 else '-'}{dn}", d, sgn * mult * p99))
    acc = {c[0]: {L: {"pos_logrank": [], "neg_logrank": [], "pos_top10": [], "top10_overlap": []} for L in READ_L} for c in conds}
    for i in ev_items:
        ps = [p for p in range(len(items[i]["ids"])) if (i, p) in lab_of]
        ids = torch.tensor(items[i]["ids"], device="cuda")[None]
        base = None
        for name, d, alpha in conds:
            state["vec"] = None if d is None else alpha * d
            with torch.no_grad():
                model.model(input_ids=ids)
            ro = readout(ps)
            if name == "none":
                base = ro
            for L in READ_L:
                a_ = acc[name][L]
                a_["pos_logrank"].append(ro[L]["pos_logrank"].mean().item())
                a_["pos_top10"].append(ro[L]["pos_top10"].mean().item())
                if neg_ids is not None:
                    a_["neg_logrank"].append(ro[L]["neg_logrank"].mean().item())
                ov = [len(set(x.tolist()) & set(z.tolist())) / 10 for x, z in zip(ro[L]["top10"], base[L]["top10"])]
                a_["top10_overlap"].append(float(np.mean(ov)))
        state["vec"] = None
    for h in hooks:
        h.remove()
    res: dict = {"feature": fname, "steer_layer_jlens": STEER_L, "steer_hs_index": STEER_L + 1, "read_layers_jlens": READ_L,
                 "r_list": r_list, "p99_absdev": p99, "median_resid_norm": med_norm, "n_eval_items": len(ev_items),
                 "target_tokens": [tok.decode([t]) for t in pos_ids.tolist()], "conditions": {}}
    for name, _, alpha in conds:
        res["conditions"][name] = {"alpha": alpha, "alpha_over_resid_norm": abs(alpha) / med_norm,
                                   **{str(L): {k: (float(np.mean(v)) if v else None) for k, v in acc[name][L].items()} for L in READ_L}}
    # summary: shift in target-label log-rank (negative = target labels moved UP the readout)
    summ = {}
    for L in READ_L:
        b = res["conditions"]["none"][str(L)]
        s = {}
        for name in res["conditions"]:
            if name == "none":
                continue
            c = res["conditions"][name][str(L)]
            s[name] = {"d_pos_logrank": c["pos_logrank"] - b["pos_logrank"], "pos_top10": c["pos_top10"],
                       "top10_overlap_with_unsteered": c["top10_overlap"]}
            if c["neg_logrank"] is not None:
                s[name]["d_neg_logrank"] = c["neg_logrank"] - b["neg_logrank"]
        s["baseline_pos_top10"] = b["pos_top10"]
        s["baseline_pos_logrank"] = b["pos_logrank"]
        summ[L] = s
    res["summary"] = summ
    save_json(res, f"{OUT}/leak/{fname}.json")
    VOLUMES["/results"].commit()
    print(fname, json.dumps({L: {k: v for k, v in s.items() if "2xp99" in k or k.startswith("baseline")} for L, s in summ.items()}, default=float)[:3000], flush=True)
    return res


@app.local_entrypoint()
def main(features: str = "col,depth,pred_pos,tense,lang,col_report") -> None:
    for r in leak.map(features.split(","), return_exceptions=True):
        print(r if not isinstance(r, dict) else r["feature"])
