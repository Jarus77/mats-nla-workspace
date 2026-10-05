"""04 backup features: data -> capture -> gate (G1, G2-geometric, G2-functional, secondary J-only).
G4 causal + leak + self-report live in bf_causal.py.

Layer convention: layer L = output of decoder block L = HF hidden_states[L+1] (hook on blocks[L]).
Primary L42; 40/44 sensitivity; L38 = steering layer.

    cd experiments/04_backup_features
    modal run bf_modal.py --stage data
    modal run bf_modal.py --stage capture
    modal run bf_modal.py --stage gate
"""

from __future__ import annotations

import modal

from env import OUT, VOLUMES, gpu_image, net_image

app = modal.App("mats-04-backup-features")
LAYERS = [38, 40, 42, 44]
RANKS = {38: [192, 384, 768, 1536], 40: [384, 1536], 42: [384, 1536], 44: [256, 1024]}
WIKI = "/results/02_nonj_features/data/raw_wikitext103_train.parquet"  # read-only reuse


def _wikitext_paragraphs(min_chars: int, seed: int) -> list[str]:
    """Same cleaning as 02 s1_data._wikitext_paragraphs (copied; 02 is read-only)."""
    import random
    import re

    import pandas as pd

    paras = []
    for t in pd.read_parquet(WIKI)["text"].tolist():
        t = t.strip()
        if len(t) < min_chars or t.startswith("="):
            continue
        t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        t = re.sub(r" ([.,;:!?)\]'])", r"\1", t)
        t = re.sub(r"([(\[]) ", r"\1", t)
        t = t.replace(" n't", "n't").replace(" 's", "'s").replace('" ', '"', 1)
        if "<unk>" in t:
            continue
        paras.append(t)
    random.Random(seed).shuffle(paras)
    return paras


def _humanevalpack() -> dict[str, list[dict]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out = {}
    for lang in ("python", "js"):
        p = hf_hub_download("bigcode/humanevalpack", f"{lang}/test-00000-of-00001.parquet", repo_type="dataset")
        out[lang] = pd.read_parquet(p).to_dict("records")
    return out


@app.function(image=net_image, volumes=VOLUMES, timeout=3600, memory=32768, cpu=8,
              secrets=[modal.Secret.from_name("hf-token")])
def stage_data(datasets: list[str]) -> dict:
    import json
    import os
    import pickle

    from datasets import load_dataset
    from transformers import AutoTokenizer

    import bf_items as BI
    from bf_feats import FEATURES

    VOLUMES["/results"].reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    paras = _wikitext_paragraphs(600, seed=4)
    data: dict = {}
    if "prose" in datasets:
        data["prose"] = BI.build_prose(tok, paras[:1200], seed=0)
    if "list_num" in datasets:
        long_paras = [p for p in paras[1200:] if len(BI.split_sentences(p)) >= 6][:900]
        data["list_num"] = BI.build_lists(tok, long_paras, numbered=True)
    if "code_pair" in datasets or "code_py" in datasets:
        hep = _humanevalpack()
        recs = [{"task": int(r["task_id"].split("/")[-1]), "lang": lang, "prompt": r["prompt"],
                 "solution": r["canonical_solution"]} for lang, rows in hep.items() for r in rows]
        data["code_pair"] = BI.build_code_pair(tok, recs)
        mbpp = []
        for split in ("train", "test", "validation", "prompt"):
            mbpp += [r["code"].replace("\r\n", "\n") for r in
                     load_dataset("google-research-datasets/mbpp", "full", split=split)]
        data["code_py"] = BI.build_code_py(tok, mbpp + [r["prompt"] + r["canonical_solution"] for r in hep["python"]])
    if "dialogue" in datasets:
        ts = load_dataset("roneneldan/TinyStories", split="validation")
        data["dialogue"] = BI.build_dialogue(tok, [r["text"] for r in ts.select(range(6000))])
    if "french" in datasets:
        import spacy

        nlp = spacy.load("fr_core_news_md")
        wp = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
        fr_paras = []
        for art in wp:
            for para in art["text"].split("\n"):
                para = para.strip()
                if 300 <= len(para) <= 1000 and para.count(" ") > 40:
                    fr_paras.append(para)
            if len(fr_paras) >= 3000:
                break
        data["french"], lex = BI.build_french(tok, nlp, fr_paras)
        os.makedirs(f"{OUT}/data", exist_ok=True)
        json.dump({str(k): [v["Fem"], v["Masc"]] for k, v in lex.items()}, open(f"{OUT}/data/gender_lex.json", "w"))
    for f in FEATURES:
        if f.position_matched and f.dataset in data:
            data[f.dataset] = BI.position_match(data[f.dataset], f.name)
    os.makedirs(f"{OUT}/data", exist_ok=True)
    summary = {}
    for name, items in data.items():
        pickle.dump(items, open(f"{OUT}/data/{name}.pkl", "wb"))
        summary[name] = {"items": len(items)}
    for f in FEATURES:
        if f.dataset not in data:
            continue
        its = [it for it in data[f.dataset] if f.name in it["feats"]]
        ys = [y for it in its for y in it["feats"][f.name]["y"]]
        summary[f.name] = {"items": len(its), "rows": len(ys), "pos_frac": sum(ys) / max(len(ys), 1),
                           "eval_pos": sum(len(it["eval"][f.name]) for it in its)}
        ex = its[0]
        summary[f"example_{f.name}"] = {
            "read": [(repr(tok.decode(ex["ids"][max(0, p - 3):p + 1])), y) for p, y in
                     zip(ex["feats"][f.name]["pos"][:6], ex["feats"][f.name]["y"][:6])],
            "eval_next": [repr(tok.decode(ex["ids"][p:p + 2])) for p in ex["eval"][f.name][:4]]}
    print(json.dumps(summary, indent=1, default=str), flush=True)
    json.dump(summary, open(f"{OUT}/data/summary_{'_'.join(sorted(data))}.json", "w"), indent=1, default=str)
    VOLUMES["/results"].commit()
    return summary


# ============================================================================ capture


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def stage_capture(dataset: str) -> dict:
    import os
    import pickle

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from mats.capture import decoder_blocks

    VOLUMES["/results"].reload()
    items = pickle.load(open(f"{OUT}/data/{dataset}.pkl", "rb"))
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    blocks = decoder_blocks(model)
    del blocks[max(LAYERS) + 1:]
    pos = [sorted({p for v in it["feats"].values() for p in v["pos"]}) for it in items]
    cur: dict = {}

    class Stop(Exception):
        pass

    def mk(L: int):  # noqa: ANN202
        def hook(_m, _i, o):  # noqa: ANN001, ANN202
            cur[L] = (o[0] if isinstance(o, tuple) else o).detach()
            if L == max(LAYERS):
                raise Stop
        return hook

    hs = [blocks[L].register_forward_hook(mk(L)) for L in LAYERS]
    acts = {L: [] for L in LAYERS}
    order = np.argsort([len(it["ids"]) for it in items])
    idx_item, idx_pos = [], []
    bs = 16
    for s in range(0, len(order), bs):
        rows = order[s:s + bs]
        n = max(len(items[r]["ids"]) for r in rows)
        ids = torch.zeros(len(rows), n, dtype=torch.long)
        mask = torch.zeros(len(rows), n, dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, :len(items[r]["ids"])] = torch.tensor(items[r]["ids"])
            mask[b, :len(items[r]["ids"])] = 1
        with torch.no_grad():
            try:
                model(input_ids=ids.cuda(), attention_mask=mask.cuda(), logits_to_keep=1)
            except Stop:
                pass
        for b, r in enumerate(rows):
            for L in LAYERS:
                acts[L].append(cur[L][b, pos[r]].float().cpu().half())
            idx_item += [int(r)] * len(pos[r])
            idx_pos += pos[r]
    for h in hs:
        h.remove()
    d = f"{OUT}/acts/{dataset}"
    os.makedirs(d, exist_ok=True)
    for L in LAYERS:
        np.save(f"{d}/L{L}.npy", torch.cat(acts[L]).numpy())
    np.savez(f"{d}/index.npz", item=np.array(idx_item), pos=np.array(idx_pos))
    VOLUMES["/results"].commit()
    return {"dataset": dataset, "rows": len(idx_item)}


# ============================================================================ gate


def _load_lens_and_unembed() -> tuple:
    import glob
    import json

    import torch
    from safetensors import safe_open

    from mats.jspace import load_jlens_layers

    try:
        from huggingface_hub import hf_hub_download

        from mats.readers.jlens import FILE, REPO
        path = hf_hub_download(REPO, FILE)
    except Exception:  # noqa: BLE001
        path = glob.glob("/results/02_nonj_features/lenses/np/**/*n1000*.pt", recursive=True)[0]
    Js = {L: m.cuda() for L, m in load_jlens_layers(path, [40, 42, 44]).items()}
    snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    widx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    ukey = [k for k in widx if k.endswith("lm_head.weight")][0]
    nkey = [k for k in widx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k and "mtp" not in k][0]

    def get(k: str):  # noqa: ANN202
        with safe_open(f"{snap}/{widx[k]}", framework="pt") as f:
            return f.get_tensor(k)

    W_U = get(ukey).cuda().float()
    gain = 1 + get(nkey).float().cuda()  # Qwen3_5RMSNorm uses (1 + weight)
    return Js, W_U, gain, path


def _load_cb_lens(layer: int = 42):  # noqa: ANN202
    from huggingface_hub import hf_hub_download

    from mats.jspace import load_jlens_layers

    p = hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt")
    return load_jlens_layers(p, [layer])[layer].cuda()


def feature_rows(fname: str) -> tuple:
    """(rows into the dataset acts, y, groups, item index, pos) for a feature."""
    import pickle

    import numpy as np

    from bf_feats import FEATURES_BY_NAME

    f = FEATURES_BY_NAME[fname]
    items = pickle.load(open(f"{OUT}/data/{f.dataset}.pkl", "rb"))
    idx = np.load(f"{OUT}/acts/{f.dataset}/index.npz")
    lut = {(int(i), int(p)): r for r, (i, p) in enumerate(zip(idx["item"], idx["pos"]))}
    rows, y, g, it_, ps = [], [], [], [], []
    for k, it in enumerate(items):
        if fname not in it["feats"]:
            continue
        for p, yy in zip(it["feats"][fname]["pos"], it["feats"][fname]["y"]):
            rows.append(lut[(k, p)])
            y.append(yy)
            g.append(it["group"])
            it_.append(k)
            ps.append(p)
    return items, np.array(rows), np.array(y), np.array(g), np.array(it_), np.array(ps)


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=131072)
def stage_gate(fname: str) -> dict:
    import json
    import os

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    import bf_gate_lib as G
    from bf_feats import FEATURES_BY_NAME
    from bf_vocab import emitted_sets
    from mats.jspace import jspace_basis, load_basis_file
    from mats.readers.jlens import token_variants

    VOLUMES["/results"].reload()
    f = FEATURES_BY_NAME[fname]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    items, rows, y, groups, item_k, pos = feature_rows(fname)
    out: dict = {"feature": fname, "description": f.description, "n_rows": int(len(rows)),
                 "n_items": int(len(np.unique(item_k))), "n_groups": int(len(np.unique(groups))),
                 "pos_frac": float(y.mean()), "layers": {}}
    X = {L: torch.from_numpy(np.load(f"{OUT}/acts/{f.dataset}/L{L}.npy", mmap_mode="r")[rows].astype(np.float32)).cuda()
         for L in LAYERS}
    bases = {L: load_basis_file(f"/results/03_pipeline_smoke/jspace_basis_L{L}.pt", "cuda") for L in LAYERS}
    Js, W_U, gain, lens_path = _load_lens_and_unembed()
    out["lens_path"] = lens_path
    lab_ids = [token_variants(tok, list(ws)) for ws in f.label_words]
    vocab = [tok.decode([i]) for i in range(W_U.shape[0])]
    glex = json.load(open(f"{OUT}/data/gender_lex.json")) if fname == "gender_next" else None
    sets = emitted_sets(fname, vocab, tok, glex)
    out["emitted_sets_size"] = [len(sets[0]), len(sets[1])]
    out["emitted_sets_examples"] = [[vocab[i] for i in sets[c][:15]] for c in (0, 1)]
    out["label_tokens"] = [[tok.decode([i]) for i in ids] for ids in lab_ids]
    # text baselines: absolute position; bag of the last 4 tokens
    out["baseline_abs_position_auroc"] = G.auroc(pos[y == 1].astype(float), pos[y == 0].astype(float))
    loc = np.stack([np.array([items[k]["ids"][p - o] if p - o >= 0 else -1 for k, p in zip(item_k, pos)])
                    for o in range(4)], 1)
    out["baseline_local4_bag"] = G.bag_auroc(loc, y, groups)
    for L in LAYERS:
        Qs = {r: bases[L].Q(r) for r in RANKS[L]}
        res: dict = {"G1": G.probe_report(X[L], y, groups),
                     "geometric": G.geometric(X[L], y, Qs, with_inlp=(L == 42))}
        if L in Js:
            rd = G.jlens_readout(X[L], Js[L], W_U, gain, lab_ids, sets)
            res["functional"] = G.functional_stat(rd["ranks"], y)
            res["emitted_soft"] = G.emitted_stat(rd["soft"], y)
            res["emitted_top10"] = G.emitted_stat(rd["top10"], y)
            res["lens_bag_top25"] = G.bag_auroc(rd["top"], y, groups)
            if L == 42:
                top = rd["top"]
                vals, cnt = np.unique(top[:, :10], return_counts=True)
                res["lens_top10_common"] = [(tok.decode([int(v)]), int(c)) for v, c in
                                            sorted(zip(vals, cnt), key=lambda t: -t[1])[:25]]
                # cross-check lens: camilablank/workspace-lenses (25 prompts), own J-space basis
                Jcb = _load_cb_lens()
                bcb = jspace_basis(Jcb, W_U, gain, r_max=1536, layer=42)
                rcb = G.jlens_readout(X[L], Jcb, W_U, gain, lab_ids, sets)
                res["camilablank"] = {
                    "geometric": G.geometric(X[L], y, {r: bcb.Q(r) for r in (384, 1536)}, n_perm=3),
                    "basis_overlap_with_n1000_r384": float((bcb.Q(384).T @ bases[42].Q(384)).pow(2).sum() / 384),
                    "functional": G.functional_stat(rcb["ranks"], y),
                    "emitted_soft": G.emitted_stat(rcb["soft"], y),
                    "emitted_top10": G.emitted_stat(rcb["top10"], y)}
                del Jcb, bcb
        if L == 42:
            res["jonly"] = {str(r): G.jonly(X[L], y, groups, Qs[r]) for r in (384, 1536)}
        out["layers"][str(L)] = res
        print(fname, L, {k: v for k, v in res.items() if k in ("G1", "functional")},
              {r: round(v["diffmeans"], 2) for r, v in res["geometric"].items()}, flush=True)
    # steering directions for G4 (2-fold cross-fit by group) at L38 and L42
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    fold_of_group = {int(gq): i % 2 for i, gq in enumerate(ug)}
    fold = np.array([fold_of_group[int(gq)] for gq in groups])
    dirs: dict = {"fold_of_group": fold_of_group}
    for L, rstar in ((38, 192), (42, 384)):
        med_norm = float(X[L].norm(dim=1).median())
        for fo in (0, 1):
            tr = fold != fo
            w = G.mean_diff(X[L][tr], y[tr])
            dd: dict = {"raw": w.cpu()}
            for r in sorted({rstar, 1536}):
                Q = bases[L].Q(r)
                u = w - (w @ Q) @ Q.T
                dd[f"jrem{r}"] = (u / u.norm()).cpu()
            doses = {}
            for name, u in list(dd.items()):
                proj = (X[L][tr] @ u.cuda()).cpu().numpy()
                ad = np.abs(proj - np.median(proj))
                doses[name] = {"p90": float(np.percentile(ad, 90)), "p99": float(np.percentile(ad, 99))}
            dirs[(L, fo)] = {"dirs": dd, "doses": doses, "median_norm": med_norm}
    os.makedirs(f"{OUT}/gate", exist_ok=True)
    torch.save(dirs, f"{OUT}/gate/{fname}_dirs.pt")
    json.dump(out, open(f"{OUT}/gate/{fname}.json", "w"), indent=1, default=float)
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main(stage: str = "data", only: str = "") -> None:
    import json

    from bf_feats import DATASETS, FEATURES

    if stage == "data":
        print(json.dumps(stage_data.remote(only.split(",") if only else DATASETS), indent=1, default=str))
    elif stage == "capture":
        ds = only.split(",") if only else DATASETS
        for r in stage_capture.map(ds, return_exceptions=True):
            print(r)
    elif stage == "gate":
        fs = only.split(",") if only else [f.name for f in FEATURES]
        for r in stage_gate.map(fs, return_exceptions=True):
            print(r if not isinstance(r, dict) else (r["feature"], "ok"))
