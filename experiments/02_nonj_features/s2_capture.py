"""Step 2: capture residuals hs[40..45] for every dataset, J-lens readouts and k=25 NN gradient
pursuit (J-space decomposition) at hs[43] (J-lens L42) and hs[42] (J-lens L41).

Outputs under /results/02_nonj_features/acts/<dataset>/:
  hs{i}.npy          fp16 [N, 5120]   residual at hidden_states index i (= block i-1 output)
  index.npz          item, pos (token index), token id, next token id
  lens_hs{i}.npz     top50 ids (standard readout softmax(W_U norm(J h))), top50 (cosine readout),
                     label-token ranks [N, n_label] (standard), next-token rank (standard)
  jsp_hs{i}.npz      pursuit idx [N,25] int32, coef [N,25] fp16, var explained by J-space comp.

    modal run s2_capture.py --datasets wrap,prose,brackets,topic --lens np   (np = neuronpedia n1000, primary; cb = camilablank n25)
"""

import modal

from nonj_common import LENS_FILE_CB, LENS_REPO_CB, OUT, VOLUMES, nonj_image

app = modal.App("nonj-s2-capture")

LABEL_WORDS = [
    # part of speech
    "noun", "nouns", "verb", "verbs", "adjective", "adjectives", "adj", "adverb", "adverbs",
    "preposition", "prepositions", "determiner", "determiners", "article", "pronoun", "pronouns",
    "conjunction", "punctuation", "proper", "auxiliary", "participle", "infinitive", "gerund",
    "名词", "动词", "形容词", "副词", "介词",
    # tense
    "past", "present", "future", "tense", "preterite", "过去", "现在", "将来", "historical", "ago",
    # topic
    "world", "politics", "international", "war", "government", "sports", "sport", "game",
    "football", "baseball", "business", "economy", "finance", "market", "markets", "stocks",
    "technology", "tech", "science", "scientific", "computer", "software", "internet",
    # counting / structure
    "characters", "character", "chars", "length", "width", "column", "line", "count", "number",
    "depth", "nested", "nesting", "level", "brackets", "bracket", "open", "close",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred",
    # language identity (IJ2)
    "language", "Arabic", "Greek", "Polish", "Russian", "Turkish", "Vietnamese",
    "阿拉伯语", "阿拉伯", "希腊语", "希腊", "波兰语", "波兰", "俄语", "俄罗斯", "土耳其语", "土耳其", "越南语", "越南",
]


def label_tokens(tok) -> list[tuple[str, int, str]]:  # noqa: ANN001
    """(word, token id, surface) for single-token variants of LABEL_WORDS and digits 0-9."""
    out, seen = [], set()
    words = LABEL_WORDS + [str(d) for d in range(10)]
    for w in words:
        for surf in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tok.encode(surf, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                out.append((w, ids[0], surf))
    return out


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=6 * 3600, memory=262144)
def capture(dataset: str, lens: str = "np", batch: int = 8) -> dict:
    import glob
    import os
    import pickle
    import time

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    from nonj_lib import (HS_LAYERS, decoder_blocks, final_norm_module, jlens_atoms, load_jlens_layers,
                          load_model, nn_pursuit, norm_gain, save_json)

    VOLUMES["/results"].reload()
    t0 = time.time()
    items = pickle.load(open(f"{OUT}/data/{dataset}.pkl", "rb"))
    tok, model = load_model(max_layers=max(HS_LAYERS))  # hs[45] = output of block 44
    blocks = decoder_blocks(model)
    W_U = model.lm_head.weight.detach()  # bf16 [V,d]
    gain = norm_gain(model)
    fnorm = final_norm_module(model)
    lens_path = (hf_hub_download(LENS_REPO_CB, LENS_FILE_CB) if lens == "cb"
                 else glob.glob(f"{OUT}/lenses/np/**/*.pt", recursive=True)[0])
    lens_hs = {43: 42, 42: 41}  # hs index -> J-lens (block-output) index
    Js = {h: J.cuda() for h, J in zip(lens_hs, load_jlens_layers(lens_path, list(lens_hs.values())).values())}
    atoms = {}
    for h, J in Js.items():
        a, n = jlens_atoms(J, W_U, gain)
        atoms[h] = (a, n)
    labs = label_tokens(tok)
    lab_ids = torch.tensor([x[1] for x in labs], device="cuda")
    print(f"loaded in {time.time()-t0:.0f}s; {len(items)} items; {len(labs)} label tokens", flush=True)

    d = f"{OUT}/acts/{dataset}" + ("" if lens == "np" else f"_lens{lens}")
    os.makedirs(d, exist_ok=True)
    N = sum(len(it["ids"]) for it in items)
    mm = {h: np.lib.format.open_memmap(f"{d}/hs{h}.npy", mode="w+", dtype=np.float16, shape=(N, 5120))
          for h in HS_LAYERS}
    item_ix, pos_ix, tid, ntid = (np.zeros(N, np.int32) for _ in range(4))
    lens_out = {h: {"top_std": np.zeros((N, 50), np.int32), "top_cos": np.zeros((N, 50), np.int32),
                    "lab_rank": np.zeros((N, len(labs)), np.int32), "next_rank": np.zeros(N, np.int32)}
                for h in lens_hs}
    jsp = {h: {"idx": np.zeros((N, 25), np.int32), "coef": np.zeros((N, 25), np.float16),
               "fve": np.zeros(N, np.float32), "norm": np.zeros(N, np.float32)} for h in lens_hs}

    cap: dict = {}
    hooks = []
    for h in HS_LAYERS:
        def mk(hh: int):  # noqa: ANN202
            def hook(_m, _i, out):  # noqa: ANN001, ANN202
                cap[hh] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook
        hooks.append(blocks[h - 1].register_forward_hook(mk(h)))
    cos_den = {h: None for h in lens_hs}

    row = 0
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for bi in range(0, len(order), batch):
        idxs = order[bi:bi + batch]
        L = max(len(items[i]["ids"]) for i in idxs)
        ids = torch.full((len(idxs), L), tok.pad_token_id or 0, dtype=torch.long)
        for r, i in enumerate(idxs):
            ids[r, : len(items[i]["ids"])] = torch.tensor(items[i]["ids"])
        with torch.no_grad():
            model.model(input_ids=ids.cuda())  # right padding: causal model, real positions unaffected
            for r, i in enumerate(idxs):
                n = len(items[i]["ids"])
                sl = slice(row, row + n)
                for h in HS_LAYERS:
                    mm[h][sl] = cap[h][r, :n].float().cpu().numpy().astype(np.float16)
                item_ix[sl], pos_ix[sl] = i, np.arange(n)
                tid[sl] = items[i]["ids"]
                ntid[sl] = items[i]["ids"][1:] + [-1]
                for h, jl in lens_hs.items():
                    H = cap[h][r, :n].float()
                    Z = H @ Js[h].T
                    std = fnorm(Z.to(cap[h].dtype)).float() @ W_U.float().T  # [n,V]
                    if cos_den[h] is None:  # WorkspaceBench "magnitude-free cosine" denominator
                        cos_den[h] = (W_U.float() @ Js[h]).norm(dim=1).clamp_min(1e-9)
                    cos = (Z @ W_U.float().T) / cos_den[h]
                    lo = lens_out[h]
                    lo["top_std"][sl] = std.topk(50, dim=1).indices.cpu().numpy()
                    lo["top_cos"][sl] = cos.topk(50, dim=1).indices.cpu().numpy()
                    ls = std[:, lab_ids]
                    srt = std.sort(dim=1).values
                    V = srt.shape[1]
                    lo["lab_rank"][sl] = (V - torch.searchsorted(srt, ls.contiguous(), right=True)).cpu().numpy()
                    nt = torch.tensor(items[i]["ids"][1:] + [items[i]["ids"][-1]], device="cuda")
                    lo["next_rank"][sl] = (std > std.gather(1, nt[:, None])).sum(1).cpu().numpy()
                    pi, pc, rec = nn_pursuit(H, atoms[h][0], k=25)
                    jsp[h]["idx"][sl] = pi.cpu().numpy()
                    jsp[h]["coef"][sl] = pc.cpu().numpy().astype(np.float16)
                    jsp[h]["fve"][sl] = (1 - ((H - rec) ** 2).sum(1) / (H ** 2).sum(1)).cpu().numpy()
                    jsp[h]["norm"][sl] = H.norm(dim=1).cpu().numpy()
                row += n
        if bi % (batch * 20) == 0:
            print(f"{bi}/{len(items)} rows={row} t={time.time()-t0:.0f}s", flush=True)
    for hk in hooks:
        hk.remove()
    for h in HS_LAYERS:
        mm[h].flush()
    np.savez(f"{d}/index.npz", item=item_ix, pos=pos_ix, tid=tid, ntid=ntid)
    for h in lens_hs:
        np.savez(f"{d}/lens_hs{h}.npz", **lens_out[h])
        np.savez(f"{d}/jsp_hs{h}.npz", **jsp[h])
    save_json({"labels": labs, "lens": lens, "lens_path": lens_path, "N": N,
               "lens_hs_to_jlayer": lens_hs}, f"{d}/meta.json")
    VOLUMES["/results"].commit()
    info = {"dataset": dataset, "N": N, "secs": time.time() - t0,
            "jsp_fve_mean_hs43": float(jsp[43]["fve"].mean())}
    print(info, flush=True)
    return info


@app.local_entrypoint()
def main(datasets: str = "wrap,prose,brackets,topic", lens: str = "np") -> None:
    for r in capture.starmap([(d, lens) for d in datasets.split(",")]):
        print(r)
