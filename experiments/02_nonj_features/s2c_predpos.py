"""Step 2c: the model's PREDICTED part of speech of the next word (lead's NJ3 ground truth).

Full forward of Qwen3.6-27B on every `prose` item; at every position the next-token distribution
is mapped onto POS classes with a corpus lexicon (word-initial tokens whose next-word UPOS in our
spaCy-tagged corpus is >= 80% one class, >= 3 occurrences). Saves, aligned with acts/prose rows:
  class_mass [N, 8] (POS_CLASSES order), top1 id, top1 prob   -> acts/prose/pred.npz

    modal run s2c_predpos.py
"""

import modal

from nonj_common import OUT, VOLUMES, nonj_image

app = modal.App("nonj-s2c-predpos")


@app.function(image=nonj_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def predpos() -> dict:
    import collections
    import pickle

    import numpy as np
    import torch

    from features import POS_CLASSES
    from nonj_lib import load_model, save_json

    VOLUMES["/results"].reload()
    items = pickle.load(open(f"{OUT}/data/prose.pkl", "rb"))
    index = dict(np.load(f"{OUT}/acts/prose/index.npz"))
    cnt = collections.defaultdict(collections.Counter)
    for it in items:
        for j, lab in enumerate(it["next_pos"]):
            if lab and j + 1 < len(it["ids"]):
                cnt[it["ids"][j + 1]][lab] += 1
    lex = {}
    for t, c in cnt.items():
        tot = sum(c.values())
        lab, k = c.most_common(1)[0]
        if tot >= 3 and k / tot >= 0.8 and lab in POS_CLASSES:
            lex[t] = POS_CLASSES.index(lab)
    tok, model = load_model()
    V = model.lm_head.weight.shape[0]
    M = torch.zeros(V, len(POS_CLASSES), device="cuda")
    for t, k in lex.items():
        M[t, k] = 1.0
    N = len(index["item"])
    mass = np.zeros((N, len(POS_CLASSES)), np.float32)
    top1 = np.zeros(N, np.int32)
    p1 = np.zeros(N, np.float32)
    # rows are laid out by s2 in length-sorted batch order; map (item,pos) -> row
    row_of = {(int(i), int(p)): r for r, (i, p) in enumerate(zip(index["item"], index["pos"]))}
    for i, it in enumerate(items):
        ids = torch.tensor(it["ids"], device="cuda")[None]
        with torch.no_grad():
            pr = torch.softmax(model(input_ids=ids).logits[0].float(), -1)
        cm = (pr @ M).cpu().numpy()
        tp = pr.max(1)
        rr = [row_of[(i, p)] for p in range(len(it["ids"]))]
        mass[rr], top1[rr], p1[rr] = cm, tp.indices.cpu().numpy(), tp.values.cpu().numpy()
        if i % 200 == 0:
            print(i, flush=True)
    np.savez(f"{OUT}/acts/prose/pred.npz", class_mass=mass, top1=top1, p_top1=p1)
    info = {"lexicon_size": len(lex), "lexicon_per_class": collections.Counter(POS_CLASSES[k] for k in lex.values()),
            "mean_lexicon_mass": float(mass.sum(1).mean())}
    save_json(info, f"{OUT}/acts/prose/pred_meta.json")
    VOLUMES["/results"].commit()
    print(info)
    return info


@app.local_entrypoint()
def main() -> None:
    print(predpos.remote())
