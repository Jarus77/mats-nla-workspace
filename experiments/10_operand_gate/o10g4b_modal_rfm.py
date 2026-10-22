"""10 operand gate, stage 2a-G4b: positive control for the "=" position (model side only).

Full donor-activation patch at "=" (L42) on the one-digit bank: h_i -> h_donor where the donor item has a = a'
(a' = a + 3 wrapped), the same b and op; raw, and J-complement-only (h_i + (I - QQ^T)(h_donor - h_i)), plus the
J-only part (h_i + QQ^T(h_donor - h_i)). If the raw patch cannot move the answer, nothing at this position/layer can.

    modal deploy o10g4b_modal_rfm.py ; python spawn_driver2.py spawn operand_stage2a_g4b
"""
import json
import time
from pathlib import Path

import modal

from env import BASIS_L42, OUT, VOLUMES, app_image

app = modal.App("rfm")
L = 42


def _load_model():  # noqa: ANN202
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


OUT3 = f"{OUT}/stage2a_g4"
N_GREEDY = 4
CHUNK = 16


@app.function(image=app_image, volumes=VOLUMES, gpu="A100-80GB", max_containers=1, single_use_containers=True,
              retries=1, timeout=3600, memory=131072)
def operand_stage2a_g4b() -> dict:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import numpy as np
    import torch

    import o10g4_lib as GL
    import o10s2_lib as S
    from mats.capture import decoder_blocks
    from mats.jspace import load_basis_file

    t0 = time.time()
    VOLUMES["/results"].reload()
    items = GL.onedigit_items()
    n = len(items)
    A = torch.load(f"{OUT3}/acts_onedigit_hs43.pt", map_location="cpu", weights_only=False)
    H1 = A["H"].cuda().float()
    texts = A["texts"]
    model, tok = _load_model()
    seqs = [tok.encode(t, add_special_tokens=False) for t in texts]
    space_id = tok.encode(" ", add_special_tokens=False)[0]
    dig1 = torch.tensor([tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)], device="cuda")
    Q = load_basis_file(BASIS_L42, "cuda").Q(1536)
    key = {(it["a"], it["b"], it["op"]): i for i, it in enumerate(items)}
    donor, pred = [], []
    for it in items:
        a2 = GL.aprime(it["a"])
        j = key.get((a2, it["b"], it["op"]))
        if j is None:  # "-" item whose a' <= b: no same-op donor; use the "+" item with a = a' and the same b (always exists)
            j = key[(a2, it["b"], "+")]
        donor.append(j)
        pred.append(GL.predicted_answer(a2, it["b"], it["op"]))
    donor = np.array(donor)
    pred = np.array(pred)
    truth = np.array([it["answer"] for it in items])
    same_b = np.array([items[j]["b"] == it["b"] for j, it in zip(donor, items)])
    same_op = np.array([items[j]["op"] == it["op"] for j, it in zip(donor, items)])
    Hd = H1[torch.as_tensor(donor, device="cuda")]
    delta = Hd - H1
    dJ = (delta @ Q) @ Q.T
    COND = {"none": None, "patch_full": Hd, "patch_jcomp": H1 + (delta - dJ), "patch_jonly": H1 + dJ}
    out: dict = {"n": n, "same_b_fraction": float(same_b.mean()), "same_op_fraction": float(same_op.mean()), "displacement_over_norm": {
        "full": float((delta.norm(dim=1) / H1.norm(dim=1)).mean()), "jcomp": float(((delta - dJ).norm(dim=1) / H1.norm(dim=1)).mean()),
        "jonly": float((dJ.norm(dim=1) / H1.norm(dim=1)).mean())}}
    names = list(COND)
    state: dict = {"Hnew": None, "on": None, "p": None}

    def hook(_m, _i, o):  # noqa: ANN001, ANN202
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] == 1:
            return o
        h2 = h.clone()
        h2[:, state["p"]] = torch.where(state["on"][:, None], state["Hnew"], h[:, state["p"]].float()).to(h.dtype)
        return (h2, *o[1:]) if isinstance(o, tuple) else h2

    handle = decoder_blocks(model)[L].register_forward_hook(hook)
    M = {k: {nm: [] for nm in names} for k in ("lo", "kl_digit", "ans")}
    for i in range(n):
        d_star = S.first_digit(items[i]["answer"])
        rows = torch.stack([COND[nm][i] if COND[nm] is not None else H1[i] for nm in names])
        state["Hnew"], state["on"], state["p"] = rows, torch.tensor([COND[nm] is not None for nm in names], device="cuda"), len(seqs[i]) - 1
        with torch.no_grad():
            cur = torch.tensor([seqs[i]] * len(names), device="cuda")
            o = model(input_ids=cur, use_cache=True, logits_to_keep=1)
            logits, pkv = o.logits[:, -1].float(), o.past_key_values
            gen = []
            for step in range(N_GREEDY):
                nxt = logits.argmax(-1)
                gen.append(nxt)
                o = model(input_ids=nxt[:, None], past_key_values=pkv, use_cache=True, logits_to_keep=1)
                logits, pkv = o.logits[:, -1].float(), o.past_key_values
                if step == 0:
                    lp = torch.log_softmax(logits, -1)
                    ld = lp[:, dig1]
                    other = ld.clone()
                    other[:, d_star] = -1e9
                    lo = (ld[:, d_star] - other.max(1).values).cpu().numpy()
                    kl = (lp[0:1].exp() * (lp[0:1] - lp)).sum(-1).cpu().numpy()
            gen_t = torch.stack(gen, 1).cpu().tolist()
        for j, nm in enumerate(names):
            M["lo"][nm].append(float(lo[j]))
            M["kl_digit"][nm].append(float(kl[j]))
            M["ans"][nm].append(S.parse_answer(tok.decode(gen_t[j])))
    handle.remove()
    out["model_correct_unsteered"] = float(np.mean(GL._arr(M["ans"]["none"]) == truth))
    out["conditions"] = {}
    for nm in names[1:]:
        st = GL.swap_stats(M["ans"], nm, truth, pred)
        st["lo_drop"] = float(np.nanmean(np.array(M["lo"][nm]) - np.array(M["lo"]["none"])))
        st["kl_digit"] = float(np.nanmean(M["kl_digit"][nm]))
        st["answer_equals_donor_answer"] = float(np.mean(GL._arr(M["ans"][nm]) == np.array([items[j]["answer"] for j in donor])))
        out["conditions"][nm] = st
    out["t_total_s"] = round(time.time() - t0, 1)
    torch.save(M | {"names": names, "donor": donor, "pred": pred, "truth": truth}, f"{OUT3}/g4b_raw.pt")
    Path(f"{OUT3}/stage2a_g4b.json").write_text(json.dumps(out, indent=1, default=float))
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(operand_stage2a_g4b.remote(), indent=1, default=str))
