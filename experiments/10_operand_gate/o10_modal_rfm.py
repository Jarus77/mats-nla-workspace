"""10 operand gate, stage 1 (G1/G2/G3 only; no G4, no NLA) -- PILOT, EXPLORATORY.

Prompted by an external result seen 2026-10-12 (M. Kenney, nla-consistency): a linear probe reads both operands
of "a op b =" from the layer-42 activation at "=" (83%/93% exact) while the NLA reports them 5%/38%.
Question here: is operand identity inside our J-space (prereg §4.1 basis, r* = 384 / 4r* = 1536)?

    modal deploy o10_modal_rfm.py          # app "rfm"
    python spawn_driver.py spawn           # server-side spawn; ./watch.sh polls and downloads
    ../../.venv/bin/python o10_analyze.py  # CPU: table + labels -> RESULTS.md section

Layer convention: L42 = output of decoder block 42 = HF hidden_states[43] (02's *_hs43.pt naming).
Capture matches his: bare prompt, no chat template, no few-shot, activation at the final token (" =").
"""
import json
import time
from pathlib import Path

import modal

from env import BASIS_L42, DATA_REMOTE, OUT, VOLUMES, app_image

app = modal.App("rfm")
L = 42


def _load_model():  # noqa: ANN202
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def _load_lens_and_unembed():  # noqa: ANN202
    """n1000 J-Lens at L42, W_U and the final-norm gain (copied from 04 bf_modal._load_lens_and_unembed)."""
    import glob

    import torch
    from safetensors import safe_open

    from mats.jspace import load_jlens_layers

    try:
        from huggingface_hub import hf_hub_download

        from mats.readers.jlens import FILE, REPO
        path = hf_hub_download(REPO, FILE)
    except Exception:  # noqa: BLE001
        path = glob.glob("/results/02_nonj_features/lenses/np/**/*n1000*.pt", recursive=True)[0]
    J = load_jlens_layers(path, [L])[L].cuda()
    snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    widx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    ukey = [k for k in widx if k.endswith("lm_head.weight")][0]
    nkey = [k for k in widx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k and "mtp" not in k][0]

    def get(k: str):  # noqa: ANN202
        with safe_open(f"{snap}/{widx[k]}", framework="pt") as f:
            return f.get_tensor(k)

    W_U = get(ukey).cuda().float()
    gain = 1 + get(nkey).float().cuda()  # Qwen3_5RMSNorm uses (1 + weight)
    return J, W_U, gain, path


def _capture(model, tok, items):  # noqa: ANN001, ANN202
    """h@L42 at the final token [N, d] fp32, the model's greedy next token, and tokenisation facts."""
    import torch

    from mats import capture as C

    seqs = [tok.encode(it["text"], add_special_tokens=False) for it in items]
    H = C.capture(model, seqs, [L], [[-1]] * len(seqs), batch_size=16)[L][:, 0]
    lg = C.logits_at(model, seqs, [[-1]] * len(seqs), batch_size=16)[:, 0]
    nxt = [tok.decode([int(i)]) for i in lg.argmax(-1)]
    facts = {"example_tokens": [tok.decode([t]) for t in seqs[0]], "last_token": tok.decode([seqs[0][-1]]),
             "len_encode_51": len(tok.encode("51", add_special_tokens=False)),
             "len_encode_space5": len(tok.encode(" 5", add_special_tokens=False)),
             "seq_len_min_max": [min(map(len, seqs)), max(map(len, seqs))]}
    return H, nxt, facts, torch.log_softmax(lg.float(), -1)


@app.function(image=app_image, volumes=VOLUMES, gpu="A100-80GB", max_containers=1, max_inputs=1, retries=2,
              timeout=2 * 3600, memory=131072)
def operand_gate() -> dict:
    import numpy as np
    import torch

    import bf_gate_lib as G
    import o10_lib as OL
    from mats.jspace import load_basis_file

    t0 = time.time()
    VOLUMES["/results"].reload()
    items = OL.load_items(DATA_REMOTE)
    y = OL.labels(items)
    groups = np.arange(len(items))  # one prompt = one item; no shared passages
    model, tok = _load_model()
    H, nxt, facts, lp = _capture(model, tok, items)
    del model
    torch.cuda.empty_cache()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    torch.save({"H": H, "ids": [it["id"] for it in items], "layer": L, "hf_hidden_states_index": L + 1,
                "next_token": nxt, "facts": facts}, f"{OUT}/acts_hs43.pt")
    t_cap = time.time() - t0
    his_next = [it["model_next"] for it in items]
    out: dict = {"n": len(items), "capture": facts, "t_capture_s": round(t_cap, 1),
                 "next_token": {"ours_top5": sorted({t: nxt.count(t) for t in set(nxt)}.items(), key=lambda kv: -kv[1])[:5],
                                "agree_with_his_model_next": float(np.mean([a == b for a, b in zip(nxt, his_next)]))},
                 "slots": {}}
    X = H.cuda()
    basis = load_basis_file(BASIS_L42, "cuda")
    Qs = {r: basis.Q(r) for r in OL.RANKS}
    out["random_band_iso_K100"] = OL.random_band(Qs, basis.d)
    out["acts_enrichment"] = {str(r): float(((X - X.mean(0)) @ Q).pow(2).sum() / (X - X.mean(0)).pow(2).sum() / (r / basis.d))
                              for r, Q in Qs.items()}
    J, W_U, gain, lens_path = _load_lens_and_unembed()
    out["lens_path"] = lens_path
    dig_ids = OL.digit_label_ids(tok)
    out["digit_label_tokens"] = [[tok.decode([i]) for i in ids] for ids in dig_ids]
    dirs: dict = {}
    for slot in OL.SLOTS:
        ys = y[slot]
        S = OL.ovr_cv_scores(X, ys, groups)
        g1 = OL.g1_multiclass(S, ys)
        g2 = OL.g2_slot(X, ys, Qs)
        g3 = OL.g3_slot(X, J, W_U, gain, dig_ids, ys, tok)
        dirs[slot] = g2.pop("dirs")
        out["slots"][slot] = {"G1": g1, "G2": g2["ranks"], "G3": g3, "label": OL.label_slot(g1, g2, g3)}
        print(slot, "acc", round(g1["acc"], 3), "mAUROC", round(g1["macro_auroc"], 3),
              "enr", {r: round(v["class_mean"]["median"], 2) for r, v in g2["ranks"].items()},
              "lens", round(g3["stat"], 3), g3["null_band95"], out["slots"][slot]["label"], flush=True)
    # exact operand accuracy (both digits right), his comparison numbers 0.827 / 0.932
    for name, (t, u) in {"a": ("a_tens", "a_units"), "b": ("b_tens", "b_units"), "ans": ("ans_tens", "ans_units")}.items():
        ok = (np.array(out["slots"][t]["G1"]["pred"]) == y[t]) & (np.array(out["slots"][u]["G1"]["pred"]) == y[u])
        out[f"exact_{name}"] = {"acc": float(ok.mean()), "ci95": OL.wilson(int(ok.sum()), len(ok))}
    # operator (binary): G1, G2, G3
    yo = y["op"]
    op_ids = OL.op_label_ids(tok)
    d_op = G.mean_diff(X, yo)
    out["op"] = {"G1": G.probe_report(X, yo, groups),
                 "G2": {str(r): {"diffmeans": G.enrich(Q, d_op)} for r, Q in Qs.items()},
                 "G3": G.functional_stat(G.jlens_ranks_topk(X, J, W_U, gain, op_ids, k=10)[0], yo),
                 "label_tokens": [[tok.decode([i]) for i in ids] for ids in op_ids]}
    dirs["op"] = {"class_mean": d_op.cpu()}
    # magnitude directions (the NLA keeps rough size, r = 0.57 / 0.88 in his data)
    out["magnitude"] = {}
    for name in ("a", "b", "answer"):
        v, stats = OL.ridge_direction(X, y[name].astype(float))
        out["magnitude"][name] = {**stats, "enrichment": {str(r): float(G.enrich(Q, v)) for r, Q in Qs.items()}}
        dirs[f"mag_{name}"] = v.cpu()
    torch.save(dirs, f"{OUT}/dirs.pt")
    out["t_total_s"] = round(time.time() - t0, 1)
    Path(f"{OUT}/gate.json").write_text(json.dumps(out, indent=1, default=float))
    VOLUMES["/results"].commit()
    return {k: v for k, v in out.items() if k not in ("slots",)} | {"labels": {s: v["label"] for s, v in out["slots"].items()}}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(operand_gate.remote(), indent=1, default=str))
