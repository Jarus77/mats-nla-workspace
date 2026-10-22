"""10 operand gate, stage 2a-G4 (model side only; no NLA, no Oracle Lens) -- PILOT, EXPLORATORY.

One-digit few-shot bank (117 items: a, b in 1..9, "+" and "-" with a > b, 3 exemplars each). Directions from the
600-item few-shot bank (stage 2a `dirs_fewshot.pt`): a_units one-vs-rest class-means and the a-magnitude ridge
direction, raw and J-removed (top-1536 n1000 J-space). Edits at "=" (L42): centred ±Δ_i at r11 / r22
(Δ_i = 0.11 / 0.22 ‖h_i‖), and Kenney's class-mean SUBSPACE SWAP (9-dim a_units class-mean span, raw and
J-removed, target profile = this bank's mean profile for a' = a + 3 wrapped in 1..9). Bands: K = 20 J-complement +
20 isotropic directions at the same per-item norm; for the swap, 20 + 20 random 9-dim subspaces at the same
displacement norm. Leak test (prereg §4.4.3) for every J-removed condition with K = 100 J-complement contrasts.

    modal deploy o10g4_modal_rfm.py ; python spawn_driver2.py spawn operand_stage2a_g4
"""
import glob
import json
import time
from pathlib import Path

import modal

from env import BASIS_L42, OUT, VOLUMES, app_image

app = modal.App("rfm")
L = 42
OUT2 = f"{OUT}/stage2a"
OUT3 = f"{OUT}/stage2a_g4"
K_JC, K_ISO, K_LEAK = 20, 20, 100
N_GREEDY = 4
CHUNK = 16
DIRS = ("cm_own_jrem", "cm_own_raw", "mag_jrem", "mag_raw")
DOSES = ("r11", "r22")


def _load_model():  # noqa: ANN202
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def _unembed():  # noqa: ANN202
    import torch
    from safetensors import safe_open

    snap = glob.glob("/weights/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    widx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    ukey = [k for k in widx if k.endswith("lm_head.weight")][0]
    nkey = [k for k in widx if k.endswith("norm.weight") and "layers" not in k and "visual" not in k and "mtp" not in k][0]

    def get(k: str):  # noqa: ANN202
        with safe_open(f"{snap}/{widx[k]}", framework="pt") as f:
            return f.get_tensor(k)

    return get(ukey).cuda().float(), 1 + get(nkey).float().cuda()


def _n1000_lens():  # noqa: ANN202
    from mats.jspace import load_jlens_layers

    path = glob.glob("/weights/hf/hub/models--neuronpedia--jacobian-lens/snapshots/*/**/*n1000*.pt", recursive=True)[0]
    return load_jlens_layers(path, [L])[L].cuda(), path


@app.function(image=app_image, volumes=VOLUMES, gpu="A100-80GB", max_containers=1, single_use_containers=True,
              retries=1, timeout=2 * 3600, memory=131072)
def operand_stage2a_g4() -> dict:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import numpy as np
    import torch

    import bf_gate_lib as G
    import o10g4_lib as GL
    import o10s2_lib as S
    from mats.jspace import load_basis_file

    t0 = time.time()
    VOLUMES["/results"].reload()
    Path(OUT3).mkdir(parents=True, exist_ok=True)
    items = GL.onedigit_items()
    n = len(items)
    a_lab = np.array([it["a"] for it in items])
    b_lab = np.array([it["b"] for it in items])
    truth = np.array([it["answer"] for it in items])
    pred = np.array([GL.predicted_answer(GL.aprime(it["a"]), it["b"], it["op"]) for it in items])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    texts = [GL.fewshot_text_onedigit(it) for it in items]
    seqs = [tok.encode(t, add_special_tokens=False) for t in texts]
    space_id = tok.encode(" ", add_special_tokens=False)
    assert len(space_id) == 1
    space_id = space_id[0]
    out: dict = {"n": n, "fewshot_example": texts[0], "seq_len_min_max": [min(map(len, seqs)), max(map(len, seqs))], "stages": {}}

    # ---- A: capture
    f_acts = f"{OUT3}/acts_onedigit_hs43.pt"
    model = None
    if not Path(f_acts).exists():
        model, _ = _load_model()
        from mats import capture as C
        H1 = C.capture(model, seqs, [L], [[-1]] * n, batch_size=16)[L][:, 0]
        torch.save({"H": H1, "texts": texts, "ids": [it["id"] for it in items], "layer": L}, f_acts)
        VOLUMES["/results"].commit()
        out["stages"]["capture"] = "ran"
    else:
        out["stages"]["capture"] = "skipped (exists)"
    H1 = torch.load(f_acts, map_location="cpu", weights_only=False)["H"].cuda().float()
    norms = H1.norm(dim=1)
    out["median_norm"] = float(norms.median())

    # ---- directions from the 600-item few-shot bank
    basis = load_basis_file(BASIS_L42, "cuda")
    Q = basis.Q(1536)
    d = basis.d
    DF = torch.load(f"{OUT2}/dirs_fewshot.pt", map_location="cpu", weights_only=False)
    cm_all = DF["slots"]["a_units@eq"]["class_mean"].cuda().float()  # [10, d] unit rows, digit 0..9
    cm = cm_all[1:10]  # digits 1..9
    cm = cm / cm.norm(dim=1, keepdim=True)
    cm_j = GL.project_out_rows(Q, cm)
    cm_j = cm_j / cm_j.norm(dim=1, keepdim=True)
    mag = DF["slots"]["mag_a"].cuda().float()
    mag = mag / mag.norm()
    mag_j = GL.project_out_rows(Q, mag[None])[0]
    mag_j = mag_j / mag_j.norm()
    U_raw = GL.orthonormal_span(cm)  # [d, 9]
    U_j = GL.orthonormal_span(cm_j)
    out["dir_enrichment"] = {"cm_raw": [float(G.enrich(basis.Q(384), v)) for v in cm], "cm_jrem_1536": [float(G.enrich(Q, v)) for v in cm_j],
                             "mag_raw_384": float(G.enrich(basis.Q(384), mag)), "mag_jrem_1536": float(G.enrich(Q, mag_j)),
                             "cm_raw_1536": [float(G.enrich(Q, v)) for v in cm]}
    # swap targets: this bank's mean profiles per digit a
    P_raw = GL.class_profiles(H1, a_lab, U_raw, GL.DIGITS)
    P_j = GL.class_profiles(H1, a_lab, U_j, GL.DIGITS)
    T_raw = torch.stack([P_raw[GL.aprime(int(a))] for a in a_lab])
    T_j = torch.stack([P_j[GL.aprime(int(a))] for a in a_lab])
    Hs_raw = GL.swap_vectors(H1, U_raw, T_raw)
    Hs_j = GL.swap_vectors(H1, U_j, T_j)
    disp_raw = (Hs_raw - H1).norm(dim=1)
    disp_j = (Hs_j - H1).norm(dim=1)
    out["swap_displacement_over_norm"] = {"raw": float((disp_raw / norms).mean()), "jrem": float((disp_j / norms).mean())}
    # per-item direction vectors
    V = {"cm_own_jrem": cm_j[a_lab - 1], "cm_own_raw": cm[a_lab - 1], "mag_jrem": mag_j[None].expand(n, -1), "mag_raw": mag[None].expand(n, -1)}
    R = S.random_dirs(d, K_JC, K_ISO, Q, seed=4242).cuda()
    SS_jc = GL.random_subspaces(d, 9, K_JC, Q, seed=99).cuda()
    SS_iso = GL.random_subspaces(d, 9, K_ISO, None, seed=100).cuda()

    def centred_rows(Hrow: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return Hrow - (Hrow * v).sum(-1, keepdim=True) * v + a[:, None] * v

    # ---- condition table: name -> [n, d] edited vectors (None = unsteered)
    COND: dict[str, torch.Tensor | None] = {"none": None}
    for dn in DIRS:
        for dose in DOSES:
            for sgn in (1, -1):
                COND[f"{dn}@{dose}{'+' if sgn > 0 else '-'}"] = centred_rows(H1, V[dn], sgn * GL.REL[dose] * norms)
    COND["swap_raw"] = Hs_raw
    COND["swap_jrem"] = Hs_j
    for dose in DOSES:
        for k in range(K_JC + K_ISO):
            fam, kk = ("jc", k) if k < K_JC else ("iso", k - K_JC)
            for sgn in (1, -1):
                COND[f"{fam}{kk}@{dose}{'+' if sgn > 0 else '-'}"] = centred_rows(H1, R[k][None].expand(n, -1), sgn * GL.REL[dose] * norms)
    for k in range(K_JC):
        COND[f"swapnull_jc{k}"] = H1 + GL.random_displacements(SS_jc[k], disp_j.cpu(), seed=1000 + k)
    for k in range(K_ISO):
        COND[f"swapnull_iso{k}"] = H1 + GL.random_displacements(SS_iso[k], disp_j.cpu(), seed=2000 + k)
    names = list(COND)
    out["n_conditions"] = len(names)

    # ---- D: G4 forward loop (chunked, checkpointed)
    f_g4 = f"{OUT3}/g4_raw.pt"
    f_part = f"{OUT3}/g4_partial.pt"
    if not Path(f_g4).exists():
        if model is None:
            model, _ = _load_model()
        dig1 = torch.tensor([tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)], device="cuda")
        state: dict = {"Hnew": None, "on": None, "p": None}

        def hook(_m, _i, o):  # noqa: ANN001, ANN202
            h = o[0] if isinstance(o, tuple) else o
            if h.shape[1] == 1:
                return o
            h2 = h.clone()
            cur = h[:, state["p"]].float()
            h2[:, state["p"]] = torch.where(state["on"][:, None], state["Hnew"], cur).to(h.dtype)
            return (h2, *o[1:]) if isinstance(o, tuple) else h2

        from mats.capture import decoder_blocks
        handle = decoder_blocks(model)[L].register_forward_hook(hook)
        if Path(f_part).exists():
            P = torch.load(f_part, map_location="cpu", weights_only=False)
            M, done = P["M"], int(P["done"])
        else:
            M = {key: {nm: [] for nm in names} for key in ("lo", "kl_digit", "kl_eq", "ans", "first_tok")}
            done = 0
        t_g4 = time.time()
        for i in range(done, n):
            d_star = S.first_digit(items[i]["answer"])
            rows_all = torch.stack([COND[nm][i] if COND[nm] is not None else H1[i] for nm in names])  # [C, d]
            on_all = torch.tensor([COND[nm] is not None for nm in names], device="cuda")
            lo_all, kld_all, kle_all, gen_all = [], [], [], []
            lp_eq_none = lp_dig_none = None
            for c0 in range(0, len(names), CHUNK):
                sl = slice(c0, c0 + CHUNK)
                B = rows_all[sl].shape[0]
                state["Hnew"], state["on"], state["p"] = rows_all[sl], on_all[sl], len(seqs[i]) - 1
                with torch.no_grad():
                    cur = torch.tensor([seqs[i]] * B, device="cuda")
                    o = model(input_ids=cur, use_cache=True, logits_to_keep=1)
                    logits, pkv = o.logits[:, -1].float(), o.past_key_values
                    lp_eq = torch.log_softmax(logits, -1)
                    if c0 == 0:
                        lp_eq_none = lp_eq[0:1].clone()
                    kle_all.append((lp_eq_none.exp() * (lp_eq_none - lp_eq)).sum(-1).cpu().numpy())
                    gen, lo, kld = [], None, None
                    for step in range(N_GREEDY):
                        nxt = logits.argmax(-1)
                        gen.append(nxt)
                        o = model(input_ids=nxt[:, None], past_key_values=pkv, use_cache=True, logits_to_keep=1)
                        logits, pkv = o.logits[:, -1].float(), o.past_key_values
                        if step == 0:
                            lp = torch.log_softmax(logits, -1)
                            if c0 == 0:
                                lp_dig_none = lp[0:1].clone()
                            ld = lp[:, dig1]
                            other = ld.clone()
                            other[:, d_star] = -1e9
                            lo = (ld[:, d_star] - other.max(1).values).cpu().numpy()
                            kld = (lp_dig_none.exp() * (lp_dig_none - lp)).sum(-1).cpu().numpy()
                    gen_all += torch.stack(gen, 1).cpu().tolist()
                lo_all.append(lo)
                kld_all.append(kld)
                del o, pkv, logits, cur
            torch.cuda.empty_cache()
            lo, kld, kle = np.concatenate(lo_all), np.concatenate(kld_all), np.concatenate(kle_all)
            valid = np.array([g[0] == space_id for g in gen_all]) & (gen_all[0][0] == space_id)
            lo, kld = np.where(valid, lo, np.nan), np.where(valid, kld, np.nan)
            for j, nm in enumerate(names):
                M["lo"][nm].append(float(lo[j]))
                M["kl_digit"][nm].append(float(kld[j]))
                M["kl_eq"][nm].append(float(kle[j]))
                M["ans"][nm].append(S.parse_answer(tok.decode(gen_all[j])))
                M["first_tok"][nm].append(int(gen_all[j][0]))
            if (i + 1) % 30 == 0 or i + 1 == n:
                torch.save({"M": M, "done": i + 1}, f_part)
                VOLUMES["/results"].commit()
                print("g4", i + 1, n, round(time.time() - t_g4), "s", flush=True)
        handle.remove()
        out["t_g4_s"] = round(time.time() - t_g4, 1)
        torch.save(M | {"names": names, "truth": truth, "pred": pred}, f_g4)
        VOLUMES["/results"].commit()
        out["stages"]["g4"] = "ran"
    else:
        out["stages"]["g4"] = "skipped (exists)"
    if model is not None:
        del model
        torch.cuda.empty_cache()
    M = torch.load(f_g4, map_location="cpu", weights_only=False)

    # ---- summaries
    none_ans = GL._arr(M["ans"]["none"])
    out["model_correct_unsteered"] = float(np.nanmean(none_ans == truth))
    out["first_token_is_space"] = float(np.mean(np.array(M["first_tok"]["none"]) == space_id))
    lo = {k: np.array(v, float) for k, v in M["lo"].items()}
    kld = {k: np.array(v, float) for k, v in M["kl_digit"].items()}
    kle = {k: np.array(v, float) for k, v in M["kl_eq"].items()}
    val = {k: GL._arr(v) for k, v in M["ans"].items()}

    def stats_pm(key: str) -> dict:
        """± contrast statistics for a direction@dose key (without the sign)."""
        fp, fm = GL.flip_rate(M["ans"], key + "+"), GL.flip_rate(M["ans"], key + "-")
        lo_eff = float(np.nanmean((lo[key + "+"] - lo[key + "-"]) / 2))
        lo_drop = float(np.nanmean(np.concatenate([lo[key + "+"] - lo["none"], lo[key + "-"] - lo["none"]])))
        v_eff = float(np.nanmean((val[key + "+"] - val[key + "-"]) / 2))
        v_abs = float(np.nanmean(np.concatenate([np.abs(val[key + "+"] - truth), np.abs(val[key + "-"] - truth)])))
        return {"flip_plus": fp, "flip_minus": fm, "flip_mean": (fp + fm) / 2, "lo_signed_effect": lo_eff, "lo_drop": lo_drop,
                "value_signed_effect": v_eff, "value_abs_error": v_abs,
                "kl_digit": float(np.nanmean(np.concatenate([kld[key + "+"], kld[key + "-"]]))),
                "kl_eq": float(np.nanmean(np.concatenate([kle[key + "+"], kle[key + "-"]])))}

    out["directions"] = {}
    for dose in DOSES:
        nulls = {"jc": [stats_pm(f"jc{k}@{dose}") for k in range(K_JC)], "iso": [stats_pm(f"iso{k}@{dose}") for k in range(K_ISO)]}
        for dn in DIRS:
            st = stats_pm(f"{dn}@{dose}")
            res = dict(st)
            for key in ("flip_mean", "lo_signed_effect", "lo_drop", "value_signed_effect", "value_abs_error", "kl_digit"):
                res["band_" + key] = GL.banded(st[key], {fam: [x[key] for x in v] for fam, v in nulls.items()})
            out["directions"][f"{dn}@{dose}"] = res
        out["directions"][f"null_kl_digit@{dose}"] = {fam: float(np.mean([x["kl_digit"] for x in v])) for fam, v in nulls.items()}
    out["swap"] = {}
    sn = {"jc": [GL.swap_stats(M["ans"], f"swapnull_jc{k}", truth, pred) for k in range(K_JC)],
          "iso": [GL.swap_stats(M["ans"], f"swapnull_iso{k}", truth, pred) for k in range(K_ISO)]}
    for nm in ("swap_raw", "swap_jrem"):
        st = GL.swap_stats(M["ans"], nm, truth, pred)
        st["lo_drop"] = float(np.nanmean(lo[nm] - lo["none"]))
        st["kl_digit"] = float(np.nanmean(kld[nm]))
        st["kl_eq"] = float(np.nanmean(kle[nm]))
        for key in ("follow_exact", "moved_toward", "norm_shift_mean", "flip_rate"):
            st["band_" + key] = GL.banded(st[key], {fam: [x[key] for x in v] for fam, v in sn.items()})
        out["swap"][nm] = st
    out["swap"]["null_kl_digit"] = {fam: float(np.nanmean([np.nanmean(kld[f"swapnull_{fam}{k}"]) for k in range(K_JC)])) for fam in ("jc", "iso")}

    # ---- leak (prereg §4.4.3) on the captured vectors
    W_U, gain = _unembed()
    J, lens_path = _n1000_lens()
    sets = S.digit_token_sets(tok)
    set_ids = [sets["content"], sets["label"]]

    def set_scores(Hs: torch.Tensor) -> np.ndarray:
        rk, _ = G.jlens_ranks_topk(Hs, J, W_U, gain, set_ids, k=1)
        return -np.log10(rk.astype(np.float64) + 1)

    RL = S.random_dirs(d, K_LEAK, 0, Q, seed=77).cuda()
    SL = GL.random_subspaces(d, 9, K_LEAK, Q, seed=78).cuda()
    Sc: dict = {"none": set_scores(H1)}
    for nm in ("cm_own_jrem", "cm_own_raw", "mag_jrem", "mag_raw"):
        for dose in DOSES:
            for sgn in ("+", "-"):
                Sc[f"{nm}@{dose}{sgn}"] = set_scores(COND[f"{nm}@{dose}{sgn}"])
    for dose in DOSES:
        for k in range(K_LEAK):
            v = RL[k]
            base = H1 - (H1 @ v)[:, None] * v
            for sgn in (1, -1):
                Sc[f"jcL{k}@{dose}{'+' if sgn > 0 else '-'}"] = set_scores(base + sgn * (GL.REL[dose] * norms)[:, None] * v)
    for nm in ("swap_raw", "swap_jrem"):
        Sc[nm + "+"] = set_scores(COND[nm])
        Sc[nm + "-"] = Sc["none"]
    for k in range(K_LEAK):
        Sc[f"swapL{k}+"] = set_scores(H1 + GL.random_displacements(SL[k], disp_j.cpu(), seed=3000 + k))
        Sc[f"swapL{k}-"] = Sc["none"]
    out["leak"] = {}
    for si, sname in enumerate(("content", "label")):
        Ssub = {k: v[:, si] for k, v in Sc.items()}
        res = {}
        for dose in DOSES:
            res.update(S.leak_summary(Ssub, [f"{nm}@{dose}" for nm in ("cm_own_jrem", "cm_own_raw", "mag_jrem", "mag_raw")],
                                      [f"jcL{k}@{dose}" for k in range(K_LEAK)]))
        res.update(S.leak_summary(Ssub, ["swap_raw", "swap_jrem"], [f"swapL{k}" for k in range(K_LEAK)]))
        out["leak"][sname] = res
    out["leak_sets"] = {"content_n": len(sets["content"]), "label": [tok.decode([i]) for i in sets["label"]]}
    out["lens_path"] = lens_path
    out["t_total_s"] = round(time.time() - t0, 1)
    Path(f"{OUT3}/stage2a_g4.json").write_text(json.dumps(out, indent=1, default=float))
    VOLUMES["/results"].commit()
    return {k: v for k, v in out.items() if k not in ("directions", "swap", "leak")}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(operand_stage2a_g4.remote(), indent=1, default=str))
