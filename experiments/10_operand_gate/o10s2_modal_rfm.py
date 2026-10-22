"""10 operand gate, stage 2a (model-side only; no NLA, no Oracle Lens) -- PILOT, EXPLORATORY.

Few-shot bank (3 exemplars + the item), L42 at the final "=": G1-G3 re-run (answer slots as the in-J
reference, also read at the appended-space position where the first answer digit is the next token),
the prereg §4.4.3 leak test and the model-side G4 for the J-removed a-magnitude and a_tens class-mean
directions (2-fold cross-fit, centred steering, q90 doses, J-complement and isotropic bands), and a J++
(koayon/jpp-lenses) second-basis enrichment check.

    modal deploy o10s2_modal_rfm.py          # app "rfm" (replaces the deployed function set)
    python spawn_driver2.py spawn jpp_download ; python spawn_driver2.py spawn operand_stage2a
    nohup python watch2.py &                 # polls, chains jpp_enrich, downloads results/stage2a/*
"""
import glob
import json
import time
from pathlib import Path

import modal

from env import BASIS_L42, DATA_REMOTE, OUT, VOLUMES, app_image
from modal_common import download_image

dl_image = download_image.add_local_python_source("env", "o10_lib", "o10s2_lib")

app = modal.App("rfm")
L = 42
OUT2 = f"{OUT}/stage2a"
JPP_REPO, JPP_FILE = "koayon/jpp-lenses", "qwen3.6-27b/lens.pt"
REAL = ("mag_a_raw", "mag_a_jrem", "cm_atens_jrem")
K_JC, K_ISO = 20, 20  # nulls at the mag_a_jrem dose
K_CM = 10  # jc and iso nulls at the cm_atens_jrem dose
N_GREEDY = 4
G4_CHUNK = 16  # rows per forward (the 127-row batch OOM'd inside fla chunk_gated_delta_rule)


def _load_model():  # noqa: ANN202
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def _unembed():  # noqa: ANN202
    """W_U [V, d] fp32 and the final-norm gain (1 + w) from the safetensors snapshot."""
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


def _jpp_path() -> str | None:
    hits = glob.glob(f"/weights/hf/hub/models--koayon--jpp-lenses/snapshots/*/{JPP_FILE}")
    return hits[0] if hits else None


@app.function(image=dl_image, volumes=VOLUMES, timeout=3600, cpu=4, memory=16384)
def jpp_download() -> dict:
    """Fetch the J++ lens file (6.6 GB, all layers) into the HF cache on rfm-weights."""
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    p = hf_hub_download(JPP_REPO, JPP_FILE)
    VOLUMES["/weights"].commit()
    return {"path": p, "t_s": round(time.time() - t0, 1)}


@app.function(image=app_image, volumes=VOLUMES,
              gpu="A100-80GB", max_containers=1, single_use_containers=True, retries=1, timeout=3 * 3600, memory=131072)
def operand_stage2a() -> dict:
    """Stages, each checkpointed to rfm-results and skipped when its file exists:
    A capture -> acts_fewshot_hs43.pt; D G4 (chunked, partial saves) -> g4_raw.pt; B gates -> gate_fewshot.json;
    C leak -> leak.json; then stage2a.json merges everything."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # before torch initialises CUDA
    import numpy as np
    import torch

    import o10_lib as OL
    import o10s2_lib as S

    t0 = time.time()
    VOLUMES["/results"].reload()
    items = OL.load_items(DATA_REMOTE)
    y = OL.labels(items)
    n = len(items)
    Path(OUT2).mkdir(parents=True, exist_ok=True)
    out: dict = {"n": n, "stages": {}}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    texts = [S.fewshot_text(it) for it in items]
    seqs = [tok.encode(t, add_special_tokens=False) for t in texts]
    sp = tok.encode(" ", add_special_tokens=False)
    assert len(sp) == 1, sp
    space_id = sp[0]
    out["fewshot_example"] = texts[0]
    out["seq_len_min_max"] = [min(map(len, seqs)), max(map(len, seqs))]

    # ---- A: capture
    f_acts = f"{OUT2}/acts_fewshot_hs43.pt"
    model = None
    if not Path(f_acts).exists():
        model, _ = _load_model()
        from mats import capture as C
        ta = time.time()
        H = C.capture(model, seqs, [L], [[-1]] * n, batch_size=16)[L][:, 0]
        H_sp = C.capture(model, [s_ + [space_id] for s_ in seqs], [L], [[-1]] * n, batch_size=16)[L][:, 0]
        torch.save({"H": H, "H_space": H_sp, "ids": [it["id"] for it in items], "layer": L, "hf_hidden_states_index": L + 1,
                    "texts": texts, "t_s": round(time.time() - ta, 1)}, f_acts)
        VOLUMES["/results"].commit()
        out["stages"]["capture"] = "ran"
    else:
        out["stages"]["capture"] = "skipped (exists)"
    A = torch.load(f_acts, map_location="cpu", weights_only=False)
    H, H_sp = A["H"], A["H_space"]
    out["t_capture_s"] = A.get("t_s")

    # ---- cross-fit directions and doses (cheap; recomputed deterministically every run)
    from mats.jspace import load_basis_file
    import bf_gate_lib as G
    basis = load_basis_file(BASIS_L42, "cuda")
    Q = basis.Q(1536)
    X = H.cuda()
    fold = S.folds(n)
    c_mode = int(np.bincount(y["a_tens"]).argmax())
    D: dict = {}
    for f in (0, 1):
        tr = np.where(fold != f)[0]
        v_raw, st = OL.ridge_direction(X[tr], y["a"][tr].astype(float))
        v_raw = S.unit(v_raw)
        v_jrem = S.unit(S.project_out_rows(Q, v_raw))
        v_cm = G.mean_diff(X[tr], (y["a_tens"][tr] == c_mode).astype(int))
        v_cm_j = S.unit(S.project_out_rows(Q, v_cm))
        dirs = {"mag_a_raw": v_raw, "mag_a_jrem": v_jrem, "cm_atens_raw": v_cm, "cm_atens_jrem": v_cm_j}
        D[f] = {"dirs": dirs, "doses": {k: S.q90_dose(X[tr], v) for k, v in dirs.items()}, "ridge_stats": st,
                "median_norm": float(X[tr].norm(dim=1).median())}
    out["c_mode_atens"] = c_mode
    out["doses"] = {f: {k: round(v, 3) for k, v in D[f]["doses"].items()} for f in D}
    out["dose_over_median_norm"] = {k: float(np.mean([D[f]["doses"][k] / D[f]["median_norm"] for f in D])) for k in REAL}
    R = S.random_dirs(basis.d, K_JC, K_ISO, Q, seed=2026).cuda()
    null_names = {"jc": [f"jc{k}" for k in range(K_JC)], "iso": [f"iso{k}" for k in range(K_ISO)]}
    Qs = {r: basis.Q(r) for r in OL.RANKS}
    out["g4_dirs_enrichment"] = {f: {k: {str(r): float(G.enrich(Qq, v)) for r, Qq in Qs.items()} for k, v in D[f]["dirs"].items()} for f in D}

    def conds(f: int) -> list[tuple[str, torch.Tensor | None, float]]:
        cs: list = [("none", None, 0.0)]
        for nm in REAL:
            for s_ in (1, -1):
                cs.append((nm + ("+" if s_ > 0 else "-"), D[f]["dirs"][nm], s_ * D[f]["doses"][nm]))
        for dn, kj, ki in (("mag_a_jrem", K_JC, K_ISO), ("cm_atens_jrem", K_CM, K_CM)):
            a = D[f]["doses"][dn]
            for fam, kk, off in (("jc", kj, 0), ("iso", ki, K_JC)):
                for k in range(kk):
                    for s_ in (1, -1):
                        cs.append((f"{fam}{k}@{dn}" + ("+" if s_ > 0 else "-"), R[off + k], s_ * a))
        return cs

    cond_names = [c[0] for c in conds(0)]

    # ---- D: G4, model side, chunked (<= G4_CHUNK rows per forward), partial checkpoints every 100 items
    f_g4 = f"{OUT2}/g4_raw.pt"
    f_part = f"{OUT2}/g4_partial.pt"
    if not Path(f_g4).exists():
        if model is None:
            model, _ = _load_model()
        dig1_l = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
        assert all(len(e) == 1 for e in dig1_l), dig1_l
        dig1 = torch.tensor([e[0] for e in dig1_l], device="cuda")
        state: dict = {"V": None, "A": None, "on": None, "p": None}

        def hook(_m, _i, o):  # noqa: ANN001, ANN202
            h = o[0] if isinstance(o, tuple) else o
            if h.shape[1] == 1:  # KV-cache decode step: the "=" position was edited at prefill
                return o
            p = state["p"]
            hf = h[:, p].float()
            V, A_, on = state["V"], state["A"], state["on"]
            new = hf - (hf * V).sum(-1, keepdim=True) * V + A_[:, None] * V
            h2 = h.clone()
            h2[:, p] = torch.where(on[:, None], new, hf).to(h.dtype)
            return (h2, *o[1:]) if isinstance(o, tuple) else h2

        from mats.capture import decoder_blocks
        handle = decoder_blocks(model)[L].register_forward_hook(hook)
        if Path(f_part).exists():
            P = torch.load(f_part, map_location="cpu", weights_only=False)
            M, done = P["M"], int(P["done"])
            out["stages"]["g4"] = f"resumed at {done}"
        else:
            M = {key: {nm: [] for nm in cond_names} for key in ("lo", "kl", "ans", "first_tok")}
            done = 0
            out["stages"]["g4"] = "ran"
        t_g4 = time.time()
        for i in range(done, n):
            it = items[i]
            cs = conds(int(fold[i]))
            d_star = S.first_digit(it["answer"])
            lo_all, kl_all, gen_all = [], [], []
            lp_none = None
            for c0 in range(0, len(cs), G4_CHUNK):
                ch = cs[c0:c0 + G4_CHUNK]
                B = len(ch)
                state["V"] = torch.stack([c[1] if c[1] is not None else torch.zeros(basis.d, device="cuda") for c in ch])
                state["A"] = torch.tensor([c[2] for c in ch], device="cuda", dtype=torch.float32)
                state["on"] = torch.tensor([c[1] is not None for c in ch], device="cuda")
                state["p"] = len(seqs[i]) - 1
                with torch.no_grad():
                    cur = torch.tensor([seqs[i]] * B, device="cuda")
                    o = model(input_ids=cur, use_cache=True, logits_to_keep=1)
                    logits, pkv = o.logits[:, -1].float(), o.past_key_values
                    gen, lo, kl = [], None, None
                    for step in range(N_GREEDY):
                        nxt = logits.argmax(-1)
                        gen.append(nxt)
                        o = model(input_ids=nxt[:, None], past_key_values=pkv, use_cache=True, logits_to_keep=1)
                        logits, pkv = o.logits[:, -1].float(), o.past_key_values
                        if step == 0:  # distribution after the first generated token: the first answer digit when that token is " "
                            lp = torch.log_softmax(logits, -1)
                            if c0 == 0:
                                lp_none = lp[0:1].clone()  # row 0 of chunk 0 is "none"
                            ld = lp[:, dig1]
                            other = ld.clone()
                            other[:, d_star] = -1e9
                            lo = (ld[:, d_star] - other.max(1).values).cpu().numpy()
                            kl = (lp_none.exp() * (lp_none - lp)).sum(-1).cpu().numpy()
                    gen_all += torch.stack(gen, 1).cpu().tolist()
                lo_all.append(lo)
                kl_all.append(kl)
                del o, pkv, logits, cur
            torch.cuda.empty_cache()
            lo = np.concatenate(lo_all)
            kl = np.concatenate(kl_all)
            valid = np.array([g[0] == space_id for g in gen_all]) & (gen_all[0][0] == space_id)
            lo = np.where(valid, lo, np.nan)
            kl = np.where(valid, kl, np.nan)
            for j, c in enumerate(cs):
                M["lo"][c[0]].append(float(lo[j]))
                M["kl"][c[0]].append(float(kl[j]))
                M["ans"][c[0]].append(S.parse_answer(tok.decode(gen_all[j])))
                M["first_tok"][c[0]].append(int(gen_all[j][0]))
            if (i + 1) % 100 == 0 or i + 1 == n:
                torch.save({"M": M, "done": i + 1}, f_part)
                VOLUMES["/results"].commit()
                print("g4", i + 1, n, round(time.time() - t_g4), "s", flush=True)
        handle.remove()
        out["t_g4_s"] = round(time.time() - t_g4, 1)
        torch.save(M | {"fold": fold, "cond_names": cond_names}, f_g4)
        VOLUMES["/results"].commit()
    else:
        out["stages"]["g4"] = "skipped (exists)"
    if model is not None:
        del model
        torch.cuda.empty_cache()
    M = torch.load(f_g4, map_location="cpu", weights_only=False)
    out["t_g4_s"] = out.get("t_g4_s")

    # G4 summaries
    truth = np.array([it["answer"] for it in items])
    none_ans = np.array([a if a is not None else -1 for a in M["ans"]["none"]])
    out["fewshot_model_correct"] = float(np.mean(none_ans == truth))
    out["fewshot_first_token_is_space"] = float(np.mean(np.array(M["first_tok"]["none"]) == space_id))
    out["g4_valid_digit_rows"] = {k: int(np.isfinite(np.array(M["lo"][k], dtype=float)).sum()) for k in ("none", "mag_a_jrem+", "cm_atens_jrem+")}
    lo = {k: np.array(v, dtype=float) for k, v in M["lo"].items()}
    nn_mag = {fam: [k + "@mag_a_jrem" for k in ks] for fam, ks in null_names.items()}
    nn_cm = {fam: [k + "@cm_atens_jrem" for k in ks[:K_CM]] for fam, ks in null_names.items()}
    out["g4"] = {
        "digit_logodds_signed": {"mag_a_raw": S.effect_summary(lo, ["mag_a_raw"], nn_mag)["mag_a_raw"],
                                 "mag_a_jrem": S.effect_summary(lo, ["mag_a_jrem"], nn_mag)["mag_a_jrem"],
                                 "cm_atens_jrem": S.effect_summary(lo, ["cm_atens_jrem"], nn_cm)["cm_atens_jrem"]},
        "digit_logodds_drop": {**S.drop_summary(lo, ["mag_a_raw", "mag_a_jrem"], nn_mag),
                               **S.drop_summary(lo, ["cm_atens_jrem"], nn_cm)},
        "flip": {**S.flip_summary(M["ans"], ["mag_a_raw", "mag_a_jrem"], nn_mag),
                 **S.flip_summary(M["ans"], ["cm_atens_jrem"], nn_cm)},
        "kl_mean": {k: float(np.nanmean(M["kl"][k + "+"] + M["kl"][k + "-"])) for k in REAL},
        "kl_mean_null": {fam: float(np.nanmean([np.nanmean(M["kl"][k + "+"] + M["kl"][k + "-"]) for k in ks]))
                         for fam, ks in nn_mag.items()},
    }
    sign_mag = np.ones(n)
    pred_cm = (c_mode * 10 + y["a"] % 10) + np.where(y["op"] == 1, y["b"], -y["b"])
    sign_cm = np.sign(pred_cm - truth)
    val = {k: S.signed_value_shift(M["ans"], k, sign_mag) for k in ["mag_a_raw", "mag_a_jrem"] + nn_mag["jc"] + nn_mag["iso"]}
    val_cm = {k: S.signed_value_shift(M["ans"], k, sign_cm) for k in ["cm_atens_jrem"] + nn_cm["jc"] + nn_cm["iso"]}

    def band_of(d: dict, names: list[str], fams: dict) -> dict:
        from mats.metrics import null_band
        res = {}
        for nm in names:
            e = d[nm]
            res[nm] = {"mean_shift": float(np.nanmean(e)), "n_valid": int(np.isfinite(e).sum()),
                       "frac_moved_expected": float(np.nanmean(np.sign(e[np.isfinite(e)]) > 0)),
                       **{f"band_{fam}": null_band(float(np.nanmean(e)), [np.nanmean(d[k]) for k in ks]) for fam, ks in fams.items()}}
        return res

    out["g4"]["value_shift"] = {**band_of(val, ["mag_a_raw", "mag_a_jrem"], nn_mag), **band_of(val_cm, ["cm_atens_jrem"], nn_cm)}
    rf = out["g4"]
    out["g4"]["retained_fraction_jrem_over_raw"] = {
        "value_shift": rf["value_shift"]["mag_a_jrem"]["mean_shift"] / max(abs(rf["value_shift"]["mag_a_raw"]["mean_shift"]), 1e-9),
        "flip": rf["flip"]["mag_a_jrem"]["flip_rate"] / max(rf["flip"]["mag_a_raw"]["flip_rate"], 1e-9),
        "digit_logodds_drop": rf["digit_logodds_drop"]["mag_a_jrem"]["drop"] / (rf["digit_logodds_drop"]["mag_a_raw"]["drop"] or 1e-9)}

    # ---- B: G1-G3 on the few-shot bank ("=" for all slots; ans slots also at the appended space)
    W_U, gain = _unembed()
    J, lens_path = _n1000_lens()
    f_gate = f"{OUT2}/gate_fewshot.json"
    dig_ids = OL.digit_label_ids(tok)
    if not Path(f_gate).exists():
        groups = np.arange(n)
        gate: dict = {"slots": {}, "pred": {}}
        dirs_fs = {}
        for tag, Xg in (("eq", X), ("space", H_sp.cuda())):
            slots = OL.SLOTS if tag == "eq" else ("ans_tens", "ans_units")
            for slot in slots:
                ys = y[slot]
                Sc = OL.ovr_cv_scores(Xg, ys, groups)
                g1, g2 = OL.g1_multiclass(Sc, ys), OL.g2_slot(Xg, ys, Qs)
                g3 = OL.g3_slot(Xg, J, W_U, gain, dig_ids, ys, tok)
                dirs_fs[f"{slot}@{tag}"] = g2.pop("dirs")
                key = slot if tag == "eq" else f"{slot}@space"
                gate["pred"][key] = g1.pop("pred")
                gate["slots"][key] = {"G1": g1, "G2": g2["ranks"], "G3": g3, "label": OL.label_slot(g1, g2, g3)}
                print(key, "acc", round(g1["acc"], 3), "enr", {r: round(v["class_mean"]["median"], 2) for r, v in g2["ranks"].items()},
                      "lens", round(g3["stat"], 3), g3["null_band95"], gate["slots"][key]["label"], flush=True)
        for name, (t, u) in {"a": ("a_tens", "a_units"), "b": ("b_tens", "b_units"), "ans": ("ans_tens", "ans_units"),
                             "ans@space": ("ans_tens@space", "ans_units@space")}.items():
            ok = (np.array(gate["pred"][t]) == y[t.split("@")[0]]) & (np.array(gate["pred"][u]) == y[u.split("@")[0]])
            gate[f"exact_{name}"] = {"acc": float(ok.mean()), "ci95": OL.wilson(int(ok.sum()), n)}
        gate.pop("pred")
        gate["magnitude"] = {}
        for name in ("a", "b", "answer"):
            v, stats = OL.ridge_direction(X, y[name].astype(float))
            gate["magnitude"][name] = {**stats, "enrichment": {str(r): float(G.enrich(Qq, v)) for r, Qq in Qs.items()}}
            dirs_fs[f"mag_{name}"] = v.cpu()
        gate["random_band_iso_K100"] = OL.random_band(Qs, basis.d)
        torch.save({"slots": dirs_fs, "g4": {f: {"dirs": {k: v.cpu() for k, v in D[f]["dirs"].items()}, "doses": D[f]["doses"]} for f in D},
                    "fold": fold, "c_mode_atens": c_mode}, f"{OUT2}/dirs_fewshot.pt")
        Path(f_gate).write_text(json.dumps(gate, indent=1, default=float))
        VOLUMES["/results"].commit()
        out["stages"]["gates"] = "ran"
    else:
        out["stages"]["gates"] = "skipped (exists)"
    out.update(json.loads(Path(f_gate).read_text()))

    # ---- C: leak test (prereg §4.4.3): J-Lens set scores on the steered vectors, vs K=100 J-complement contrasts
    f_leak = f"{OUT2}/leak.json"
    if not Path(f_leak).exists():
        sets = S.digit_token_sets(tok)
        leak: dict = {"leak_sets": {"content_n": len(sets["content"]), "label": [tok.decode([i]) for i in sets["label"]],
                                    "multi_digit_single_tokens": sets["multi_digit_single_tokens"]}, "leak": {}}
        set_ids = [sets["content"], sets["label"]]
        RL = S.random_dirs(basis.d, S.K_JC_LEAK, 0, Q, seed=7).cuda()
        Sc_: dict = {}

        def set_scores(Hs: torch.Tensor) -> np.ndarray:
            rk, _ = G.jlens_ranks_topk(Hs, J, W_U, gain, set_ids, k=1)
            return -np.log10(rk.astype(np.float64) + 1)  # [N, 2]

        dose_of = {nm: np.array([D[int(fold[i])]["doses"][nm] for i in range(n)]) for nm in REAL}
        vec_of = {nm: torch.stack([D[int(fold[i])]["dirs"][nm] for i in range(n)]) for nm in REAL}
        for nm in REAL:
            for s_ in (1, -1):
                a = torch.tensor(s_ * dose_of[nm], device="cuda", dtype=torch.float32)
                Hs = X - (X * vec_of[nm]).sum(-1, keepdim=True) * vec_of[nm] + a[:, None] * vec_of[nm]
                Sc_[nm + ("+" if s_ > 0 else "-")] = set_scores(Hs)
        for dn in ("mag_a_jrem", "cm_atens_jrem"):
            a_abs = torch.tensor(dose_of[dn], device="cuda", dtype=torch.float32)
            for k in range(S.K_JC_LEAK):
                v = RL[k]
                base = X - (X @ v)[:, None] * v
                for s_ in (1, -1):
                    Sc_[f"jc{k}@{dn}" + ("+" if s_ > 0 else "-")] = set_scores(base + s_ * a_abs[:, None] * v)
        for si, sname in enumerate(("content", "label")):
            Ssub = {k: v[:, si] for k, v in Sc_.items()}
            leak["leak"][sname] = {**S.leak_summary(Ssub, ["mag_a_raw", "mag_a_jrem"], [f"jc{k}@mag_a_jrem" for k in range(S.K_JC_LEAK)]),
                                   **S.leak_summary(Ssub, ["cm_atens_jrem"], [f"jc{k}@cm_atens_jrem" for k in range(S.K_JC_LEAK)])}
        Path(f_leak).write_text(json.dumps(leak, indent=1, default=float))
        VOLUMES["/results"].commit()
        out["stages"]["leak"] = "ran"
    else:
        out["stages"]["leak"] = "skipped (exists)"
    out.update(json.loads(Path(f_leak).read_text()))
    out["lens_path"] = lens_path
    out["t_total_s"] = round(time.time() - t0, 1)
    Path(f"{OUT2}/stage2a.json").write_text(json.dumps(out, indent=1, default=float))
    VOLUMES["/results"].commit()
    return {k: v for k, v in out.items() if k not in ("slots", "leak", "g4")} | {"labels": {s_: v["label"] for s_, v in out["slots"].items()}}


@app.function(image=app_image, volumes=VOLUMES, gpu="A10G", timeout=3600, memory=65536)
def jpp_enrich() -> dict:
    """Second-basis check: enrichment of every stage-1 and stage-2a direction under a J++ basis (layer 42)."""
    import numpy as np
    import torch

    import o10_lib as OL
    from mats.jspace import jspace_basis, load_basis_file

    t0 = time.time()
    VOLUMES["/results"].reload()
    VOLUMES["/weights"].reload()
    p = _jpp_path()
    if p is None:
        return {"skipped": "no J++ lens file on the volume"}
    obj = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    cfg = obj.get("config")
    meta = {"source_layers": str(obj.get("source_layers"))[:300], "top_level_keys": list(obj.keys()),
            "config": ({k: str(v)[:200] for k, v in cfg.items()} if isinstance(cfg, dict) else str(cfg)[:500])}
    # koayon/jpp-lenses layout (inspected 2026-10-12): {"parameters": {"jacobians": {"<layer>": [5120, 5120]}},
    # "source_layers": [0..62], "config": {hf_model_name: Qwen/Qwen3.6-27B, d_model: 5120, ...}}
    jac = obj["parameters"]["jacobians"] if "parameters" in obj else obj["jacobians"]
    Jpp = (jac[L] if L in jac else jac[str(L)]).float().clone().cuda()
    meta["J_shape"] = list(Jpp.shape)
    del obj, jac
    W_U, gain = _unembed()
    bpp = jspace_basis(Jpp, W_U, gain, r_max=1536, layer=L)
    b0 = load_basis_file(BASIS_L42, "cuda")
    Qs_pp = {r: bpp.Q(r) for r in OL.RANKS}
    Qs_0 = {r: b0.Q(r) for r in OL.RANKS}
    out: dict = {"jpp_meta": meta, "overlap_with_n1000": {str(r): float((bpp.Q(r).T @ b0.Q(r)).pow(2).sum() / r) for r in OL.RANKS},
                 "random_band_jpp": OL.random_band(Qs_pp, b0.d), "dirs": {}}

    def both(v: torch.Tensor) -> dict:
        v = v.cuda().float()
        e = {}
        for r in OL.RANKS:
            e[str(r)] = {"n1000": float(np.median(OL.enrich_rows(Qs_0[r], v.reshape(-1, v.shape[-1])))),
                         "jpp": float(np.median(OL.enrich_rows(Qs_pp[r], v.reshape(-1, v.shape[-1]))))}
        return e

    for tag, path in (("stage1_bare", f"{OUT}/dirs.pt"), ("stage2a_fewshot", f"{OUT2}/dirs_fewshot.pt")):
        if not Path(path).exists():
            out["dirs"][tag] = "missing"
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        res: dict = {}
        pool = d["slots"] if "slots" in d else d
        for k, v in pool.items():
            if isinstance(v, dict) and "class_mean" in v:
                res[k] = {kk: both(v[kk]) for kk in ("class_mean", "probe_weight") if kk in v}
            elif torch.is_tensor(v) and k.startswith("mag_"):
                res.setdefault("magnitude", {})[k] = both(v)
        if "g4" in d:
            res["g4_dirs"] = {f: {k: both(v) for k, v in d["g4"][f]["dirs"].items()} for f in d["g4"]}
        out["dirs"][tag] = res
    out["t_s"] = round(time.time() - t0, 1)
    Path(f"{OUT2}/jpp.json").write_text(json.dumps(out, indent=1, default=float))
    VOLUMES["/results"].commit()
    return out


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(operand_stage2a.remote(), indent=1, default=str))
