"""04 backup features, G4: causal use + leak pre-check + self-report.

Steering (D11): CENTRED steering at block-L output, every position of the context:
    h' = h - (v.h) v + a v,  a = +/-dose
(equal norm on both sides; ``mats.capture.centred_steer_vectors`` formula applied inside the
forward pass). v = J-REMOVED diff-in-means direction (top-1536 J-space projected out; r* variant
too; raw reported), 2-fold cross-fit by group (direction from the other fold). Doses = p90/p99 of
|proj - median| of the natural projection (02 convention).
Nulls at the p99 dose of jrem1536, same centred construction: K=24 J-complement random directions
(random, top-1536 J-space removed) and K=20 isotropic random directions.
Effect per item e_i = (m(+a) - m(-a))/2, m = behaviour metric averaged over the item's eval positions.
Leak (steer L38, read 40/42/44): J-Lens class log-rank contrast LO = S1 - S0 (standard readout,
02 label tokens) at the feature's read positions; AUROC(LO+ vs LO-) and mean shift vs the
J-complement band (two-sided), z reported.

    modal run bf_causal.py [--only indent,sent_pos] [--layers 38,42]
"""

from __future__ import annotations

import modal

from env import OUT, VOLUMES, gpu_image

app = modal.App("mats-04-backup-causal")
K_JC, K_ISO = 24, 20
RSTAR = {38: 192, 42: 384}
READ = {38: [40, 42, 44], 42: [44]}


def behaviour_fn(name: str, tok, V: int):  # noqa: ANN001, ANN201
    import torch

    from bf_feats import JS_MARKERS, PY_MARKERS

    vocab = [tok.decode([i]) for i in range(V)]

    def ids(pred) -> torch.Tensor:  # noqa: ANN001
        return torch.tensor([i for i, s in enumerate(vocab) if pred(s)], device="cuda")

    def single(words: list[str]) -> torch.Tensor:
        out = []
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            if len(e) == 1:
                out.append(e[0])
        return torch.tensor(sorted(set(out)), device="cuda")

    def logit(lse: torch.Tensor) -> torch.Tensor:
        return lse - torch.log1p(-lse.exp().clamp(max=1 - 1e-6))

    if name == "next_indent":
        sp = ids(lambda s: len(s) > 0 and set(s) == {" "})
        nos = ids(lambda s: len(s) > 0 and not s[0].isspace())
        ind = torch.tensor([len(vocab[i]) + 1 for i in sp.tolist()], device="cuda", dtype=torch.float32)

        def f(lp: torch.Tensor) -> torch.Tensor:
            p_sp, p_no = lp[..., sp].exp(), lp[..., nos].exp().sum(-1)
            return (p_sp * ind).sum(-1) / (p_sp.sum(-1) + p_no).clamp_min(1e-9)
        return f
    if name == "indent8_vs_4":  # POST-HOC (after next_indent failed): log P(8-space) - log P(4-space) next-line indent
        a = torch.tensor([i for i, s in enumerate(vocab) if s == " " * 7], device="cuda")
        b = torch.tensor([i for i, s in enumerate(vocab) if s == " " * 3], device="cuda")
        return lambda lp: lp[..., a].logsumexp(-1) - lp[..., b].logsumexp(-1)
    if name == "sent_end_logit":
        se = ids(lambda s: len(s) > 0 and s[0] in ".!?" and not s.startswith(".."))
        return lambda lp: logit(torch.logsumexp(lp[..., se], -1))
    if name == "next_list_number":
        dig = single([str(k) for k in range(1, 10)])
        val = torch.tensor([float(vocab[i]) for i in dig.tolist()], device="cuda")

        def f(lp: torch.Tensor) -> torch.Tensor:
            p = lp[..., dig].exp()
            return (p * val).sum(-1) / p.sum(-1).clamp_min(1e-9)
        return f
    if name in ("close_vs_open_quote", "fem_vs_masc"):
        import json

        from bf_vocab import emitted_sets
        glex = json.load(open(f"{OUT}/data/gender_lex.json")) if name == "fem_vs_masc" else None
        s0, s1 = emitted_sets("in_quote" if name == "close_vs_open_quote" else "gender_next", vocab, tok, glex)
        a, b = torch.tensor(s1, device="cuda"), torch.tensor(s0, device="cuda")
        return lambda lp: torch.logsumexp(lp[..., a], -1) - torch.logsumexp(lp[..., b], -1)
    if name == "cap_upper_vs_lower":
        up = ids(lambda s: len(s) > 1 and s[0] == " " and s[1].isalpha() and s[1].isupper())
        lo = ids(lambda s: len(s) > 1 and s[0] == " " and s[1].isalpha() and s[1].islower())
        return lambda lp: torch.logsumexp(lp[..., up], -1) - torch.logsumexp(lp[..., lo], -1)
    if name == "py_vs_js":
        a, b = single(PY_MARKERS), single(JS_MARKERS)
        return lambda lp: torch.logsumexp(lp[..., a], -1) - torch.logsumexp(lp[..., b], -1)
    raise ValueError(name)


def _summ(m: dict, names: list[str], n_items: int) -> dict:
    """Effect statistics of every steered direction vs the J-complement / isotropic bands."""
    import numpy as np
    from scipy.stats import wilcoxon

    from mats.metrics import null_band

    def eff(key: str) -> np.ndarray:
        return (m[key + "+"] - m[key + "-"]) / 2

    jc = {f"jc{k}": eff(f"jc{k}") for k in range(K_JC)}
    iso = {f"iso{k}": eff(f"iso{k}") for k in range(K_ISO)}
    jcE = np.array([v.mean() for v in jc.values()])
    isoE = np.array([v.mean() for v in iso.values()])
    kstar = list(jc)[int(np.argmax(np.abs(jcE)))]
    out = {"n_items": n_items, "jc_abs_max": float(np.abs(jcE).max()), "iso_abs_max": float(np.abs(isoE).max())}
    rng = np.random.default_rng(0)
    for nm in names:
        e = eff(nm)
        E = float(e.mean())
        boot = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(2000)]
        out[nm] = {
            "effect": E, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "paired_win": float(np.mean(m[nm + "+"] > m[nm + "-"])),
            "ratio_vs_jc_absmax": abs(E) / max(out["jc_abs_max"], 1e-12),
            "band_jc": null_band(E, jcE), "band_iso": null_band(E, isoE),
            "wilcoxon_p_vs_strongest_jc": float(wilcoxon(e, jc[kstar]).pvalue),
        }
    return out


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=6 * 3600, memory=196608)
def stage_causal(fname: str, layer: int, n_items: int = 260, behaviour: str = "") -> dict:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import json
    import pickle

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import bf_gate_lib as G
    from bf_feats import FEATURES_BY_NAME
    from bf_modal import _load_lens_and_unembed
    from mats.capture import decoder_blocks
    from mats.jspace import load_basis_file
    from mats.metrics import null_band, paired_stats
    from mats.readers.jlens import token_variants

    VOLUMES["/results"].reload()
    f = FEATURES_BY_NAME[fname]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda").eval()
    blocks = decoder_blocks(model)
    Js, W_U, gain, _ = _load_lens_and_unembed()
    lab_ids = [token_variants(tok, list(ws)) for ws in f.label_words]
    beh = behaviour or f.behaviour
    tag = f"_{behaviour}" if behaviour else ""
    metric = behaviour_fn(beh, tok, W_U.shape[0])
    items = [it for it in pickle.load(open(f"{OUT}/data/{f.dataset}.pkl", "rb"))
             if fname in it["feats"] and it["eval"].get(fname)]
    rng = np.random.default_rng(1)
    items = [items[i] for i in rng.permutation(len(items))[:n_items]]
    D = torch.load(f"{OUT}/gate/{fname}_dirs.pt", weights_only=False)
    Q = load_basis_file(f"/results/03_pipeline_smoke/jspace_basis_L{layer}.pt", "cuda").Q(1536)
    g = torch.Generator(device="cpu").manual_seed(1000 + layer)
    R = torch.randn(K_JC + K_ISO, Q.shape[0], generator=g).cuda()
    R[:K_JC] = R[:K_JC] - (R[:K_JC] @ Q) @ Q.T
    R = R / R.norm(dim=1, keepdim=True)
    names = [f"jrem1536_{d}" for d in ("p90", "p99")] + [f"jrem{RSTAR[layer]}_{d}" for d in ("p90", "p99")] + \
            [f"raw_{d}" for d in ("p90", "p99")]

    def conds(fo: int) -> list[tuple[str, torch.Tensor | None, float]]:
        dd = D[(layer, fo)]
        cs: list = [("none", None, 0.0)]
        for nm in names:
            key, dn = nm.rsplit("_", 1)
            for s in (1, -1):
                cs.append((nm + ("+" if s > 0 else "-"), dd["dirs"][key].cuda(), s * dd["doses"][key][dn]))
        a = dd["doses"]["jrem1536"]["p99"]
        for k in range(K_JC + K_ISO):
            nm = f"jc{k}" if k < K_JC else f"iso{k - K_JC}"
            for s in (1, -1):
                cs.append((nm + ("+" if s > 0 else "-"), R[k], s * a))
        return cs

    state: dict = {"V": None, "A": None, "on": None}

    def steer_hook(_m, _i, o):  # noqa: ANN001, ANN202
        h = o[0] if isinstance(o, tuple) else o
        V, A, on = state["V"], state["A"], state["on"]
        hf = h.float()
        proj = (hf * V[:, None, :]).sum(-1, keepdim=True)
        new = hf - proj * V[:, None, :] + A[:, None, None] * V[:, None, :]
        h2 = torch.where(on[:, None, None], new, hf).to(h.dtype)
        return (h2, *o[1:]) if isinstance(o, tuple) else h2

    cap: dict = {}
    hooks = [blocks[layer].register_forward_hook(steer_hook)]
    for Lr in READ[layer]:
        hooks.append(blocks[Lr].register_forward_hook(
            lambda _m, _i, o, Lr=Lr: cap.__setitem__(Lr, (o[0] if isinstance(o, tuple) else o).detach())))
    M: dict = {}
    LO: dict = {Lr: {} for Lr in READ[layer]}
    KL: dict = {}
    for ii, it in enumerate(items):
        fo = D["fold_of_group"][int(it["group"])]
        cs = conds(fo)
        ev = torch.tensor(it["eval"][fname], device="cuda")
        rp = it["feats"][fname]["pos"]
        T = len(it["ids"])
        bs = max(4, min(len(cs), 9000 // T))
        res_m, res_lp0, res_lo = [], None, {Lr: [] for Lr in READ[layer]}
        for s in range(0, len(cs), bs):
            ch = cs[s:s + bs]
            state["V"] = torch.stack([c[1] if c[1] is not None else torch.zeros(Q.shape[0], device="cuda") for c in ch])
            state["A"] = torch.tensor([c[2] for c in ch], device="cuda")
            state["on"] = torch.tensor([c[1] is not None for c in ch], device="cuda")
            ids = torch.tensor([it["ids"]] * len(ch), device="cuda")
            with torch.no_grad():
                lg = model(input_ids=ids, logits_to_keep=ev).logits.float()
            lp = torch.log_softmax(lg, -1)
            if s == 0:
                res_lp0 = lp[0:1]
            kl = (res_lp0.exp() * (res_lp0 - lp)).sum(-1).mean(-1)
            res_m.append(torch.stack([metric(lp), kl[:, None].expand(-1, lp.shape[1])], 0).mean(-1).cpu())
            for Lr in READ[layer]:
                H = cap[Lr][:, rp].reshape(-1, cap[Lr].shape[-1])
                rk, _ = G.jlens_ranks_topk(H, Js[Lr], W_U, gain, lab_ids, k=1)
                S = -np.log10(rk.astype(np.float64) + 1)
                res_lo[Lr].append((S[:, 1] - S[:, 0]).reshape(len(ch), len(rp)).mean(1))
        mm = torch.cat(res_m, 1).numpy()
        for j, c in enumerate(cs):
            M.setdefault(c[0], []).append(mm[0, j])
            KL.setdefault(c[0], []).append(mm[1, j])
            for Lr in READ[layer]:
                LO[Lr].setdefault(c[0], []).append(np.concatenate(res_lo[Lr])[j])
        if ii % 40 == 0:
            print(fname, layer, ii, len(items), flush=True)
    for h in hooks:
        h.remove()
    M = {k: np.array(v) for k, v in M.items()}
    out = {"feature": fname, "layer": layer, "behaviour": beh, "steering": "centred",
           "g4": _summ(M, names, len(items)),
           "kl_mean": {k: float(np.mean(v)) for k, v in KL.items() if k[:2] not in ("jc", "is")},
           "kl_mean_jc": float(np.mean([np.mean(v) for k, v in KL.items() if k.startswith("jc")])),
           "dose_over_median_norm": {f"{key}_{dn}": float(np.mean([D[(layer, fo)]["doses"][key][dn] / D[(layer, fo)]["median_norm"] for fo in (0, 1)]))
                                     for key in ("jrem1536", f"jrem{RSTAR[layer]}", "raw") for dn in ("p90", "p99")},
           "leak": {}}
    for Lr in READ[layer]:
        lo = {k: np.array(v) for k, v in LO[Lr].items()}
        nulls = [paired_stats(lo[f"jc{k}+"], lo[f"jc{k}-"]) for k in range(K_JC)]
        lk = {}
        for nm in names:
            r = paired_stats(lo[nm + "+"], lo[nm + "-"])
            bands = {key: null_band(r[key], [x[key] for x in nulls]) for key in ("auroc", "mean_shift")}
            lk[nm] = {"auroc": r["auroc"], "mean_shift": r["mean_shift"], "band_auroc": bands["auroc"],
                      "band_mean_shift": bands["mean_shift"],
                      "leak": bool(bands["auroc"]["outside_two_sided"] or bands["mean_shift"]["outside_two_sided"])}
        out["leak"][str(Lr)] = lk
    torch.save({"M": M, "LO": LO, "KL": KL}, f"{OUT}/causal/{fname}_L{layer}{tag}_raw.pt") if _mk(f"{OUT}/causal") else None
    json.dump(out, open(f"{OUT}/causal/{fname}_L{layer}{tag}.json", "w"), indent=1, default=float)
    if layer == 38 and not behaviour:
        out["selfreport"] = selfreport(model, tok, f, items)
        json.dump(out, open(f"{OUT}/causal/{fname}_L{layer}{tag}.json", "w"), indent=1, default=float)
    VOLUMES["/results"].commit()
    return out


def _mk(d: str) -> bool:
    import os

    os.makedirs(d, exist_ok=True)
    return True


def selfreport(model, tok, f, items) -> dict:  # noqa: ANN001
    """Logit 2AFC self-report (reported, not gated): answer-token logit difference vs true label."""
    import numpy as np

    from mats.capture import logits_at
    from mats.metrics import auroc_ci

    seqs, ys, gs = [], [], []
    for it in items:
        p, y = it["feats"][f.name]["pos"][0], it["feats"][f.name]["y"][0]
        prefix = tok.decode(it["ids"][: p + 1])[-1500:]
        msg = f"Here is a text:\n<text>\n{prefix}\n</text>\n{f.question} Answer with just {f.answers[1]} or {f.answers[0]}."
        s = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        seqs.append(tok.encode(s, add_special_tokens=False))
        ys.append(y)
        gs.append(it["group"])
    a1, a0 = (tok.encode(a, add_special_tokens=False)[0] for a in (f.answers[1], f.answers[0]))
    lg = logits_at(model, seqs, [[-1]] * len(seqs), batch_size=8)[:, 0]
    sc = (lg[:, a1] - lg[:, a0]).numpy()
    ci = auroc_ci(sc, np.array(ys), np.array(gs), 1000)
    return {"auroc": ci.auroc, "ci95": [ci.lo, ci.hi], "acc": float(((sc > 0) == (np.array(ys) == 1)).mean()),
            "n": len(ys), "frac_answer1": float((sc > 0).mean())}


@app.local_entrypoint()
def main(only: str = "", layers: str = "38,42", behaviour: str = "") -> None:
    from bf_feats import FEATURES

    fs = only.split(",") if only else [f.name for f in FEATURES]
    jobs = [(fn, int(L), 260, behaviour) for fn in fs for L in layers.split(",")]
    for r in stage_causal.starmap(jobs, return_exceptions=True):
        if isinstance(r, dict):
            g = r["g4"]["jrem1536_p99"]
            print(r["feature"], r["layer"], "E", round(g["effect"], 4), "ratio", round(g["ratio_vs_jc_absmax"], 1),
                  "zjc", round(g["band_jc"]["z"], 1))
        else:
            print("ERR", r)
