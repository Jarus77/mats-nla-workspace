"""05 (C) planted intermediates: yield check (K4).

Stages (app ``mats-05-planted``):
  gate   (H100) : every item's own answer, greedy-consistency gate; capture block outputs at the
                  read token for PLANT_LAYERS + 42 (unplanted).
  sweep  (H100, .map over planting layers): per gated item, a pre-assigned partner bridge Y;
                  plant v = alpha (mu_Y - mu_X) norm-preserving at block L, read-token..end;
                  record answer flips (greedy-consistent Y alias), X still produced, next-token
                  watch log-probs, neutral-prompt KL with the same edit, and the planted L42 vector.
  readers (H100): OL@42 + J-Lens@42 paired LO (Y vs X) on kept items: unplanted, planted, and
                  K equal-norm random-direction plants (null band).
  nla     (H200): NLA@42 paired LO on the same vectors.

    modal run plant_modal.py --stage gate
    modal run plant_modal.py --stage sweep
    python analyze.py            # -> results/yield.json, keep.json
    modal run plant_modal.py --stage readers ; modal run plant_modal.py --stage nla
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from env import OUT, VOLUMES, app_image

app = modal.App("mats-05-planted")
PLANT_LAYERS = (24, 28, 34, 38, 42)
READ_LAYER = 42
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
MODEL = "Qwen/Qwen3.6-27B"


def _load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def _items():
    from plant_items import build_items

    bank = json.loads(Path("/root/multihop_items.json").read_text())["items"]
    return build_items(bank)


def _watch_ids(tok, items):
    from plant_items import TABLE, answers_of
    from plant_lib import alias_ids

    ids = set()
    for it in items:
        for c in TABLE:
            for a in answers_of(c, it["relation"]) or []:
                ids.add(alias_ids(tok, it["prompt"], a)[0])
    return sorted(ids)


# ============================================================================ gate


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=2 * 3600, memory=131072)
def stage_gate() -> dict:
    import numpy as np
    import torch

    from plant_lib import Row, alias_ids, run_rows

    t0 = time.time()
    VOLUMES["/results"].reload()
    model, tok = _load_model()
    items = _items()
    rows, owner = [], []
    for i, it in enumerate(items):
        p = tok.encode(it["prompt"], add_special_tokens=False)
        read = len(p) + it["read_offset"]
        for j, a in enumerate(it["answers"]):
            rows.append(Row(ids=p + alias_ids(tok, it["prompt"], a), plen=len(p), vec=-1,
                            edit_pos=(), cap_pos=read if j == 0 else -1))
            owner.append(i)
    layers = sorted({*PLANT_LAYERS, READ_LAYER})
    res = run_rows(model, rows, None, None, cap_layers=layers, batch=64,
                   pad_id=tok.pad_token_id or 0)
    owner = np.array(owner)
    gate = np.array([bool(res["greedy"][owner == i].any()) for i in range(len(items))])
    first = np.array([int(np.where(owner == i)[0][0]) for i in range(len(items))])
    acts = {L: res["caps"][L][first] for L in layers}
    top1 = [tok.decode([int(res["top1"][f])]) for f in first]
    Path(OUT).mkdir(parents=True, exist_ok=True)
    torch.save({"items": items, "gate": gate, "acts": acts, "top1": top1,
                "p_top1": res["p_top1"][first]}, f"{OUT}/gate.pt")
    VOLUMES["/results"].commit()
    by_src = {s: [int(gate[[k for k, it in enumerate(items) if it["source"] == s]].sum()),
                  sum(it["source"] == s for it in items)] for s in ("cons", "bank")}
    fails = [(it["id"], it["answers"][0], top1[k]) for k, it in enumerate(items) if not gate[k]]
    return {"gate_pass": by_src, "secs": round(time.time() - t0), "fails_sample": fails[:60]}


# ============================================================================ sweep


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=3 * 3600, memory=131072)
def stage_sweep(job: tuple[int, str]) -> dict:
    layer, mode = job
    import numpy as np
    import torch

    from plant_items import NEUTRAL_PROMPTS, TABLE, answers_of
    from plant_lib import Row, alias_ids, assign_partners, centred, item_means, relation_means, run_rows

    t0 = time.time()
    VOLUMES["/results"].reload()
    g = torch.load(f"{OUT}/gate.pt", weights_only=False)
    items, gate = g["items"], g["gate"]
    model, tok = _load_model()
    partners = assign_partners(items, gate, tok, TABLE)
    idx = [i for i, p in enumerate(partners) if p is not None]
    A = torch.tensor(g["acts"][layer]).float()
    rm = relation_means(A, items, gate)
    cen = centred(A, items, rm)
    dirs, nat_sd, cxy = [], [], []
    for i in idx:
        mx, my = item_means(cen, items, gate, i, partners[i])
        d = my - mx
        dirs.append(d)
        uu = d / d.norm()
        r0 = rm[items[i]["relation"]]
        cxy.append((float((r0 + mx) @ uu), float((r0 + my) @ uu)))
        # natural spread of X's class along d-hat (constructed, gated, own clue included)
        cls = [k for k, it in enumerate(items) if gate[k] and it["source"] == "cons"
               and it["country"] == items[i]["country"]]
        u = d / d.norm()
        nat_sd.append(float((cen[cls] @ u).std()))
    D = torch.stack(dirs)  # [n, d]
    n, nA = len(idx), len(ALPHAS)
    U = D / D.norm(dim=1, keepdim=True)
    cxy_t = torch.tensor(cxy)  # [n, 2] natural X / Y class-mean projections (raw coords)
    hL = A[idx]
    if mode == "add":  # additive, norm-preserving: v = alpha (mu_Y - mu_X)
        vecs = torch.cat([a * D for a in ALPHAS])  # vec index = ai * n + k
        targets = None
        delta_ratio = torch.stack([a * D.norm(dim=1) / hL.norm(dim=1) for a in ALPHAS], 1)
    elif mode == "clamp":  # centred swap: set u.h to c_X + alpha (c_Y - c_X); alpha=1 -> Y mean
        vecs = torch.cat([U for _ in ALPHAS])
        tg = [cxy_t[:, 0] + a * (cxy_t[:, 1] - cxy_t[:, 0]) for a in ALPHAS]
        targets = torch.cat(tg)
        delta_ratio = torch.stack([(t - (hL * U).sum(1)).abs() / hL.norm(dim=1) for t in tg], 1)
    else:
        raise ValueError(mode)
    watch = _watch_ids(tok, items)
    wpos = {t: j for j, t in enumerate(watch)}

    # main rows: Y aliases + X aliases per (item, alpha); capture L42 on the first row
    rows, meta = [], []
    for k, i in enumerate(idx):
        it = items[i]
        p = tok.encode(it["prompt"], add_special_tokens=False)
        read = len(p) + it["read_offset"]
        epos = tuple(range(read, len(p)))
        ya, xa = answers_of(partners[i], it["relation"]), it["answers"]
        for ai in range(nA):
            for j, (side, a) in enumerate([("y", a) for a in ya] + [("x", a) for a in xa]):
                rows.append(Row(ids=p + alias_ids(tok, it["prompt"], a), plen=len(p),
                                vec=ai * n + k, edit_pos=epos, cap_pos=read if j == 0 else -1))
                meta.append((k, ai, side))
    res = run_rows(model, rows, vecs, layer, cap_layers=[READ_LAYER],
                   watch_ids=torch.tensor(watch), batch=64, pad_id=tok.pad_token_id or 0,
                   targets=targets)
    gy = torch.zeros(n, nA, dtype=torch.bool)
    gx = torch.zeros(n, nA, dtype=torch.bool)
    lpy = torch.full((n, nA), -1e9)
    lpx = torch.full((n, nA), -1e9)
    top1 = torch.zeros(n, nA, dtype=torch.long)
    ptop1 = torch.zeros(n, nA)
    wlp = torch.zeros(n, nA, len(watch))
    h42 = torch.zeros(n, nA, A.shape[1], dtype=torch.float16)
    for r, (k, ai, side) in enumerate(meta):
        if side == "y":
            gy[k, ai] |= bool(res["greedy"][r])
            lpy[k, ai] = max(float(lpy[k, ai]), float(res["cont_lp"][r]))
        else:
            gx[k, ai] |= bool(res["greedy"][r])
            lpx[k, ai] = max(float(lpx[k, ai]), float(res["cont_lp"][r]))
        if rows[r].cap_pos >= 0:
            h42[k, ai] = res["caps"][READ_LAYER][r]
            top1[k, ai], ptop1[k, ai] = res["top1"][r], res["p_top1"][r]
            wlp[k, ai] = res["watch_lp"][r]

    # neutral-prompt KL with the same edit at the last token
    neu = [tok.encode(s, add_special_tokens=False) for s in NEUTRAL_PROMPTS]
    clean = run_rows(model, [Row(ids=s, plen=len(s), vec=-1, edit_pos=(), cap_pos=-1) for s in neu],
                     None, None, watch_ids=None, batch=16)
    # reference log-probs: recompute full distributions
    ref = []
    with torch.no_grad():
        for s in neu:
            lg = model(input_ids=torch.tensor([s], device="cuda")).logits[0, -1].float()
            ref.append(torch.log_softmax(lg, -1))
    ref_lp = torch.stack(ref)
    nrows = [Row(ids=s, plen=len(s), vec=ai * n + k, edit_pos=(len(s) - 1,), cap_pos=-1, ref=j)
             for k in range(n) for ai in range(nA) for j, s in enumerate(neu)]
    nres = run_rows(model, nrows, vecs, layer, ref_lp=ref_lp, batch=128,
                    pad_id=tok.pad_token_id or 0, targets=targets)
    kl = nres["kl"].reshape(n, nA, len(neu))

    hn = A[idx].norm(dim=1)
    payload = {
        "layer": layer, "mode": mode, "alphas": ALPHAS, "delta_ratio": delta_ratio,
        "U": U.half(), "cxy": cxy_t, "idx": idx, "partners": [partners[i] for i in idx],
        "greedy_y": gy, "greedy_x": gx, "lp_y": lpy, "lp_x": lpx, "top1": top1, "p_top1": ptop1,
        "watch": watch, "watch_lp": wlp, "kl_neutral": kl, "h42": h42,
        "dnorm": D.norm(dim=1), "hnorm": hn, "nat_sd": torch.tensor(nat_sd),
        "clean_top1_neutral": [tok.decode([int(t)]) for t in clean["top1"]],
        "top1_str": None,
    }
    torch.save(payload, f"{OUT}/sweep_{mode}_L{layer}.pt")
    VOLUMES["/results"].commit()
    flip = gy & ~gx
    return {"layer": layer, "mode": mode, "n_items": n,
            "delta_ratio_median_by_alpha": [float(delta_ratio[:, a].median()) for a in range(nA)], "secs": round(time.time() - t0),
            "flip_by_alpha": [int(flip[:, a].sum()) for a in range(nA)],
            "dnorm_over_hnorm_median": float((D.norm(dim=1) / hn).median()),
            "kl_median_by_alpha": [float(kl[:, a].median()) for a in range(nA)]}



# ============================================================================ readers

K_NULL = 20  # per null class (isotropic, J-complement)
MAX_READ_ITEMS = 200
OL_T = ("- {c}", "- {c} (country)", "- The country of {c}")
NLA_T = ("{c}", "The text is about {c}", "A factual question involving {c}")


def _rstar(L: int) -> int:
    p = Path("/results/02_nonj_features/rstar.json")
    if p.exists():
        lay = json.loads(p.read_text())["layers"]
        if str(L) in lay and lay[str(L)].get("r_star"):
            return int(lay[str(L)]["r_star"])
    return {38: 192, 40: 384, 42: 384, 44: 256}.get(L, 384)


@app.function(image=app_image, volumes=VOLUMES, gpu="H100", timeout=4 * 3600, memory=131072)
def stage_readers(keep: dict) -> dict:
    """Planted / unplanted / null L42 vectors for kept items -> OL@42 and J-Lens@42 paired LO."""
    import numpy as np
    import torch

    from mats import jspace as JS
    from mats.readers.jlens import load_jlens
    from mats.readers.olens import load_olens
    from plant_items import TABLE
    from plant_lib import Row, run_rows

    t0 = time.time()
    VOLUMES["/results"].reload()
    g = torch.load(f"{OUT}/gate.pt", weights_only=False)
    mode, L = keep["mode"], int(keep["layer"])
    sw = torch.load(f"{OUT}/sweep_{mode}_L{L}.pt", weights_only=False)
    items = g["items"]
    rng = np.random.default_rng(0)
    kept = keep["items"]
    if len(kept) > MAX_READ_ITEMS:
        kept = [kept[j] for j in sorted(rng.choice(len(kept), MAX_READ_ITEMS, replace=False))]
    n = len(kept)
    model, tok = _load_model()
    ol = load_olens(model, tok)
    jl = load_jlens(ol.model, tok, layers=sorted({L, READ_LAYER}))
    bp = Path(f"/results/03_pipeline_smoke/jspace_basis_L{L}.pt")
    basis = JS.load_basis_file(str(bp)) if bp.exists() else JS.jspace_basis(
        jl.J[L], jl.W_U, JS.norm_gain(jl.final_norm), r_max=2048, layer=L).to("cpu")
    r = _rstar(L)
    # null directions at the planting layer: K isotropic + K J-complement (top-r* removed)
    nulls = [JS.random_direction(basis, r, "iso", seed=1000 + k).cpu() for k in range(K_NULL)]
    nulls += [JS.random_direction(basis, r, "nonj", seed=2000 + k).cpu() for k in range(K_NULL)]
    Nd = torch.stack(nulls)  # [2K, d]
    hL = torch.as_tensor(g["acts"][L]).float()
    U = sw["U"].float()
    enr = [float(JS.enrichment(basis, U[c["k"]], r)) for c in kept]
    # realised displacement norm of each kept plant at layer L; null = additive delta * r (+/- sign)
    vecs, rows = [], []
    signs = torch.tensor(rng.choice([-1.0, 1.0], size=(n, len(nulls))))
    for j, c in enumerate(kept):
        it = items[c["item"]]
        delta = float(sw["delta_ratio"][c["k"], c["ai"]]) * float(hL[c["item"]].norm())
        p = tok.encode(it["prompt"], add_special_tokens=False)
        read = len(p) + it["read_offset"]
        for q in range(len(nulls)):
            vecs.append(float(signs[j, q]) * delta * Nd[q])
            rows.append(Row(ids=p, plen=len(p), vec=len(vecs) - 1, edit_pos=tuple(range(read, len(p))),
                            cap_pos=read))
    with ol.base():
        res = run_rows(ol.model, rows, torch.stack(vecs), L, cap_layers=[READ_LAYER], batch=64,
                       pad_id=tok.pad_token_id or 0, targets=None, renorm=(mode == "add"))
    # nulls match the planting construction: clamp = plain additive displacement of the same norm
    # along a random direction; add = norm-preserving additive.
    null42 = res["caps"][READ_LAYER].float().reshape(n, len(nulls), -1)
    nat42 = torch.stack([torch.as_tensor(g["acts"][READ_LAYER][c["item"]]).float() for c in kept])
    pl42 = torch.stack([sw["h42"][c["k"], c["ai"]].float() for c in kept])
    acts = torch.cat([nat42[:, None], pl42[:, None], null42], 1)  # [n, 2+2K, d]
    C = acts.shape[1]
    names = [(TABLE[sw["partners"][c["k"]]]["name"], TABLE[items[c["item"]]["country"]]["name"]) for c in kept]
    # OL: paired LO per template, Y vs X
    lo_ol = torch.zeros(len(OL_T), n, C)
    for t, tpl in enumerate(OL_T):
        for j, (yn, xn) in enumerate(names):
            lo_ol[t, j] = ol.paired_logodds(acts[j], READ_LAYER, tpl.format(c=yn), tpl.format(c=xn))
    lo_jl = torch.zeros(n, C)
    for j, c in enumerate(kept):
        yw = TABLE[sw["partners"][c["k"]]]["jl"]
        xw = TABLE[items[c["item"]]["country"]]["jl"]
        lo_jl[j] = jl.paired_logodds(acts[j], READ_LAYER, yw, xw)
    payload = {"keep": kept, "mode": mode, "layer": L, "acts": acts.half(), "names": names,
               "lo_ol": lo_ol, "lo_jl": lo_jl, "null_kinds": ["iso"] * K_NULL + ["jc"] * K_NULL,
               "enrichment_u": enr, "r_star": r, "ol_templates": OL_T}
    torch.save(payload, f"{OUT}/readers_base.pt")
    VOLUMES["/results"].commit()
    return {"n": n, "secs": round(time.time() - t0), "enrichment_u_median": float(np.median(enr))}


@app.function(image=app_image, volumes=VOLUMES, gpu="H200", timeout=4 * 3600, memory=131072)
def stage_nla() -> dict:
    import torch

    from mats.readers import nla as NL

    t0 = time.time()
    VOLUMES["/results"].reload()
    b = torch.load(f"{OUT}/readers_base.pt", weights_only=False)
    nla = NL.load_nla(adapter=NL.DEFAULT_ADAPTER, load_ar=False)
    acts, names = b["acts"].float(), b["names"]
    n, C, _ = acts.shape
    lo = torch.zeros(len(NLA_T), n, C)
    for t, tpl in enumerate(NLA_T):
        for j, (yn, xn) in enumerate(names):
            lo[t, j] = NL.paired_logodds_batch(nla, acts[j], tpl.format(c=yn), tpl.format(c=xn))
    # a few free-form explanations (natural vs planted) for qualitative inspection
    sel = list(range(min(12, n)))
    ex_nat = nla.verbalize(acts[sel, 0], k=1)
    ex_pl = nla.verbalize(acts[sel, 1], k=1)
    torch.save({"lo_nla": lo, "nla_templates": NLA_T, "ex_nat": ex_nat, "ex_pl": ex_pl},
               f"{OUT}/readers_nla.pt")
    VOLUMES["/results"].commit()
    return {"n": n, "secs": round(time.time() - t0)}

# ============================================================================ entry


@app.local_entrypoint()
def main(stage: str = "gate", layers: str = "", modes: str = "add,clamp") -> None:
    import subprocess

    here = Path(__file__).resolve().parent / "results"
    here.mkdir(exist_ok=True)
    if stage == "gate":
        r = stage_gate.remote()
        print(json.dumps(r, indent=1, default=str))
        (here / "gate_summary.json").write_text(json.dumps(r, indent=1, default=str))
        files = ["gate.pt"]
    elif stage == "sweep":
        Ls = [int(x) for x in layers.split(",")] if layers else list(PLANT_LAYERS)
        jobs = [(L, m) for L in Ls for m in modes.split(",")]
        out = list(stage_sweep.map(jobs))
        for r in out:
            print(json.dumps(r, default=str))
        (here / "sweep_summary.json").write_text(json.dumps(out, indent=1, default=str))
        files = [f"sweep_{m}_L{L}.pt" for L, m in jobs]
    elif stage == "readers":
        keep = json.loads((here / "keep.json").read_text())
        print(json.dumps(stage_readers.remote(keep)))
        files = ["readers_base.pt"]
    elif stage == "nla":
        print(json.dumps(stage_nla.remote()))
        files = ["readers_nla.pt"]
    else:
        raise SystemExit(f"unknown stage {stage}")
    for f in files:
        subprocess.run(["modal", "volume", "get", "--force", "mats-results",
                        f"05_planted_intermediates/{f}", str(here / f)], check=False)
