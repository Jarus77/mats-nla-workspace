"""Yield analysis for (C) planted intermediates (K4). Local, CPU.

Reads results/gate.pt and results/sweep_{mode}_L{L}.pt; writes results/yield.json and
results/keep.json (the kept items for the reader pass) and prints the tables used in RESULTS.md.

Keep criteria per (item, layer, mode, alpha):
  (i)   flip: some partner-answer alias is greedy-consistent AND no own-answer alias is;
  (ii)  not broken: (a) mean neutral-prompt KL (same edit at the last token of 6 unrelated
        prompts) <= KL_MAX; (b) prompt still decodable at L42: the planted L42 read vector is
        nearer (cosine) to its own unplanted vector than to every unplanted vector of the
        partner bridge (strict variant also reported: own is the nearest of ALL items);
  (iii) X-vs-Y probe at L42 reads Y: class-mean midpoint rule with leave-own-clue-out means.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from plant_lib import centred, class_mean, relation_means  # noqa: E402

RES = HERE / "results"
LAYERS = (24, 28, 34, 38, 42)
MODES = ("add", "clamp")
KL_MAX = 0.1
ALPHA_CAP = 4.0  # per-item minimal-alpha policy never goes above this


def load():
    g = torch.load(RES / "gate.pt", weights_only=False)
    sw = {}
    for m in MODES:
        for L in LAYERS:
            p = RES / f"sweep_{m}_L{L}.pt"
            if p.exists():
                sw[(m, L)] = torch.load(p, weights_only=False)
    return g, sw


def l42_checks(g, s):
    """probe_y [n, A], near_own_vs_Y [n, A], self_nn [n, A] for one sweep."""
    items, gate = g["items"], g["gate"]
    A42 = torch.as_tensor(g["acts"][42]).float()
    rm = relation_means(A42, items, gate)
    cen = centred(A42, items, rm)
    idx, partners = s["idx"], s["partners"]
    H = s["h42"].float()  # [n, A, d]
    n, nA, _ = H.shape
    gidx = np.where(gate)[0]
    Gn = torch.nn.functional.normalize(A42[gidx], dim=1)
    pos_in_g = {int(i): k for k, i in enumerate(gidx)}
    country = np.array([it["country"] for it in items])
    # bridge subspace at L42: span of all country class means (centred, gated constructed items)
    cs = sorted({it["country"] for it in items})
    M = torch.stack([class_mean(cen, items, gate, c, None) for c in cs])  # [C, d]
    Qb, _ = torch.linalg.qr(M.T)  # [d, C]
    def resid(X):
        return X - (X @ Qb) @ Qb.T
    Gr = torch.nn.functional.normalize(resid(A42[gidx] - torch.stack([rm[items[i]["relation"]] for i in gidx])), dim=1)
    relnames = sorted(rm)
    R = torch.nn.functional.normalize(torch.stack([rm[r] for r in relnames]), dim=1)
    probe = torch.zeros(n, nA, dtype=torch.bool)
    selfr = torch.zeros(n, nA, dtype=torch.bool)
    relok = torch.zeros(n, nA, dtype=torch.bool)
    near = torch.zeros(n, nA, dtype=torch.bool)
    selfnn = torch.zeros(n, nA, dtype=torch.bool)
    score = torch.zeros(n, nA)
    for k, i in enumerate(idx):
        it = items[i]
        mx = class_mean(cen, items, gate, it["country"], it["clue"] if it["source"] == "cons" else None)
        my = class_mean(cen, items, gate, partners[k], None)
        hc = H[k] - rm[it["relation"]]
        sc = (hc - (mx + my) / 2) @ (my - mx)
        score[k] = sc / (my - mx).norm()
        probe[k] = sc > 0
        cos = torch.nn.functional.normalize(H[k], dim=1) @ Gn.T  # [A, G]
        own = cos[:, pos_in_g[int(i)]]
        ymask = torch.tensor(country[gidx] == partners[k])
        near[k] = own > cos[:, ymask].max(1).values
        other = cos.clone()
        other[:, pos_in_g[int(i)]] = -2
        selfnn[k] = own > other.max(1).values
        # prompt decodability with the bridge subspace removed (primary (ii-b))
        cr = torch.nn.functional.normalize(resid(H[k] - rm[it["relation"]]), dim=1) @ Gr.T
        ownr = cr[:, pos_in_g[int(i)]].clone()
        cr[:, pos_in_g[int(i)]] = -2
        selfr[k] = ownr > cr.max(1).values
        # relation still decodable (nearest relation mean by cosine)
        relok[k] = (torch.nn.functional.normalize(H[k], dim=1) @ R.T).argmax(1) == relnames.index(it["relation"])
    return probe, near, selfnn, score, selfr, relok


def analyse():
    g, sw = load()
    items = g["items"]
    out = {"n_items": len(items), "gate": {s: [int(g["gate"][[k for k, it in enumerate(items) if it["source"] == s]].sum()),
                                               sum(it["source"] == s for it in items)] for s in ("cons", "bank")},
           "cells": {}}
    keep_best = None
    for (m, L), s in sorted(sw.items(), key=lambda t: (t[0][0], t[0][1])):
        alphas = list(s["alphas"])
        flip = (s["greedy_y"] & ~s["greedy_x"])
        klm = s["kl_neutral"].mean(-1)
        klok = klm <= KL_MAX
        probe, near, selfnn, score, selfr, relok = l42_checks(g, s)
        keep = flip & klok & selfr & relok & probe  # primary
        keep_rawcos = flip & klok & near & probe  # raw-cosine variant (bridge-confounded)
        keep_strict = keep_rawcos & selfnn
        src = np.array([items[i]["source"] for i in s["idx"]])
        per_alpha = []
        for a, al in enumerate(alphas):
            per_alpha.append({
                "alpha": al, "flip": int(flip[:, a].sum()), "kl_ok": int(klok[:, a].sum()),
                "near_own": int(near[:, a].sum()), "self_nn": int(selfnn[:, a].sum()),
                "probe_y": int(probe[:, a].sum()), "keep": int(keep[:, a].sum()),
                "keep_strict": int(keep_strict[:, a].sum()),
                "self_nn_resid": int(selfr[:, a].sum()), "rel_ok": int(relok[:, a].sum()),
                "keep_rawcos": int(keep_rawcos[:, a].sum()),
                "keep_bank": int(keep[src == "bank", a].sum()),
                "flip_and_probe": int((flip & probe)[:, a].sum()),
                "delta_ratio_median": float(s["delta_ratio"][:, a].median()),
                "kl_median": float(klm[:, a].median()),
                "kl_p90": float(klm[:, a].quantile(0.9)),
            })
        # per-item minimal alpha <= cap satisfying keep
        cap = [a for a, al in enumerate(alphas) if 0 < al <= ALPHA_CAP]
        kmin = []
        for k in range(len(s["idx"])):
            hit = [a for a in cap if keep[k, a]]
            kmin.append(hit[0] if hit else -1)
        kmin = np.array(kmin)
        got = kmin >= 0
        dr = np.array([float(s["delta_ratio"][k, kmin[k]]) for k in np.where(got)[0]])
        al_used = np.array([alphas[kmin[k]] for k in np.where(got)[0]])
        nat = np.array([float(s["dnorm"][k] * alphas[kmin[k]] / s["nat_sd"][k]) for k in np.where(got)[0]])
        cell = {
            "layer": L, "mode": m, "n_candidates": len(s["idx"]),
            "n_bank_candidates": int((src == "bank").sum()), "per_alpha": per_alpha,
            "minalpha_policy": {
                "cap": ALPHA_CAP, "n_kept": int(got.sum()), "n_kept_bank": int((got & (src == "bank")).sum()),
                "alpha_hist": {str(al): int((al_used == al).sum()) for al in sorted(set(al_used.tolist()))},
                "delta_ratio_quantiles": [float(x) for x in np.quantile(dr, [0.1, 0.5, 0.9])] if len(dr) else None,
                "shift_in_natural_sd_median": float(np.median(nat)) if len(nat) else None,
            },
        }
        out["cells"][f"{m}_L{L}"] = cell
        if m == "clamp" and L < 42 and (keep_best is None or got.sum() > keep_best[0]):
            keep_best = (int(got.sum()), m, L, s, kmin, got)
    # keep list for the reader pass: best planting layer < 42 under the primary (centred) mode
    _, m, L, s, kmin, got = keep_best
    keep_list = [{"item": int(s["idx"][k]), "k": int(k), "id": items[s["idx"][k]]["id"],
                  "partner": s["partners"][k], "alpha": float(s["alphas"][kmin[k]]), "ai": int(kmin[k])}
                 for k in np.where(got)[0]]
    (RES / "keep.json").write_text(json.dumps({"mode": m, "layer": L, "items": keep_list}, indent=1))
    (RES / "yield.json").write_text(json.dumps(out, indent=1))
    return out, (m, L, len(keep_list))


def main() -> None:
    out, best = analyse()
    print("gate", out["gate"])
    for name, c in out["cells"].items():
        mp = c["minalpha_policy"]
        print(f"\n== {name}: candidates {c['n_candidates']} (bank {c['n_bank_candidates']}); "
              f"min-alpha<= {mp['cap']} kept {mp['n_kept']} (bank {mp['n_kept_bank']}); "
              f"alpha hist {mp['alpha_hist']}; |D|/|h| q10/50/90 {mp['delta_ratio_quantiles']}; "
              f"shift/natural-sd {mp['shift_in_natural_sd_median']}")
        print(" alpha  |D|/|h|  flip  klok  near  selfNN selfNNres relok probeY  KEEP rawcos strict bank  KLmed")
        for r in c["per_alpha"]:
            print(f" {r['alpha']:4.1f}  {r['delta_ratio_median']:.3f}  {r['flip']:5d} {r['kl_ok']:5d} "
                  f"{r['near_own']:5d} {r['self_nn']:5d} {r['self_nn_resid']:7d} {r['rel_ok']:5d} {r['probe_y']:5d} {r['keep']:5d} "
                  f"{r['keep_rawcos']:5d} {r['keep_strict']:5d} {r['keep_bank']:4d}  {r['kl_median']:.3f}")
    print("\nreader-pass keep list:", best)


if __name__ == "__main__":
    main()
