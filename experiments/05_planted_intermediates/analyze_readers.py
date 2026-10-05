"""Reader pass on kept planted items: does each reader report the planted bridge Y (reading) or
the text-implied bridge X (re-solving)? Local, CPU. Writes results/readers.json.

LO = log p(Y-candidate) - log p(X-candidate) in each reader's native format (template mean;
J-Lens: max-variant cosine-score difference). Conditions per item: natural (unplanted),
planted, and 2K equal-norm random-direction plants at the same layer (K isotropic, K
J-complement), propagated through the model to L42. Null bands (D9/D11): each null direction
gives one value of each statistic across items; the real statistic is placed in that band as
percentile and z (never subtracted).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from mats.metrics import null_band  # noqa: E402

RES = HERE / "results"


def stats(lo_nat: np.ndarray, lo_x: np.ndarray) -> dict[str, float]:
    return {"p_reads_Y": float(np.mean(lo_x > 0)), "mean_shift": float(np.mean(lo_x - lo_nat)),
            "paired_win": float(np.mean((lo_x > lo_nat) + 0.5 * (lo_x == lo_nat)))}


def reader_block(lo: np.ndarray, kinds: list[str]) -> dict:
    """lo [n, C] (C = natural, planted, nulls...)."""
    nat, pl, nul = lo[:, 0], lo[:, 1], lo[:, 2:]
    real = stats(nat, pl)
    out = {"n": int(lo.shape[0]), "p_reads_Y_natural": float(np.mean(nat > 0)),
           "p_reads_Y_planted": real["p_reads_Y"], "p_text_implied_planted": float(np.mean(pl < 0)),
           "mean_LO_natural": float(nat.mean()), "mean_LO_planted": float(pl.mean()),
           "median_shift": float(np.median(pl - nat)), "real": real, "bands": {}}
    for kind in sorted(set(kinds)):
        cols = [j for j, k in enumerate(kinds) if k == kind]
        ns = [stats(nat, nul[:, j]) for j in cols]
        out["bands"][kind] = {s: null_band(real[s], [x[s] for x in ns]) for s in real}
    return out


def main() -> None:
    b = torch.load(RES / "readers_base.pt", weights_only=False)
    kinds = b["null_kinds"]
    rep = {"mode": b["mode"], "layer": b["layer"], "n": len(b["keep"]),
           "enrichment_u_median": float(np.median(b["enrichment_u"])), "r_star": b["r_star"],
           "readers": {}}
    rep["readers"]["OL@42"] = reader_block(b["lo_ol"].mean(0).numpy(), kinds)
    rep["readers"]["OL@42_per_template"] = {
        t: reader_block(b["lo_ol"][i].numpy(), kinds)["p_reads_Y_planted"]
        for i, t in enumerate(b["ol_templates"])}
    rep["readers"]["J-Lens@42"] = reader_block(b["lo_jl"].numpy(), kinds)
    pn = RES / "readers_nla.pt"
    if pn.exists():
        nb = torch.load(pn, weights_only=False)
        rep["readers"]["NLA@42"] = reader_block(nb["lo_nla"].mean(0).numpy(), kinds)
        rep["readers"]["NLA@42_per_template"] = {
            t: reader_block(nb["lo_nla"][i].numpy(), kinds)["p_reads_Y_planted"]
            for i, t in enumerate(nb["nla_templates"])}
        rep["nla_examples"] = [{"names": b["names"][i], "natural": nb["ex_nat"][i], "planted": nb["ex_pl"][i]}
                               for i in range(len(nb["ex_nat"]))]
    (RES / "readers.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"mode {rep['mode']} L{rep['layer']} n={rep['n']} enrichment(u)@r*={rep['enrichment_u_median']:.2f}")
    print("reader      P(Y)nat  P(Y)plant P(X)plant  shift   win | z_shift iso / jc | z_pY iso / jc | null p(Y) band iso")
    for name, r in rep["readers"].items():
        if "per_template" in name:
            continue
        bi, bj = r["bands"]["iso"], r["bands"]["jc"]
        print(f"{name:10s}  {r['p_reads_Y_natural']:.2f}     {r['p_reads_Y_planted']:.2f}      "
              f"{r['p_text_implied_planted']:.2f}   {r['real']['mean_shift']:6.2f}  {r['real']['paired_win']:.2f} | "
              f"{bi['mean_shift']['z']:6.1f} / {bj['mean_shift']['z']:6.1f} | "
              f"{bi['p_reads_Y']['z']:6.1f} / {bj['p_reads_Y']['z']:6.1f} | "
              f"[{bi['p_reads_Y']['band_lo']:.2f}, {bi['p_reads_Y']['band_hi']:.2f}]")


if __name__ == "__main__":
    main()
