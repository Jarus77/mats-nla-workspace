"""TOST power simulation for the pre-registered negatives N1-N5 (prereg rev 4.1, D27.4).

CPU only; seeded; reproducible.   ../../../.venv/bin/python tost_sim.py [--nsim 1000]
  -> pilot_inputs.npz (cached pilot arrays), tost_sim.json, stdout table (tost_sim_stdout.txt).

Statistic = the one the null bands are built from (prereg sec. 3.1): the paired per-item effect
e_i = (LO_i(+D) - LO_i(-D))/2 and its mean e-bar.  Equivalence test on
    T = e-bar(real) - mean_k e-bar(null_k),      k = 1..K (K = 100), SAME fresh items,
    SE^2 = Var_i(e_real,i - mean_k e_k,i)/n + Var_k(e-bar_k)/K,
    pass iff T - 1.645 SE > -delta and T + 1.645 SE < +delta   (TOST, alpha = .05 each side),
against BOTH bands (isotropic, J-complement) -> per-family pass = both bands (IUT).
delta (SESOI) = the e-bar shift that turns cross-item AUROC(LO(+D), LO(-D)) from 0.50 into 0.55
(fraction of probe 0.10) under a pure location shift of the pilot's cross-item LO distribution.

Inputs (PILOT split, excluded from the main analysis): round-5 OL@42 raw scores at the P1/P2 cell
(col reg_jrem4, centred, q90 = 7.098), N1-N5 template families; null blocks iso/nonj4 K=20 at the
nearest available norm (q90 @ reg_raw = 9.465), rescaled by 7.098/9.465 (linear dose response;
OL col is linear in dose, RESULTS.md R5) -- '--noscale' keeps the larger-norm nulls (conservative).

Generative model (per family f, band c; resampling the pilot's per-item structure):
  null_k,i,f = mu_c,f + a_c,k,f + R_c[j, item_i, f]      a_c,k ~ MVN(0, Sigma_dir,c)  (direction level,
                                                          fixed in n), R = direction-centred pilot
                                                          residual (keeps the shared item component),
                                                          column j drawn per (k, i)
  real_i,f   = mu_Jc,f + theta_f + [a_r,f] + rho[item_i, f]   rho = centred pilot real-cell residual
  item_i ~ bootstrap from the 60 pilot passages (joint across families and bands).
Scenarios: A0 theta = 0;  Ahalf theta = delta/2;  Asize theta = delta (size check, should be <= .05);
           B0 theta = 0 plus a_r ~ MVN(0, Sigma_dir,Jc): the real direction behaves exactly like a
           random J-complement direction (worst case for 'the band does not shrink with n');
           Apilot theta = the pilot's observed deviation of the real cell from the (rescaled) Jc band mean.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PILOT = ROOT / "experiments" / "08_col_pilot"
FAMS_BY_READER = {"ol": ("count_qual", "count_num", "decision", "f07_count", "f07_break"),  # N1..N5
                  "nla": ("count_qual", "count_num", "decision", "f07_break")}             # S9 (declared NLA negatives)
BANDS = ("iso", "nonj4")
DOSE_REAL, DOSE_NULL = 7.098, 9.465
NS = (120, 240, 480, 960)
K = 100
Z95 = 1.6448536269514722
SEED = 20261006
AUC_SESOI = 0.55


# ----------------------------------------------------------------------------- pilot inputs
def extract(reader: str, fams: tuple) -> dict:
    """Pull per-item e_i for the real cell and both K=20 null blocks from round-5 raw OL scores."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(PILOT))
    import r5_analyze as A  # noqa: PLC0415
    from r5_lib import K_NULL, conditions  # noqa: PLC0415

    X, temps = A.load(reader)["col"]
    ci = {c: i for i, c in enumerate(conditions("col"))}
    S = A.fam_scores(X, temps)
    real = np.stack([(S[f][ci[("real", "reg_jrem4", "q90", 1, "reg_jrem4")]]
                      - S[f][ci[("real", "reg_jrem4", "q90", -1, "reg_jrem4")]]) / 2 for f in fams], -1)
    lo_p = np.stack([S[f][ci[("real", "reg_jrem4", "q90", 1, "reg_jrem4")]] for f in fams], -1)
    lo_m = np.stack([S[f][ci[("real", "reg_jrem4", "q90", -1, "reg_jrem4")]] for f in fams], -1)
    lo0 = np.stack([S[f][ci[("zero", None, None, 0, None)]] for f in fams], -1)
    out = {"real": real, "lo_p": lo_p, "lo_m": lo_m, "lo0": lo0,
           "files": np.array(sorted(Path(p).name for p in glob.glob(str(PILOT / f"results/08_col_pilot/r5_{reader}_chunk*.pt"))))}
    for c in BANDS:
        out[f"null_{c}"] = np.stack([np.stack([(S[f][ci[("null", f"{c}_{k}", "q90", 1, "reg_raw")]]
                                                - S[f][ci[("null", f"{c}_{k}", "q90", -1, "reg_raw")]]) / 2
                                               for f in fams], -1) for k in range(K_NULL)])  # [K, N, F]
    return out


def load_inputs(reader: str, fams: tuple) -> dict:
    cache = HERE / ("pilot_inputs.npz" if reader == "ol" else f"pilot_inputs_{reader}.npz")
    if not cache.exists():
        np.savez(cache, **extract(reader, fams))
    return dict(np.load(cache))


# ----------------------------------------------------------------------------- SESOI mapping
def auc_shift(lo: np.ndarray, d: float) -> float:
    """Cross-item AUROC of (lo + d) vs (lo - d) over distinct pairs i != j (pure location shift)."""
    diff = (lo[:, None] + d) - (lo[None, :] - d)
    off = ~np.eye(len(lo), dtype=bool)
    return float(((diff > 0) + 0.5 * (diff == 0))[off].mean())


def sesoi(lo: np.ndarray) -> dict:
    """delta in e-bar units: empirical (bisection on the pilot LO distribution) and normal-theory."""
    a, b = 0.0, 5.0 * lo.std()
    for _ in range(60):
        mid = (a + b) / 2
        a, b = (mid, b) if auc_shift(lo, mid) < AUC_SESOI else (a, mid)
    from scipy.stats import norm  # noqa: PLC0415
    return {"delta_emp": (a + b) / 2, "delta_norm": float(norm.ppf(AUC_SESOI) * lo.std(ddof=1) / np.sqrt(2)),
            "sd_lo": float(lo.std(ddof=1))}


# ----------------------------------------------------------------------------- band model
def band_model(E: np.ndarray, n_pilot: int) -> dict:
    """Decompose a pilot null block E [K, N, F] into mean, direction covariance, item residuals."""
    eb = E.mean(1)                                     # [K, F]
    R = E - eb[:, None, :]                             # direction-centred residuals (item comp. + noise)
    W = np.einsum("kif,kig->fg", R, R) / (E.shape[0] * (n_pilot - 1))   # within-direction item cov
    Sig = np.cov(eb, rowvar=False) - W / n_pilot       # de-noised direction-level covariance
    w, V = np.linalg.eigh(Sig)
    Sig = (V * np.clip(w, 0, None)) @ V.T              # PSD projection
    return {"mu": eb.mean(0), "Sig": Sig, "R": R, "sd_band_pilot": eb.std(0, ddof=1),
            "sd_dir": np.sqrt(np.diag(Sig))}


# ----------------------------------------------------------------------------- simulation
def one_experiment(rng, n, models, rho, mu_ref, delta, thetas, ar):  # noqa: ANN001, PLR0913
    """Simulate one fresh experiment at n items; return pass flags [scenario, band, F]."""
    idx = rng.integers(0, rho.shape[0], n)
    F = rho.shape[1]
    base_real = mu_ref + rho[idx]                                          # [n, F]
    T0, SE = [], []
    for c in BANDS:
        m = models[c]
        a = rng.multivariate_normal(np.zeros(F), m["Sig"], size=K, method="eigh")   # [K, F]
        cols = rng.integers(0, m["R"].shape[0], (K, n))
        nul = m["mu"] + a[:, None, :] + m["R"][cols, idx[None, :], :]     # [K, n, F]
        band_i = nul.mean(0)                                               # [n, F]
        ebk = nul.mean(1)                                                  # [K, F]
        d = base_real - band_i
        T0.append(d.mean(0))
        SE.append(np.sqrt(d.var(0, ddof=1) / n + ebk.var(0, ddof=1) / K))
    T0, SE = np.array(T0), np.array(SE)                                    # [B, F]
    out, ns = [], []
    for name, th in thetas.items():
        T = T0 + th + (ar if name.startswith("B") else 0.0)
        out.append((T - Z95 * SE > -delta) & (T + Z95 * SE < delta))
        ns.append(T + Z95 * SE < delta)
    return np.array(out), np.array(ns)                                     # [S, B, F] x2


def simulate(inp: dict, scale: float, nsim: int, seed: int, fams: tuple, dirinflate: float = 1.0) -> dict:  # noqa: PLR0913
    n_pilot = inp["real"].shape[0]
    models = {c: band_model(inp[f"null_{c}"] * scale, n_pilot) for c in BANDS}
    for m in models.values():                       # sensitivity: direction-level SD x dirinflate
        m["Sig"] = m["Sig"] * dirinflate**2
        m["sd_dir"] = m["sd_dir"] * dirinflate
    lo = np.concatenate([inp["lo_p"] - inp["lo_p"].mean(0), inp["lo_m"] - inp["lo_m"].mean(0)])
    ses = [sesoi(lo[:, f]) for f in range(len(fams))]
    delta = np.array([s["delta_emp"] for s in ses])
    rho = inp["real"] - inp["real"].mean(0)
    mu_ref = models["nonj4"]["mu"]
    pilot_dev = inp["real"].mean(0) - mu_ref     # observed pilot deviation of the real cell from the Jc band
    thetas = {"A0": 0.0 * delta, "Ahalf": 0.5 * delta, "Apilot": pilot_dev, "Asize": 1.0 * delta, "B0": 0.0 * delta}
    rng = np.random.default_rng(seed)
    res = {}
    for n in NS:
        acc = np.zeros((len(thetas), len(BANDS), len(fams)))
        both = np.zeros((len(thetas), len(fams)))
        iut = np.zeros(len(thetas))
        iut_ns = np.zeros(len(thetas))
        for _ in range(nsim):
            ar = rng.multivariate_normal(np.zeros(len(fams)), models["nonj4"]["Sig"], method="eigh")
            p, pns = one_experiment(rng, n, models, rho, mu_ref, delta, thetas, ar)
            iut_ns += pns.all(axis=(1, 2))
            acc += p
            both += p.all(axis=1)
            iut += p.all(axis=(1, 2))
        acc /= nsim
        both /= nsim
        res[n] = {s: {"per_band": {c: dict(zip(fams, acc[i, b].round(4).tolist())) for b, c in enumerate(BANDS)},
                      "per_family_both_bands": dict(zip(fams, both[i].round(4).tolist())),
                      "iut_all5_both_bands": round(float(iut[i] / nsim), 4),
                      "iut_nonsuperiority": round(float(iut_ns[i] / nsim), 4)} for i, s in enumerate(thetas)}
    info = {"scale": scale, "delta": dict(zip(fams, delta.round(4).tolist())),
            "sesoi": {f: {k: round(v, 4) for k, v in s.items()} for f, s in zip(fams, ses)},
            "bands": {c: {"mu": dict(zip(fams, m["mu"].round(4).tolist())),
                          "sd_dir": dict(zip(fams, m["sd_dir"].round(4).tolist())),
                          "sd_band_pilot_n60": dict(zip(fams, m["sd_band_pilot"].round(4).tolist())),
                          "delta_over_sd_dir": dict(zip(fams, (delta / np.maximum(m["sd_dir"], 1e-9)).round(2).tolist()))}
                      for c, m in models.items()},
            "real_pilot": {"ebar": dict(zip(fams, inp["real"].mean(0).round(4).tolist())),
                           "item_sd": dict(zip(fams, inp["real"].std(0, ddof=1).round(4).tolist()))}}
    return {"info": info, "power": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=1000)
    ap.add_argument("--noscale", action="store_true")
    ap.add_argument("--reader", choices=("ol", "nla"), default="ol")
    ap.add_argument("--dirinflate", type=float, default=1.0, help="multiply direction-level SD (sensitivity)")
    args = ap.parse_args()
    fams = FAMS_BY_READER[args.reader]
    inp = load_inputs(args.reader, fams)
    scale = 1.0 if args.noscale else DOSE_REAL / DOSE_NULL
    out = simulate(inp, scale, args.nsim, SEED, fams, args.dirinflate)
    out |= {"nsim": args.nsim, "K": K, "seed": SEED, "dirinflate": args.dirinflate, "reader": args.reader, "pilot_files": inp["files"].tolist()}
    tag = ("" if args.reader == "ol" else f"_{args.reader}") + ("_noscale" if args.noscale else "") + (f"_dir{args.dirinflate:g}" if args.dirinflate != 1.0 else "")
    (HERE / f"tost_sim{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    I = out["info"]
    print(f"# TOST power sim ({args.reader.upper()}; scale {scale:.3f}; dirinflate {args.dirinflate:g}; nsim {args.nsim}; K {K}; seed {SEED})")
    for f in fams:
        print(f"{f:10s} delta {I['delta'][f]:.3f} (normal {I['sesoi'][f]['delta_norm']:.3f}; sd_LO {I['sesoi'][f]['sd_lo']:.2f})"
              f" | sd_dir iso {I['bands']['iso']['sd_dir'][f]:.3f} Jc {I['bands']['nonj4']['sd_dir'][f]:.3f}"
              f" | delta/sd_dir iso {I['bands']['iso']['delta_over_sd_dir'][f]:.1f} Jc {I['bands']['nonj4']['delta_over_sd_dir'][f]:.1f}"
              f" | pilot real ebar {I['real_pilot']['ebar'][f]:+.3f} vs Jc mu {I['bands']['nonj4']['mu'][f]:+.3f}")
    print("\nscenario  n   | per-family TOST power, both bands (iso/Jc) | IUT TOST all 5 x 2 bands | IUT non-superiority")
    for s in ("A0", "Ahalf", "Apilot", "Asize", "B0"):
        for n in NS:
            r = out["power"][n][s]
            cells = " ".join(f"{r['per_family_both_bands'][f]:.2f}({r['per_band']['iso'][f]:.2f}/{r['per_band']['nonj4'][f]:.2f})"
                             for f in fams)
            print(f"{s:6s} {n:4d} | {cells} | {r['iut_all5_both_bands']:.3f} | {r['iut_nonsuperiority']:.3f}")


if __name__ == "__main__":
    main()
