"""Is the null-band SD item-limited (shrinks with n) or direction-limited (floor)? PILOT."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import numpy as np
import analyze as A
from pilot_lib import conditions, K_NULL
c42 = {c: i for i, c in enumerate(conditions(42))}
dec = {"NLA": A.load_decoder("nla")[42], "OL": A.load_decoder("ol")[42]}
for r, (lo, temps) in dec.items():
    S = A.fam_scores(lo, temps)
    for fam in ("count_qual", "count_num", "decision"):
        for d, cls in (("jrem1", "nonj1"), ("jrem4", "nonj4"), ("jrem1", "iso")):
            for q in ("q90",):
                X = S[fam]
                sh = np.array([X[c42[("null", f"{cls}_{k}", "centred", q, 1, d)]] - X[c42[("null", f"{cls}_{k}", "centred", q, -1, d)]] for k in range(K_NULL)])  # [K, N]
                real = X[c42[("real", d, "centred", q, 1, d)]] - X[c42[("real", d, "centred", q, -1, d)]]
                K, N = sh.shape
                between = sh.mean(1).std(ddof=1)
                within_se = np.sqrt((sh.var(1, ddof=1) / N).mean())
                true_dir_sd = np.sqrt(max(between**2 - within_se**2, 0))
                z60 = (real.mean() - sh.mean(1).mean()) / between
                zinf = (real.mean() - sh.mean(1).mean()) / true_dir_sd if true_dir_sd > 0 else np.inf
                print(f"{r} {fam} {d} vs {cls} {q}: real shift {real.mean():+.3f}; null-mean SD {between:.3f} = sqrt(dir {true_dir_sd:.3f}^2 + item-SE {within_se:.3f}^2); z@60 {z60:+.2f}; z@n=inf {zinf:+.2f}")
