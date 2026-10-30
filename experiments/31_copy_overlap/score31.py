"""Exp 31: score the four pre-registered predictions from results/overlap31.json and results/carry31.json
-> results/score31.json (numbers quoted in RESULTS.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
NAT7 = ["py_string", "js_string", "latex_math", "py_arg", "json_depth", "md_heading", "md_list"]
CARRY4 = ["py_string", "js_string", "latex_math", "json_depth"]


def main() -> None:
    ov = json.loads((R / "overlap31.json").read_text())
    ca = json.loads((R / "carry31.json").read_text())["carry"]
    out = {}
    for bank in ("e25", "e20"):
        b = {}
        ex = {r: [ov[f"{bank} | {row}"][r]["tok"]["excess"] for row in NAT7] for r in ("ol", "nla")}
        ci_pos = sum(ov[f"{bank} | {row}"]["ol"]["tok"]["excess_ci"][0] > 0 for row in NAT7)
        b["p1"] = {"ol_mean_excess": float(np.mean(ex["ol"])), "nla_mean_excess": float(np.mean(ex["nla"])),
                   "ratio": float(np.mean(ex["ol"]) / np.mean(ex["nla"])), "ol_rows_ci_above_0": int(ci_pos),
                   "ol_own_range": [min(ov[f"{bank} | {r}"]["ol"]["tok"]["own"] for r in NAT7), max(ov[f"{bank} | {r}"]["ol"]["tok"]["own"] for r in NAT7)],
                   "nla_own_range": [min(ov[f"{bank} | {r}"]["nla"]["tok"]["own"] for r in NAT7), max(ov[f"{bank} | {r}"]["nla"]["tok"]["own"] for r in NAT7)]}
        for r in ("ol", "nla"):
            full = [ca[f"{bank} | {row}"][r]["full"]["auroc"] for row in CARRY4]
            bal = [ca[f"{bank} | {row}"][r]["del_bal"]["auroc"] for row in CARRY4]
            low = [ca[f"{bank} | {row}"][r]["low_half"]["auroc"] for row in CARRY4]
            rh = [ca[f"{bank} | {row}"][r]["low_half"]["rand_half_mean"] for row in CARRY4]
            b[r] = {"full": full, "del_bal": bal, "mean_drop": float(np.mean(np.subtract(full, bal))),
                    "low_half_mean": float(np.mean(low)), "rand_half_mean": float(np.mean(rh)),
                    "low_minus_rand": float(np.mean(low) - np.mean(rh))}
        b["p2_holds"] = all(v >= 0.70 for v in b["nla"]["del_bal"]) and b["nla"]["mean_drop"] <= 0.10
        b["p3_clause_a"] = sum(v >= 0.65 for v in b["ol"]["del_bal"])
        b["p3_clause_b"] = b["ol"]["mean_drop"] - b["nla"]["mean_drop"]
        out[bank] = b
    rp = ca["e27 | code (py+js) / in_string"]
    out["e27_code_in_string_del_bal"] = {r: rp[r]["del_bal"]["auroc"] for r in ("ol", "nla")}
    (R / "score31.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
