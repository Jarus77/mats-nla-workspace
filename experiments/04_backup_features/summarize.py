"""Local: pull gate/causal JSONs from the rfm-results volume, apply the D9/D11 gate rules, write
results/gates.json and print the markdown gate table used in RESULTS.md.

Rules (L42 primary; J-Lens = neuronpedia n1000):
  G1          held-out grouped-CV probe AUROC >= 0.85
  G2-geom     diff-in-means enrichment <= 1.5 at r*=384 AND 4r*=1536
  G2-func     NOT readable: label-word test (02 s3d replica) inside/below band AND emitted-form
              tests (soft lens mass, top-10 count) inside the two-sided permutation band
  label       NON-J (geom+func) | DIFFUSE-BUT-J-READABLE (geom, not func) | IN-J (neither) |
              J-ENRICHED-NOT-READABLE (func, not geom)
  G4          steer L38 centred along the J-removed (1536) diff-in-means direction: at p90 or p99,
              |effect| >= 5x max |effect| of K=24 J-complement random directions AND |z| >= 3 AND
              item-paired Wilcoxon p < 0.01 vs the strongest random direction
  leak        J-Lens AUROC / mean shift at 40/42/44 outside the two-sided J-complement band at the
              G4-passing dose (any layer) -> dose not leak-free
    python summarize.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FEATS = ["cap_next", "cap_next_all", "code_lang", "in_quote", "gender_next", "indent", "sent_pos", "list_num", "list_hdr"]


def pull() -> None:
    RES.mkdir(exist_ok=True)
    for sub in ("gate", "causal"):
        subprocess.run(["modal", "volume", "get", "rfm-results", f"04_backup_features/{sub}", str(RES), "--force"],
                       check=False, capture_output=True)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def find(sub: str, name: str) -> dict | None:
    for p in (RES / sub / name, RES / name, RES / "04_backup_features" / sub / name):
        if p.exists():
            return load(p)
    return None


def readable(fx: dict) -> bool:
    """PRIMARY functional rule: J-Lens reads the feature if the 02 label-word test is above its band OR
    the emitted-form top-10 category count (Fable round 4: "classify the top-10 tokens") is above
    its band. One-sided, as in 02 s3d (below-band = anti-tracking, flagged separately)."""
    return bool(fx["functional"]["j_readable"] or fx["emitted_top10"]["readable_positive"])


def readable_strict(fx: dict) -> bool:
    """Supplement: any of label / top-10 / soft-mass outside its two-sided band (02's col fails this)."""
    return bool(fx["functional"]["j_readable"] or fx["functional"]["below_band"]
                or fx["emitted_soft"]["readable_two_sided"] or fx["emitted_top10"]["readable_two_sided"])


def g4_eval(c: dict | None) -> dict:
    if c is None:
        return {"pass": None}
    g = c["g4"]
    best = None
    for dn in ("p90", "p99"):
        e = g[f"jrem1536_{dn}"]
        ok = (e["ratio_vs_jc_absmax"] >= 5 and abs(e["band_jc"]["z"]) >= 3 and e["wilcoxon_p_vs_strongest_jc"] < 0.01)
        leak_layers = [L for L, lk in c.get("leak", {}).items() if lk[f"jrem1536_{dn}"]["leak"]]
        cand = {"dose": dn, "effect": e["effect"], "ratio": e["ratio_vs_jc_absmax"], "z_jc": e["band_jc"]["z"],
                "z_iso": e["band_iso"]["z"], "p": e["wilcoxon_p_vs_strongest_jc"], "pass": ok,
                "leak_layers": leak_layers, "sign_as_predicted": e["effect"] > 0,
                "raw_effect": g[f"raw_{dn}"]["effect"],
                "retained_frac": e["effect"] / g[f"raw_{dn}"]["effect"] if g[f"raw_{dn}"]["effect"] else None}
        if best is None or (cand["pass"] and not cand["leak_layers"] and not (best["pass"] and not best["leak_layers"])):
            best = cand
    return best


def summarize() -> dict:
    table = {}
    for f in FEATS:
        gt = find("gate", f"{f}.json")
        if gt is None:
            continue
        L = gt["layers"]["42"]
        geo = L["geometric"]
        g1 = L["G1"]["auroc"] >= 0.85
        geom = geo["384"]["diffmeans"] <= 1.5 and geo["1536"]["diffmeans"] <= 1.5
        func = not readable(L)
        label = ("NON-J" if geom and func else "DIFFUSE-BUT-J-READABLE" if geom else
                 "J-ENRICHED-NOT-READABLE" if func else "IN-J")
        label_strict = ("NON-J" if geom and not readable_strict(L) else label if not geom or not func
                        else "DIFFUSE-BUT-J-READABLE")
        cb = L.get("camilablank")
        cb_label = None
        if cb:
            cgeom = cb["geometric"]["384"]["diffmeans"] <= 1.5 and cb["geometric"]["1536"]["diffmeans"] <= 1.5
            cfunc = not readable(cb)
            cb_label = ("NON-J" if cgeom and cfunc else "DIFFUSE-BUT-J-READABLE" if cgeom else
                        "J-ENRICHED-NOT-READABLE" if cfunc else "IN-J")
        c38, c42 = find("causal", f"{f}_L38.json"), find("causal", f"{f}_L42.json")
        g4 = g4_eval(c38)
        g4_42 = g4_eval(c42)
        clean = bool(g1 and label == "NON-J" and g4.get("pass") and not g4.get("leak_layers"))
        table[f] = {"n_items": gt["n_items"], "n_rows": gt["n_rows"], "G1": L["G1"], "G1_pass": g1,
                    "G1_by_layer": {k: v["G1"]["auroc"] for k, v in gt["layers"].items()},
                    "enrich": {r: geo[r]["diffmeans"] for r in ("384", "1536")},
                    "enrich_shuffled": {r: geo[r]["shuffled_label_diffmeans_mean"] for r in ("384", "1536")},
                    "enrich_acts": {r: geo[r]["acts"] for r in ("384", "1536")},
                    "enrich_by_layer": {k: {r: v["diffmeans"] for r, v in lv["geometric"].items()} for k, lv in gt["layers"].items()},
                    "functional": L["functional"], "emitted_soft": L["emitted_soft"], "emitted_top10": L["emitted_top10"],
                    "func_by_layer": {k: readable(v) for k, v in gt["layers"].items() if "functional" in v},
                    "lens_bag": L["lens_bag_top25"], "local4": gt["baseline_local4_bag"],
                    "abs_pos_auroc": gt["baseline_abs_position_auroc"], "jonly": L.get("jonly"),
                    "label": label, "label_strict": label_strict, "camilablank_label": cb_label, "camilablank": cb,
                    "G4_L38": g4, "G4_L42": g4_42,
                    "kl_L38": c38["kl_mean"] if c38 else None, "kl_jc_L38": c38["kl_mean_jc"] if c38 else None,
                    "dose_ratio_L38": c38["dose_over_median_norm"] if c38 else None,
                    "selfreport": c38.get("selfreport") if c38 else None, "clean_nonj": clean}
    return table


def fmt(t: dict) -> str:
    rows = ["| feature | items/rows | G1 AUROC @L42 [CI] | enrich r*/4r* (shuf) | J-Lens label AUROC [band] | "
            "emitted soft / top10 AUROC [band] | label (n1000) | label (cb) | G4 @L38 jrem1536 (dose): effect, x max-rand, z_jc | "
            "leak 40/42/44 | self-report AUROC | QUALIFIES |", "|" + "---|" * 12]
    for f, r in t.items():
        fu, es, et = r["functional"], r["emitted_soft"], r["emitted_top10"]
        g = r["G4_L38"] or {}
        g4s = (f"{g['dose']}: {g['effect']:+.3f}, {g['ratio']:.1f}x, z {g['z_jc']:.1f}" if g.get("dose") else "–")
        sr = r["selfreport"]
        rows.append(
            f"| {f} | {r['n_items']}/{r['n_rows']} | {r['G1']['auroc']:.3f} [{r['G1']['ci95'][0]:.3f}, {r['G1']['ci95'][1]:.3f}] | "
            f"{r['enrich']['384']:.2f}/{r['enrich']['1536']:.2f} ({r['enrich_shuffled']['384']:.2f}/{r['enrich_shuffled']['1536']:.2f}) | "
            f"{fu['stat']:.3f} [{fu['null_band95'][0]:.3f}, {fu['null_band95'][1]:.3f}] | "
            f"{es['auroc']:.3f} [{es['null_band95'][0]:.3f}, {es['null_band95'][1]:.3f}] / {et['auroc']:.3f} | "
            f"{r['label']} | {r['camilablank_label']} | {g4s} {'PASS' if g.get('pass') else 'fail'} | "
            f"{','.join(g.get('leak_layers') or []) or 'none'} | "
            f"{(format(sr['auroc'], '.2f') if sr else '–')} | {'**YES**' if r['clean_nonj'] else 'no'} |")
    return "\n".join(rows)


if __name__ == "__main__":
    pull()
    t = summarize()
    (RES / "gates.json").write_text(json.dumps(t, indent=1, default=float))
    print(fmt(t))
