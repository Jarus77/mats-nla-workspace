"""Local analysis of the col pilot (PILOT -- EXCLUDED FROM MAIN ANALYSIS).

    ../../.venv/bin/python analyze.py      -> results/metrics.json + results/analyze_stdout.txt
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from mats import metrics as M  # noqa: E402
from pilot_lib import DOSES, JLENS_CONDS, K_NULL, NULL_SETS, REAL_DIRS, VARIANTS, ZERO, conditions  # noqa: E402
from templates import FAMILIES, MENTION_RE, MENTION_RE_STRICT  # noqa: E402

RES = HERE / "results"
DATA = RES / "08_col_pilot"  # raw outputs downloaded from the volume
NB = 1000
Z80 = 1.959964 + 0.841621  # two-sided alpha .05, power .80


# ----------------------------------------------------------------------------- loading
def load_decoder(name: str) -> dict:
    """{layer: (lo [T, C, N], templates)} assembled from chunk files."""
    out: dict = {}
    for f in sorted(glob.glob(str(DATA / f"{name}_chunk*.pt"))):
        d = torch.load(f, weights_only=False)
        for L, blk in d["lo"].items():
            C = len(conditions(L))
            T, _, N = blk["lo"].shape
            if L not in out:
                out[L] = [np.full((T, C, N), np.nan), d["templates"]]
            out[L][0][:, blk["idx"], :] = blk["lo"].numpy()
    return out


def fam_scores(lo: np.ndarray, templates: list) -> dict[str, np.ndarray]:
    """{family: [C, N] template mean} + {"t<i>": [C, N]} per template."""
    out = {}
    for fam in FAMILIES:
        ix = [i for i, t in enumerate(templates) if t[0] == fam]
        if ix:
            out[fam] = lo[ix].mean(0)
    for i in range(len(templates)):
        out[f"t{i}"] = lo[i]
    return out


# ----------------------------------------------------------------------------- core stats
def contrast_stats(S: np.ndarray, ci: dict, d: str, v: str, q: str, boot: bool = True) -> dict:
    p, m = S[ci[("real", d, v, q, 1, d)]], S[ci[("real", d, v, q, -1, d)]]
    if np.isnan(p).any() or np.isnan(m).any():
        return {"missing": True}
    if boot:
        r = M.follow_feature(p, m, n_boot=NB)
    else:
        r = M.paired_stats(p, m) | {"lo": float("nan"), "hi": float("nan")}
    z = S[ci[ZERO]]
    out = {"auroc": r["auroc"], "lo": r["lo"], "hi": r["hi"], "paired_win": r["paired_win"],
           "mean_shift": r["mean_shift"], "shift_sd": float(np.std(p - m, ddof=1)),
           "auroc_zero_sub": M.auroc(p - z, m - z), "fraction_of_probe": 2 * (r["auroc"] - 0.5), "bands": {}}
    for cls in NULL_SETS[d]:
        keys = [("null", f"{cls}_{k}", "centred", q, s, d) for k in range(K_NULL) for s in (1, -1)]
        if not all(k in ci for k in keys):
            continue
        nulls = [(S[ci[("null", f"{cls}_{k}", "centred", q, 1, d)]],
                  S[ci[("null", f"{cls}_{k}", "centred", q, -1, d)]]) for k in range(K_NULL)]
        if any(np.isnan(a).any() or np.isnan(b).any() for a, b in nulls):
            continue
        out["bands"][cls] = M.null_band_contrast((p, m), nulls)
    return out


def reader_table(S_by_key: dict[str, np.ndarray], conds: list) -> dict:
    ci = {c: i for i, c in enumerate(conds)}
    res = {}
    for key, S in S_by_key.items():
        for d in REAL_DIRS:
            for v in VARIANTS:
                for q in DOSES:
                    res[f"{key}|{d}|{v}|{q}"] = contrast_stats(S, ci, d, v, q, boot=not key.startswith("t") and v == "centred")
    return res


# ----------------------------------------------------------------------------- power
def paired_boot(a_p, a_m, b_p, b_m, stat: str, n_boot: int = 4000, seed: int = 0) -> dict:
    """Difference of a statistic between readers A and B on the same items (item bootstrap)."""
    rng = np.random.default_rng(seed)
    N = len(a_p)

    def f(p, m):
        return M.auroc(p, m) if stat == "auroc" else float(np.mean((p > m) + 0.5 * (p == m)))

    obs = f(a_p, a_m) - f(b_p, b_m)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, N, N)
        bs.append(f(a_p[i], a_m[i]) - f(b_p[i], b_m[i]))
    se = float(np.std(bs, ddof=1))
    n_obs = float(N * (Z80 * se / obs) ** 2) if obs != 0 else float("inf")
    return {"diff": float(obs), "se": se, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_for_80pct_power_at_observed": n_obs,
            "n_for_80pct_power_at_0.05": float(N * (Z80 * se / 0.05) ** 2),
            "n_for_80pct_power_at_0.10": float(N * (Z80 * se / 0.10) ** 2)}


def one_reader_power(p, m, n_boot: int = 4000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    N = len(p)
    obs = M.auroc(p, m)
    bs = [M.auroc(p[i], m[i]) for i in (rng.integers(0, N, N) for _ in range(n_boot))]
    se = float(np.std(bs, ddof=1))
    eff = obs - 0.5
    return {"auroc": float(obs), "se": se,
            "n_for_80pct_power_vs_0.5": float(N * (Z80 * se / eff) ** 2) if eff != 0 else float("inf")}


def _snip(t: str, rx) -> str:
    m = rx.search(t)
    a = max(0, m.start() - 120)
    return t[a:m.end() + 120].replace("\n", " ")


# ----------------------------------------------------------------------------- main
def main() -> None:
    spec = torch.load(DATA / "spec.pt", weights_only=False)
    jl = torch.load(DATA / "jlens.pt", weights_only=False)
    nat = torch.load(DATA / "natural.pt", weights_only=False)
    items = spec["items"]
    N = len(items)
    col = np.array([it["col"] for it in items], float)
    rem = np.array([it["chars_remaining"] for it in items], float)
    assert [tuple(c) for c in jl["conds"]] == JLENS_CONDS
    out: dict = {"n_items": N, "checks": spec["checks"], "ranks": {str(k): v for k, v in spec["ranks"].items()},
                 "doses": {str(L): d for L, d in spec["doses"].items()}, "readers": {}, "leak": {},
                 "fve": {}, "mentions": {}, "natural": {}, "power": {}}

    # ---- decoders
    dec = {"OL": load_decoder("ol"), "NLA": load_decoder("nla")}
    S_all: dict[str, dict] = {}
    for rname, by_layer in dec.items():
        for L, (lo, temps) in by_layer.items():
            name = f"{rname}@{L}"
            if np.isnan(lo).all():
                continue
            S = fam_scores(lo, temps)
            S_all[name] = S
            out["readers"][name] = {"templates": temps, "stats": reader_table(S, conditions(L))}
    # ---- J-Lens (template LO) + probe, full condition list at every layer
    for L in (40, 42, 44):
        lo = jl["lo"][L].numpy()
        S = fam_scores(lo, [(f, a, b) for f, a, b in jl["templates"]])
        S["eol_narrow"] = jl["eol"][L]["narrow"].numpy()
        S["eol_broad"] = jl["eol"][L]["broad"].numpy()
        S_all[f"JLens@{L}"] = S
        out["readers"][f"JLens@{L}"] = {"templates": jl["templates"], "stats": reader_table(S, JLENS_CONDS)}
        Sp = {"ridge": jl["probe"][L].numpy()}
        S_all[f"probe@{L}"] = Sp
        out["readers"][f"probe@{L}"] = {"stats": reader_table(Sp, JLENS_CONDS)}

    # ---- leak verdicts (two-sided vs J-complement band; iso reported)
    for L in (40, 42, 44):
        st = out["readers"][f"JLens@{L}"]["stats"]
        for d in ("jrem1", "jrem4", "raw"):
            for q in DOSES:
                row = {}
                for key in ("eol_narrow", "eol_broad", "count_qual", "count_num"):
                    s = st[f"{key}|{d}|centred|{q}"]
                    b = s["bands"]
                    jc = b.get("nonj1" if d == "jrem1" else "nonj4") if d != "raw" else None
                    row[key] = {"auroc": s["auroc"], "mean_shift": s["mean_shift"],
                                "jc_z_auroc": jc["auroc"]["z"] if jc else None,
                                "jc_z_shift": jc["mean_shift"]["z"] if jc else None,
                                "jc_out": bool(jc and (jc["auroc"]["outside_two_sided"] or
                                                       jc["mean_shift"]["outside_two_sided"])),
                                "iso_z_shift": b["iso"]["mean_shift"]["z"],
                                "iso_out": bool(b["iso"]["auroc"]["outside_two_sided"] or
                                                b["iso"]["mean_shift"]["outside_two_sided"])}
                row["LEAK"] = any(v["jc_out"] for v in row.values() if isinstance(v, dict))
                out["leak"][f"L{L}|{d}|{q}"] = row

    # ---- FVE guard
    fparts = sorted(glob.glob(str(DATA / "fve_part*.pt")))
    fres = {}
    for f in fparts:
        fres.update(torch.load(f, weights_only=False)["res"])
    zero_ci = [ci for ci, r in fres.items() if r["cond"][0] == "zero"]
    expl_by_cond = {}
    if zero_ci:
        nat_fve = fres[zero_ci[0]]["per_example"].numpy()
        for ci_, r in sorted(fres.items()):
            c = r["cond"]
            key = "natural" if c[0] == "zero" else f"{c[0]}|{c[1]}|{c[3]}|{'+' if c[4] > 0 else '-'}"
            expl_by_cond[key] = r["explanations"]
            g = M.fve_guard(nat_fve, r["per_example"].numpy())
            g["n_failed_extractions"] = r["n_failed"]
            out["fve"][key] = g

    # ---- mentions (free generations)
    rx_loose, rx = re.compile(MENTION_RE, re.I), re.compile(MENTION_RE_STRICT, re.I)

    def rate(rows):
        flat = [t for row in rows for t in row if t]
        hits = [t for t in flat if rx.search(t)]
        return {"n": len(flat), "rate": len(hits) / max(len(flat), 1),
                "rate_loose_regex": sum(bool(rx_loose.search(t)) for t in flat) / max(len(flat), 1),
                "items_any": float(np.mean([any(t and rx.search(t) for t in row) for row in rows])),
                "examples": [_snip(h, rx) for h in hits[:4]]}

    for k, rows in expl_by_cond.items():
        out["mentions"][f"NLA|{k}"] = rate(rows)
    gpath = DATA / "gen_ol.pt"
    if gpath.exists():
        g = torch.load(gpath, weights_only=False)
        for c, rows in zip(g["conds"], g["texts"]):
            key = "natural" if c[0] == "zero" else f"{c[1]}|{c[3]}|{'+' if c[4] > 0 else '-'}"
            out["mentions"][f"OL|{key}"] = rate(rows)
        out["ol_natural_examples"] = [r[0][:300] for r in g["texts"][0][:6]]
    if "natural" in expl_by_cond:
        out["nla_natural_examples"] = [r[0][:300] if r[0] else None for r in expl_by_cond["natural"][:6]]
    top = nat["jl_top20"]
    num_re = re.compile(r"^\s*(\d+|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*$", re.I)
    out["mentions"]["JLens@42|natural_top20"] = {
        "items_with_newline_token": float(np.mean([any("\n" in t for t, _ in row) for row in top])),
        "items_with_number_token": float(np.mean([any(num_re.match(t) for t, _ in row) for row in top])),
        "eol_narrow_best_rank_median": float(np.median(nat["eol_rank_narrow"])),
        "example": [[t for t, _ in row[:10]] for row in top[:3]]}

    # ---- natural arm (descriptive): zero-dose LO vs true col
    def nat_corr(x):
        return {"spearman_col": float(spearmanr(x, col).statistic),
                "spearman_chars_remaining": float(spearmanr(x, rem).statistic)}

    zi42 = conditions(42).index(ZERO)
    for name, S in S_all.items():
        zi = (JLENS_CONDS if name.startswith(("JLens", "probe")) else conditions(int(name.split("@")[1]))).index(ZERO)
        out["natural"][name] = {k: nat_corr(v[zi]) for k, v in S.items() if not k.startswith("t")}
    if "textnla_lo" in nat:
        tl = nat["textnla_lo"].numpy()
        temps = nat["textnla_templates"]
        out["natural"]["TextNLA(prompt)"] = {fam: nat_corr(tl[[i for i, t in enumerate(temps) if t[0] == fam]].mean(0))
                                             for fam in FAMILIES}
    _ = zi42

    # ---- power: NLA@42 vs OL@42, centred, per family / direction / dose
    c42 = {c: i for i, c in enumerate(conditions(42))}
    if "NLA@42" in S_all and "OL@42" in S_all:
        for fam in FAMILIES:
            for d in REAL_DIRS:
                for q in DOSES:
                    ip, im = c42[("real", d, "centred", q, 1, d)], c42[("real", d, "centred", q, -1, d)]
                    A, B = S_all["NLA@42"][fam], S_all["OL@42"][fam]
                    out["power"][f"{fam}|{d}|{q}"] = {
                        "nla_minus_ol_auroc": paired_boot(A[ip], A[im], B[ip], B[im], "auroc"),
                        "nla_minus_ol_paired_win": paired_boot(A[ip], A[im], B[ip], B[im], "win"),
                        "nla_vs_0.5": one_reader_power(A[ip], A[im]),
                        "ol_vs_0.5": one_reader_power(B[ip], B[im]),
                        "corr_nla_ol_shift": float(np.corrcoef(A[ip] - A[im], B[ip] - B[im])[0, 1])}
    (RES / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    report(out)


def _b(s: dict, cls: str, k: str) -> str:
    b = s["bands"].get(cls)
    return f"{b[k]['z']:+.1f}{'*' if b[k]['outside_two_sided'] else ''}" if b else "–"


def report(o: dict) -> None:
    print(f"# col pilot (n={o['n_items']}) -- PILOT, EXCLUDED FROM MAIN ANALYSIS\n")
    print("checks:", json.dumps(o["checks"], default=float)[:3000])
    print("\ndoses:", json.dumps(o["doses"], default=float))
    for fam in ("count_qual", "count_num", "decision", "eol_narrow", "eol_broad", "ridge"):
        print(f"\n## family {fam}: centred; AUROC [CI] / paired win / mean shift±SD / zero-sub / "
              f"z(shift) iso, J-compl / z(AUROC) iso, J-compl  (* = outside 95% null band)\n")
        print("| reader | dir | dose | AUROC [CI] | win | shift±SD | 0-sub | z shift iso / Jc | z AUROC iso / Jc | frac |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r, blk in o["readers"].items():
            for d in REAL_DIRS:
                for q in DOSES:
                    s = blk["stats"].get(f"{fam}|{d}|centred|{q}")
                    if not s or s.get("missing"):
                        continue
                    jc = "nonj1" if d == "jrem1" else "nonj4"
                    print(f"| {r} | {d} | {q} | {s['auroc']:.2f} [{s['lo']:.2f},{s['hi']:.2f}] | "
                          f"{s['paired_win']:.2f} | {s['mean_shift']:+.3f}±{s['shift_sd']:.3f} | "
                          f"{s['auroc_zero_sub']:.2f} | {_b(s, 'iso', 'mean_shift')} / {_b(s, jc, 'mean_shift')} | "
                          f"{_b(s, 'iso', 'auroc')} / {_b(s, jc, 'auroc')} | {s['fraction_of_probe']:+.2f} |")
    print("\n## additive (reported) count_qual q90/q99 AUROC / win\n")
    for r, blk in o["readers"].items():
        cells = []
        for d in REAL_DIRS:
            for q in DOSES:
                s = blk["stats"].get(f"count_qual|{d}|add|{q}")
                if s and not s.get("missing"):
                    cells.append(f"{d}/{q} {s['auroc']:.2f}/{s['paired_win']:.2f}")
        if cells:
            print(r, "; ".join(cells))
    print("\n## per-template (centred jrem1) AUROC q90 / q99\n")
    for r, blk in o["readers"].items():
        temps = blk.get("templates") or []
        cells = []
        for i, t in enumerate(temps):
            a = [blk["stats"].get(f"t{i}|jrem1|centred|{q}", {}).get("auroc", float("nan")) for q in DOSES]
            cells.append(f"t{i}[{t[0]}] {a[0]:.2f}/{a[1]:.2f}")
        if cells:
            print(r, "; ".join(cells))
    print("\n## leak test (J-Lens, centred; two-sided vs J-complement band)\n")
    print("| layer|dir|dose | EOL narrow AUROC, z shift Jc | EOL broad | count_qual | count_num | LEAK |")
    print("|---|---|---|---|---|---|")
    for k, row in o["leak"].items():
        def c(x):
            v = row[x]
            z = v["jc_z_shift"]
            return f"{v['auroc']:.2f}, {z:+.1f}{'*' if v['jc_out'] else ''} (iso {v['iso_z_shift']:+.1f}{'*' if v['iso_out'] else ''})" if z is not None else f"{v['auroc']:.2f} (iso {v['iso_z_shift']:+.1f}{'*' if v['iso_out'] else ''})"
        print(f"| {k} | {c('eol_narrow')} | {c('eol_broad')} | {c('count_qual')} | {c('count_num')} | {row['LEAK']} |")
    print("\n## FVE guard (NLA, k=4, paired per example; margin 0.05 / p5)\n")
    for k, g in o["fve"].items():
        print(f"{k}: nat {g['mean_natural']:.3f} steer {g['mean_steered']:.3f} diff {g['mean_diff']:+.3f} "
              f"[{g['lo']:+.3f},{g['hi']:+.3f}] void={g['void']} (n={g['n_pairs']}, failed {g['n_failed_extractions']})")
    print("\n## mentions of line length / characters / wrapping in free readouts\n")
    for k, m in o["mentions"].items():
        print(k, json.dumps({kk: vv for kk, vv in m.items() if kk != "examples"}, default=float),
              ("| e.g. " + " || ".join(m.get("examples", [])[:2])) if m.get("examples") else "")
    print("\n## natural arm (descriptive): Spearman(zero-dose score, col)\n")
    for r, d in o["natural"].items():
        print(r, {k: round(v["spearman_col"], 2) for k, v in d.items()})
    print("\n## power (NLA@42 - OL@42, centred, item bootstrap)\n")
    print("| family|dir|dose | ΔAUROC [CI] (SE) | n@obs | n@0.05 | Δwin (SE) | n@obs | NLA AUROC (n vs .5) | OL AUROC (n vs .5) | r(shift) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, p in o["power"].items():
        a, w = p["nla_minus_ol_auroc"], p["nla_minus_ol_paired_win"]
        print(f"| {k} | {a['diff']:+.3f} [{a['ci'][0]:+.2f},{a['ci'][1]:+.2f}] ({a['se']:.3f}) | "
              f"{a['n_for_80pct_power_at_observed']:.0f} | {a['n_for_80pct_power_at_0.05']:.0f} | "
              f"{w['diff']:+.3f} ({w['se']:.3f}) | {w['n_for_80pct_power_at_observed']:.0f} | "
              f"{p['nla_vs_0.5']['auroc']:.2f} ({p['nla_vs_0.5']['n_for_80pct_power_vs_0.5']:.0f}) | "
              f"{p['ol_vs_0.5']['auroc']:.2f} ({p['ol_vs_0.5']['n_for_80pct_power_vs_0.5']:.0f}) | "
              f"{p['corr_nla_ol_shift']:+.2f} |")


if __name__ == "__main__":
    main()
