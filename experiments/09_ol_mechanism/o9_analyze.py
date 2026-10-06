"""09 OL-mechanism analysis (S6 a/b/c) -- DECLARED PILOT SPLIT, EXCLUDED FROM THE MAIN ANALYSIS.

    ../../.venv/bin/python o9_analyze.py   -> results/o9_metrics.json, results/o9_stdout.txt

CPU only; reads results/09_ol_mechanism/* (downloaded from the rfm-results volume) and, for the r5 tense
adapter-ON bands, ../08_col_pilot/results/08_col_pilot/r5_ol_chunk*.pt (read only).

Conventions (identical to 08 r5_analyze.py): per item p = LO(+dose), m = LO(-dose); shift = mean(p - m);
AUROC / paired win from mats.metrics; z = (stat - null mean) / null sd over the K = 20 iso and K = 20
J-complement (nonj4) null directions at the matched norm, never subtracted. Gain ratio = mean OL shift /
mean model-behaviour shift on the same items and the same edit (both p - m, so the ratio is per unit dose).

Analysis choices fixed here (NOT pre-registered; the prereg names the directions but not these details):
  * gain-ratio dose = r108 (0.108 ||h_i||; the one relative dose that has a null block in every set);
    q90 (col/tense) and r216 (random) are reported as sensitivity;
  * random directions are scored on col09 + tense09 pooled (160 prose items); the two used are the two with
    the largest behavioural z (min over both bands) at r108 among randA-D -- chosen from behaviour only;
  * indentation primary cells: indent_raw and indent_jrem4 at r108 with the ANALOGUE channel (codeNL
    ind_cont, codeIN ind_nlcont); S6(a) "fires" if ANY of these four has z_shift >= +3 against BOTH bands.
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mats import metrics as M  # noqa: E402
from o9_lib import K_NULL, PLAN, conditions, null_key  # noqa: E402
from o9_templates import ANALOGUE  # noqa: E402

RES = HERE / "results"
DATA = RES / "09_ol_mechanism"
R5DATA = HERE.parent / "08_col_pilot" / "results" / "08_col_pilot"
NB_AUC, NB_RATIO = 1000, 2000
GPU_USD_PER_H = 2.50  # A100-80GB
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


# ----------------------------------------------------------------------------- loading
def load_ol(S: str, adapter: str) -> dict[str, np.ndarray]:
    """{family: [C, N]} (mean over the family's pairs) + {"tK": per-pair}."""
    C = len(conditions(S))
    files = sorted(glob.glob(str(DATA / f"ol_{adapter}_{S}_chunk*.pt")))
    X, temps, seen = None, None, set()
    for f in files:
        d = torch.load(f, weights_only=False)
        lo = d["lo"].numpy()
        if X is None:
            X, temps = np.full((lo.shape[0], C, lo.shape[2]), np.nan), d["templates"]
        assert d["templates"] == temps
        X[:, d["idx"], :] = lo
        seen |= set(d["idx"])
    assert len(seen) == C and not np.isnan(X).any(), (S, adapter, len(seen), C)
    fams = list(dict.fromkeys(t[0] for t in temps))
    out = {fam: X[[i for i, t in enumerate(temps) if t[0] == fam]].mean(0) for fam in fams}
    return out | {f"t{i}": X[i] for i in range(len(temps))}


def load_behave(S: str) -> dict[str, np.ndarray]:
    b = torch.load(DATA / f"behave_{S}.pt", weights_only=False)
    assert b["conds"] == conditions(S)
    return {r: np.stack([v[r] for v in b["vals"]]) for r in b["readouts"]}


# ----------------------------------------------------------------------------- cells
def pm(S: str, X: np.ndarray, d: str, q: str, conds: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    ci = {c: i for i, c in enumerate(conds or conditions(S))}
    return X[ci[("real", d, q, 1, d)]], X[ci[("real", d, q, -1, d)]]


def bands(S: str, X: np.ndarray, p: np.ndarray, m: np.ndarray, q: str, ref, conds=None, k=K_NULL) -> dict:
    ci = {c: i for i, c in enumerate(conds or conditions(S))}
    out = {}
    for cls in ("iso", "nonj4"):
        nl = [(X[ci[("null", f"{cls}_{j}", q, 1, ref)]], X[ci[("null", f"{cls}_{j}", q, -1, ref)]]) for j in range(k)]
        out[cls] = M.null_band_contrast((p, m), nl)
    return out


def cell(S: str, X: np.ndarray, d: str, q: str, boot: bool = True) -> dict:
    p, m = pm(S, X, d, q)
    r = M.follow_feature(p, m, n_boot=NB_AUC) if boot else M.paired_stats(p, m) | {"lo": np.nan, "hi": np.nan}
    out = {"auroc": r["auroc"], "lo": r["lo"], "hi": r["hi"], "win": r["paired_win"], "shift": r["mean_shift"],
           "item_shift": (p - m).tolist(), "bands": {}}
    nk = null_key(S, d, q)
    if nk:
        out["bands"] = bands(S, X, p, m, *nk)
    return out


def zs(c: dict, k: str = "mean_shift") -> tuple[float, float]:
    b = c.get("bands") or {}
    return (b["iso"][k]["z"], b["nonj4"][k]["z"]) if b else (np.nan, np.nan)


def fmt(c: dict) -> str:
    zi, zj = zs(c)
    ai, aj = zs(c, "auroc")
    star = lambda cls, k: "*" if c["bands"] and c["bands"][cls][k]["outside_two_sided"] else ""  # noqa: E731
    core = f"{c['auroc']:.2f} [{c['lo']:.2f},{c['hi']:.2f}] / {c['win']:.2f} / {c['shift']:+.2f}"
    if not c["bands"]:
        return core + "; (no matched null)"
    return (core + f"; {zi:+.1f}{star('iso', 'mean_shift')} {zj:+.1f}{star('nonj4', 'mean_shift')}"
            f" / {ai:+.1f}{star('iso', 'auroc')} {aj:+.1f}{star('nonj4', 'auroc')}")


# ----------------------------------------------------------------------------- (a) indentation
def indentation(OL: dict, BE: dict) -> dict:
    say("## (a) INDENTATION (codeNL / codeIN; + = deeper; cell = AUROC [CI] / win / shift; z_shift iso, Jc / z_AUROC iso, Jc)")
    out, fires = {}, []
    for S, ch in (("codeNL", "ind_cont"), ("codeIN", "ind_nlcont")):
        for d in ("indent_raw", "indent_jrem4", "nextind_raw", "nextind_jrem4"):
            for q in PLAN[S]["dirs"][d]:
                row = {"beh": cell(S, BE[S]["ind8v4"], d, q)}
                for ad in ("on", "off"):
                    for fam in ("ind_cont", "ind_nlcont", "ind_any", "nl_cont", "f07_indent"):
                        row[f"{ad}|{fam}"] = cell(S, OL[(S, ad)][fam], d, q, boot=(fam == ch))
                out[f"{S}|{d}|{q}"] = row
                if q != "r108" and not d.startswith("indent"):
                    continue
                say(f"- {S} {d} {q}: model ind8v4 {fmt(row['beh'])}")
                say(f"    OL-on {ch}: {fmt(row['on|' + ch])}")
                say(f"    OL-off {ch}: {fmt(row['off|' + ch])}")
                if q == "r108":
                    other = ", ".join(f"{fam} {row['on|' + fam]['shift']:+.2f} ({zs(row['on|' + fam])[0]:+.1f},{zs(row['on|' + fam])[1]:+.1f})"
                                      for fam in ("ind_cont", "ind_nlcont", "ind_any", "nl_cont", "f07_indent") if fam != ch)
                    say(f"    OL-on other channels (shift; z iso,Jc): {other}")
                if d.startswith("indent") and q == "r108":
                    zi, zj = zs(row["on|" + ch])
                    fires.append({"cell": f"{S}|{d}|{q}|{ch}", "z_iso": zi, "z_jc": zj, "fires": bool(zi >= 3 and zj >= 3),
                                  "beh_z": zs(row["beh"])})
    a_fires = any(f["fires"] for f in fires)
    say(f"S6(a) cells (OL-on analogue channel, r108): " + "; ".join(
        f"{f['cell']} z {f['z_iso']:+.1f}/{f['z_jc']:+.1f}" for f in fires) + f" -> (a) fires: {a_fires}")
    return {"rows": out, "s6a_cells": fires, "s6a_fires": a_fires}


# ----------------------------------------------------------------------------- (b) gain ratio
def _ratio_boot(groups: list[tuple[np.ndarray, np.ndarray]], seed: int) -> np.ndarray:
    """groups: (ol_item_shift, beh_item_shift) per item set; resample items within each set."""
    rng = np.random.default_rng(seed)
    out = np.empty(NB_RATIO)
    for b in range(NB_RATIO):
        o = bb = 0.0
        for ol, be in groups:
            ix = rng.integers(0, len(ol), len(ol))
            o, bb = o + ol[ix].sum(), bb + be[ix].sum()
        out[b] = o / bb
    return out


def gain(OL: dict, BE: dict) -> dict:
    say("\n## (b) GAIN RATIO = OL continuation shift / model behavioural shift (same items, same edit, adapter ON)")
    rand_sel = {}
    for r in ("randA", "randB", "randC", "randD"):
        zz = [zs(cell(S, BE[S][r], r, "r108")) for S in ("col", "tense")]
        pooled_beh = np.concatenate([np.subtract(*pm(S, BE[S][r], r, "r108")) for S in ("col", "tense")]).mean()
        rand_sel[r] = {"beh_shift_pooled": float(pooled_beh), "beh_z_col": zz[0], "beh_z_tense": zz[1],
                       "min_z": float(min(min(z) for z in zz))}
    chosen = sorted(rand_sel, key=lambda r: -rand_sel[r]["min_z"])[:2]
    say("random directions, model behaviour at r108 (own readout; shift pooled col+tense; z iso/Jc col | tense): " + "; ".join(
        f"{r} {v['beh_shift_pooled']:+.3f} ({v['beh_z_col'][0]:+.1f}/{v['beh_z_col'][1]:+.1f} | "
        f"{v['beh_z_tense'][0]:+.1f}/{v['beh_z_tense'][1]:+.1f})" for r, v in rand_sel.items()) + f" -> used: {chosen}")
    spec = {"col_raw": [("col", "reg_raw")], "col_4r": [("col", "reg_jrem4")], "tense_raw": [("tense", "raw")],
            "tense_4r": [("tense", "jrem4")], "tense_raw~broad": [("tense", "raw", "tense_broad")],
            "tense_4r~broad": [("tense", "jrem4", "tense_broad")], **{r: [("col", r), ("tense", r)] for r in ("randA", "randB", "randC", "randD")}}
    analog = lambda d: ANALOGUE["col" if d.startswith("reg") else "tense" if d in ("raw", "jrem4") else d]  # noqa: E731
    res = {}
    for name, members in spec.items():
        doses = sorted({q for S, d, *_ in members for q in PLAN[S]["dirs"][d]}, key=lambda q: (q[0] != "q", q))
        for q in doses:
            groups, olz, bez = [], [], []
            for S, d, *ro_over in members:
                ch, ro = analog(d)
                ro = ro_over[0] if ro_over else ro
                p, m = pm(S, OL[(S, "on")][ch], d, q)
                bp, bm = pm(S, BE[S][ro], d, q)
                groups.append((p - m, bp - bm))
                if null_key(S, d, q):
                    olz.append(zs(cell(S, OL[(S, "on")][ch], d, q, boot=False)))
                    bez.append(zs(cell(S, BE[S][ro], d, q, boot=False)))
            o = np.concatenate([g[0] for g in groups]).mean()
            be = np.concatenate([g[1] for g in groups]).mean()
            bs = _ratio_boot(groups, seed=zlib.crc32((name + q).encode()))
            res[f"{name}|{q}"] = {"ol_shift": float(o), "beh_shift": float(be), "ratio": float(o / be),
                                  "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                  "ol_z": olz, "beh_z": bez, "boot": bs}
    say("| direction | dose | OL shift | model shift | ratio [95% CI] | OL z (iso/Jc) | model z (iso/Jc) |")
    say("|---|---|---|---|---|---|---|")
    for k, v in res.items():
        z1 = "; ".join(f"{a:+.1f}/{b:+.1f}" for a, b in v["ol_z"]) or "-"
        z2 = "; ".join(f"{a:+.1f}/{b:+.1f}" for a, b in v["beh_z"]) or "-"
        n, q = k.split("|")
        say(f"| {n} | {q} | {v['ol_shift']:+.3f} | {v['beh_shift']:+.3f} | {v['ratio']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] | {z1} | {z2} |")
    six = ["col_raw", "col_4r", "tense_raw", "tense_4r", *chosen]
    rule = {}
    for q_cs, q_r in (("r108", "r108"), ("q90", "r108"), ("q90", "r216")):  # the prereg rule, literally
        keys = [f"{n}|{q_cs if not n.startswith('rand') else q_r}" for n in six]
        rat = np.array([res[k]["ratio"] for k in keys])
        mu = rat.mean()
        within = bool(np.all(np.abs(rat - mu) <= 0.25 * abs(mu)))
        others = float(np.delete(rat, 1).mean())
        c4 = res[keys[1]]
        reading_b = bool(c4["ratio"] > 1.25 * others and c4["ci"][0] > others)
        rule[f"{q_cs}/{q_r}"] = {"keys": keys, "ratios": rat.tolist(), "mean": float(mu),
                                 "dev_pct": ((rat - mu) / mu * 100).tolist(), "all_within_25pct": within,
                                 "mean_others_than_col4r": float(others), "col4r_ci": c4["ci"], "reading_b": reading_b}
        say(f"rule (b) at col/tense {q_cs}, random {q_r}: ratios {np.round(rat, 2).tolist()} mean {mu:.2f}; "
            f"dev% {np.round((rat - mu) / mu * 100).astype(int).tolist()}; all within +/-25%: {within}; "
            f"col4r {c4['ratio']:.2f} [{c4['ci'][0]:.2f},{c4['ci'][1]:.2f}] vs mean of other five {others:.2f} -> reading(b): {reading_b}")
    # SENSITIVITY (not the prereg rule): drop directions whose model shift is not >= +3 z on both bands
    # (an undefined ratio), and/or use the 02 G4 tense readout (tense_broad) for the tense denominators.
    for tag, tnames in (("valid-denominator", ("tense_raw", "tense_4r")), ("valid-denominator, tense_broad", ("tense_raw~broad", "tense_4r~broad"))):
        names = ["col_raw", "col_4r", *tnames, *chosen]
        keys = [f"{n}|r108" for n in names]
        ok = [k for k in keys if res[k]["beh_z"] and min(min(z) for z in res[k]["beh_z"]) >= 3]
        rat = np.array([res[k]["ratio"] for k in ok])
        mu = rat.mean()
        oth = float(np.mean([res[k]["ratio"] for k in ok if not k.startswith("col_4r")]))
        c4 = res["col_4r|r108"]
        rule[tag] = {"keys": ok, "dropped": [k for k in keys if k not in ok], "ratios": rat.tolist(), "mean": float(mu),
                     "all_within_25pct": bool(np.all(np.abs(rat - mu) <= 0.25 * abs(mu))), "mean_others_than_col4r": oth,
                     "reading_b": bool(c4["ratio"] > 1.25 * oth and c4["ci"][0] > oth)}
        say(f"SENSITIVITY ({tag}, r108): kept {ok}, dropped {rule[tag]['dropped']}; ratios {np.round(rat, 2).tolist()} "
            f"mean {mu:.2f}; within +/-25%: {rule[tag]['all_within_25pct']}; col4r {c4['ratio']:.2f} vs others {oth:.2f} "
            f"-> reading(b): {rule[tag]['reading_b']}")
    say(f"J-removal retention per unit dose at r108 (4r*/raw): col OL {res['col_4r|r108']['ol_shift'] / res['col_raw|r108']['ol_shift']:.2f}, "
        f"col model {res['col_4r|r108']['beh_shift'] / res['col_raw|r108']['beh_shift']:.2f}; "
        f"tense OL {res['tense_4r|r108']['ol_shift'] / res['tense_raw|r108']['ol_shift']:.2f}, "
        f"tense model (tense_tok) {res['tense_4r|r108']['beh_shift'] / res['tense_raw|r108']['beh_shift']:.2f}, "
        f"(tense_broad) {res['tense_4r~broad|r108']['beh_shift'] / res['tense_raw~broad|r108']['beh_shift']:.2f}")
    for v in res.values():
        v.pop("boot")
    return {"random_selection_behaviour": rand_sel, "chosen_random": chosen, "cells": res, "rule": rule}


# ----------------------------------------------------------------------------- (c) adapter off
def adapter_off(OL: dict) -> dict:
    say("\n## (c) ADAPTER OFF vs ON (base Qwen in OL's carrier, same slot injection)")
    out = {}
    for S, chans, dirs in (("col", ("nl_cont",), ("reg_raw", "reg_jrem4")), ("tense", ("tense_tok", "f07_tense"), ("raw", "jrem4"))):
        for ch in chans:
            for d in dirs:
                for q in PLAN[S]["dirs"][d]:
                    on, off = cell(S, OL[(S, "on")][ch], d, q), cell(S, OL[(S, "off")][ch], d, q)
                    out[f"{S}|{ch}|{d}|{q}"] = {"on": on, "off": off}
                    say(f"- {S}09 {ch} {d} {q}: ON {fmt(on)} | OFF {fmt(off)}")
    z0 = {f"{S}|{ch}|{ad}": float(OL[(S, ad)][ch][0].mean()) for S, chs in (("col", ("nl_cont",)), ("tense", ("tense_tok", "f07_tense")))
          for ch in chs for ad in ("on", "off")}
    say(f"zero-dose mean LO: {json.dumps({k: round(v, 2) for k, v in z0.items()})}")
    out["zero_dose"] = z0
    out["r5_tense120"] = tense_r5()
    return out


def _r5_on_f07() -> tuple[np.ndarray, list]:
    sys.path.insert(0, str(HERE.parent / "08_col_pilot"))
    from r5_lib import conditions as r5c
    conds = r5c("tense")
    X = None
    for f in sorted(glob.glob(str(R5DATA / "r5_ol_chunk*.pt"))):
        d = torch.load(f, weights_only=False)
        blk, temps = d["lo"]["tense"], d["templates"]["tense"]
        ti = [i for i, t in enumerate(temps) if t[0] == "f07_tense"]
        if X is None:
            X = np.full((len(conds), blk["lo"].shape[2]), np.nan)
        X[blk["idx"]] = blk["lo"].numpy()[ti].mean(0)
    return X, conds


def tense_r5() -> dict:
    sys.path.insert(0, str(HERE.parent / "08_col_pilot"))
    from r5_lib import null_keys
    t = torch.load(DATA / "tense_r5_mech.pt", weights_only=False)
    conds, temps = t["conds"], t["templates"]
    off, on_real = t["ol_off"].numpy(), t["ol_on"].numpy()
    on = np.full_like(off, np.nan)
    on[:, t["on_idx"]] = on_real
    r5on, r5conds = _r5_on_f07()
    assert r5conds == conds
    out = {}
    say("r5 tense120 items (08 round 5), same activations; ON-r5 = 08's original r5 OL chunks (bands available for f07_tense only)")
    for fam in ("f07_tense", "tense_tok"):
        ti = [i for i, x in enumerate(temps) if x[0] == fam]
        Xoff, Xon = off[ti].mean(0), on[ti].mean(0)
        for d in ("raw", "jrem4"):
            for q in ("q90", "r11"):
                ref, ok = null_keys("tense", d, q)
                p, m = pm("tense", Xoff, d, q, conds)
                c_off = M.follow_feature(p, m, n_boot=NB_AUC) | {"bands": bands("tense", Xoff, p, m, q, ref, conds) if ok else {}}
                p2, m2 = pm("tense", Xon, d, q, conds)
                c_on = M.follow_feature(p2, m2, n_boot=NB_AUC)
                row = {"off": c_off, "on_rerun": c_on}
                s = (f"- r5 {fam} {d} {q}: ON-rerun {c_on['auroc']:.2f} / {c_on['paired_win']:.2f} / {c_on['mean_shift']:+.2f}"
                     f" | OFF {c_off['auroc']:.2f} [{c_off['lo']:.2f},{c_off['hi']:.2f}] / {c_off['paired_win']:.2f} / {c_off['mean_shift']:+.2f}")
                if ok:
                    s += f"; z_shift {c_off['bands']['iso']['mean_shift']['z']:+.1f} {c_off['bands']['nonj4']['mean_shift']['z']:+.1f}"
                if fam == "f07_tense":
                    p3, m3 = pm("tense", r5on, d, q, conds)
                    c5 = M.paired_stats(p3, m3) | {"bands": bands("tense", r5on, p3, m3, q, ref, conds) if ok else {}}
                    row["on_r5"] = c5
                    row["repro_max_abs_item_diff"] = float(np.nanmax(np.abs(np.r_[p2 - p3, m2 - m3])))
                    s += f" | ON-r5 {c5['auroc']:.2f} / {c5['mean_shift']:+.2f}"
                    if ok:
                        s += f"; z {c5['bands']['iso']['mean_shift']['z']:+.1f} {c5['bands']['nonj4']['mean_shift']['z']:+.1f}"
                    s += f" (max item |diff| rerun vs r5 {row['repro_max_abs_item_diff']:.2f})"
                say(s)
                out[f"{fam}|{d}|{q}"] = row
    return out


# ----------------------------------------------------------------------------- sanity / provenance
def sanity(OL: dict, BE: dict) -> dict:
    say("\n## SANITY (known positives; r5 = 08 round 5 on its 60 col items, these are 80 fresh col09 items)")
    out = {}
    ref = {"reg_raw|q90": "r5 0.76 / 0.97 / +2.88; +6.6 +6.7", "reg_jrem4|q90": "r5 0.67 / 0.98 / +1.72; +3.9 +3.9 (proxy null)",
           "reg_raw|r108": "r5 r11: 0.77 / +3.00", "reg_jrem4|r108": "r5 r11: 0.72 / +2.32"}
    for k, r in ref.items():
        d, q = k.split("|")
        c = cell("col", OL[("col", "on")]["nl_cont"], d, q)
        b = cell("col", BE["col"]["nl_logit_broad"], d, q)
        out[k] = {"ol": c, "beh_broad": b}
        say(f"- OL-on nl_cont {d} {q}: {fmt(c)}   [{r}]")
        say(f"    model newline log-odds (broad EOL), effect (p-m)/2 = {b['shift'] / 2:+.2f}; z {zs(b)[0]:+.0f}/{zs(b)[1]:+.0f}"
            f"   [r5 q90: raw +1.44, 4r* +0.72; r11: +1.50 / +1.06]")
    return out


def provenance() -> dict:
    say("\n## PROVENANCE")
    fixed = {}
    for ln in (RES / "fixed_files.txt").read_text().splitlines():
        parts = ln.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            fixed[parts[1]] = parts[0]
    chk = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() == h for f, h in fixed.items()}
    say(f"sha256 vs results/fixed_files.txt: {chk} -> all match: {all(chk.values())}")
    rs = json.loads((RES / "run_summary.json").read_text())
    tsum = {k: v["t"] for k, v in rs.items()}
    spec = torch.load(DATA / "spec.pt", weights_only=False)
    t_prep = spec["checks"]["t_select"] + 0.0
    gpu_s = sum(tsum.values()) + t_prep
    errs = [k for k, v in rs.items() if "error" in v]
    sel = json.loads((DATA / "random_selection.json").read_text())
    say(f"run_summary: {len(rs)} calls, errors: {errs or 'none'}; in-function GPU time (excl. cold start / image pull): "
        f"prep>={t_prep / 3600:.2f} h, behave {sum(v for k, v in tsum.items() if k.startswith('behave')) / 3600:.2f} h, "
        f"tense_r5 {tsum['tense_r5'] / 3600:.2f} h, score_ol {sum(v for k, v in tsum.items() if k.startswith('score_ol')) / 3600:.2f} h; "
        f"total {gpu_s / 3600:.1f} A100-80GB h ~ ${gpu_s / 3600 * GPU_USD_PER_H:.0f} at ${GPU_USD_PER_H}/h (lower bound; earlier failed attempts not included)")
    say(f"random_selection.json written_at {sel['written_at']} (before any OL scoring; OL calls spawned 2026-10-06T10:33Z per call-id ULIDs)")
    return {"sha_ok": chk, "t": tsum, "t_prep": t_prep, "gpu_h": gpu_s / 3600, "usd": gpu_s / 3600 * GPU_USD_PER_H}


# ----------------------------------------------------------------------------- main
def classify(a: dict, b: dict) -> dict:
    prim = b["rule"]["r108/r108"]
    acting = (not a["s6a_fires"]) and prim["all_within_25pct"]
    reading = a["s6a_fires"] or prim["reading_b"]
    verdict = "reading" if reading else "acting" if acting else "inconclusive"
    say("\n## S6 CLASSIFICATION (prereg §10.1, applied literally; primary dose r108 for all six directions)")
    say(f"(a) indentation fires: {a['s6a_fires']}; (b) all six ratios within +/-25% of mean: {prim['all_within_25pct']}; "
        f"(b) col4r exceeds others by >25% with CI excluding: {prim['reading_b']}")
    say(f"sensitivity: " + "; ".join(f"{k}: within={v['all_within_25pct']}, reading_b={v['reading_b']}" for k, v in b["rule"].items()))
    say(f"VERDICT: S6 = {verdict.upper()}")
    return {"verdict": verdict, "acting": acting, "reading": reading}


def main() -> None:
    say("# 09 OL mechanism (S6) -- PILOT, EXCLUDED FROM MAIN ANALYSIS")
    OL = {(S, ad): load_ol(S, ad) for S in ("codeNL", "codeIN", "col", "tense") for ad in ("on", "off")}
    BE = {S: load_behave(S) for S in ("codeNL", "codeIN", "col", "tense")}
    prov = provenance()
    san = sanity(OL, BE)
    a = indentation(OL, BE)
    b = gain(OL, BE)
    c = adapter_off(OL)
    v = classify(a, b)
    strip = lambda o: {k: strip(x) for k, x in o.items() if k != "item_shift"} if isinstance(o, dict) else o  # noqa: E731
    (RES / "o9_metrics.json").write_text(json.dumps(strip({"provenance": prov, "sanity": san, "a": a, "b": b, "c": c, "s6": v}),
                                                       indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    (RES / "o9_stdout.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
