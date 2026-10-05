"""Steps 3-4 (local): select 3 templates per reader per feature by NATURAL-ITEM discriminability on
the selection split (paired log-odds AUROC, true vs alternative value), keep the fixed label template
as a 4th column, and freeze -> templates_frozen.json (+ sha256 printed for RESULTS.md).

Selection rule (fixed before looking at any score): rank the MINED candidates of a reader by
selection-split AUROC at the reader's primary layer (NLA@42, OL@42, J-Lens@42); ties broken by d'
of LO between the two classes; greedily take the best 3 distinct (template, fillers) candidates. The fixed
label template is never eligible (it is the 4th column). No steering data is used anywhere.

    python select_freeze.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from feature_spec import (BINS, BREAK_COL_RANGE, BREAK_FAR_MIN, BREAK_SOON_MAX, FEATURES,
                          N_SELECT_TEMPLATES, OL_LAYERS, QUALITATIVE_PRIMARY, fill, is_numeric)  # noqa: E402
from mats.metrics import auroc_ci  # noqa: E402

RES = HERE / "results"
FROZEN = HERE / "templates_frozen.json"
PRIMARY = {"nla": "nla@42", "ol": "ol@42", "jlens": "jlens@42"}
COUNT_KEY = {"nj1_col": "col", "bk_wordpos": "words", "bk_listidx": "index", "bk_indent": "level",
             "ij_listnum": "index", "nj1_col_break": "remaining_to_width"}
NOT_ELIGIBLE = ("fixed_label", "fixed_numeric")


def dprime(lo: np.ndarray, y: np.ndarray) -> float:
    a, b = lo[y == 1], lo[y == 0]
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-9
    return float((a.mean() - b.mean()) / sd)


def stats(lo: list[float], y: list[int], clusters: list[str]) -> dict:
    s, yy = np.asarray(lo, float), np.asarray(y, int)
    r = auroc_ci(s, yy, clusters, n_boot=1000)
    return {"auroc": round(r.auroc, 4), "ci95": [round(r.lo, 4), round(r.hi, 4)],
            "dprime": round(dprime(s, yy), 3), "n": r.n}


def spearman(x: np.ndarray, c: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, c).statistic)


def main() -> None:
    cand = json.loads((RES / "candidates.json").read_text())
    frozen = {"templates": {"nla": {}, "ol": {}, "jlens": {}}, "items": {}}
    table = []
    for f in FEATURES:
        p_ol, p_nla = RES / f"score_ol_{f}.json", RES / f"score_nla_{f}.json"
        if not (p_ol.exists() and p_nla.exists()):
            print(f"skip {f}: scores missing")
            continue
        g = json.loads((RES / f"gen_ol_{f}.json").read_text())
        items = {it["id"]: it for it in g["items"]}
        frozen["items"][f] = {s: sorted(i for i, it in items.items() if it["split"] == s)
                              for s in ("mine", "select")}
        scores = {**json.loads(p_ol.read_text())["scores"], **json.loads(p_nla.read_text())["scores"]}
        for reader in ("nla", "ol", "jlens"):
            temps = cand[reader][f]
            keys = [PRIMARY[reader]] + ([f"{reader}@{L}" for L in OL_LAYERS if L != 42]
                                       if reader in ("ol", "jlens") else [])
            per = {}
            for t in temps:
                row = {}
                for k in keys:
                    sc = scores.get(k, {})
                    if t["id"] not in sc.get("lo", {}):
                        continue
                    clusters = [items[i]["cluster"] for i in sc["ids"]]
                    row[k] = stats(sc["lo"][t["id"]], sc["y"], clusters)
                    if f in COUNT_KEY:
                        cnt = np.array([items[i]["meta"][COUNT_KEY[f]] for i in sc["ids"]], float)
                        row[k]["spearman_count"] = round(spearman(np.asarray(sc["lo"][t["id"]]), cnt), 3)
                if row:
                    per[t["id"]] = row
            mined = [t for t in temps if t["family"] not in NOT_ELIGIBLE and t["id"] in per
                     and PRIMARY[reader] in per[t["id"]]]
            mined.sort(key=lambda t: (-per[t["id"]][PRIMARY[reader]]["auroc"],
                                      -per[t["id"]][PRIMARY[reader]]["dprime"]))

            def pick(pool: list[dict]) -> list[dict]:
                out, seen = [], set()
                for t in pool:
                    key = json.dumps([t.get("template"), t.get("fillers"), t.get("words")], sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(t)
                    if len(out) == N_SELECT_TEMPLATES:
                        break
                return out

            secondary = []
            if f in QUALITATIVE_PRIMARY and reader != "jlens":
                chosen = pick([t for t in mined if not is_numeric(t)])
                secondary = pick([t for t in mined if is_numeric(t)])
            else:
                chosen = pick(mined)
            fixed = next(t for t in temps if t["family"] == "fixed_label")
            # template-averaged LO (prereg §6.1: mean over the 3 templates)
            mean3 = {}
            for k in keys:
                sc = scores.get(k, {})
                los = [sc["lo"][t["id"]] for t in chosen if t["id"] in sc.get("lo", {})]
                if los:
                    clusters = [items[i]["cluster"] for i in sc["ids"]]
                    mean3[k] = stats(list(np.mean(np.asarray(los, float), 0)), sc["y"], clusters)

            def entry(t: dict) -> dict:
                e = {k: t[k] for k in ("id", "family", "template", "fillers", "words") if k in t}
                vals = FEATURES[f]["values"]
                if "template" in t:
                    e["candidates"] = {v: fill(t, v) for v in vals}
                e["selection"] = per.get(t["id"], {})
                return e

            extra_fixed = [t for t in temps if t["family"] == "fixed_numeric"]
            frozen["templates"][reader][f] = {
                "selected": [entry(t) for t in chosen],
                **({"secondary_numeric": [entry(t) for t in secondary],
                    "primary_rule": "qualitative primary, numerals secondary (Fable round 4)"}
                   if f in QUALITATIVE_PRIMARY and reader != "jlens" else {}),
                "fixed_label": entry(fixed),
                **({"fixed_numeric": entry(extra_fixed[0])} if extra_fixed else {}),
                "mean_of_selected": mean3,
                "n_candidates_mined": len([t for t in temps if t["family"] not in NOT_ELIGIBLE]),
                "all_candidates": {tid: r for tid, r in per.items()},
            }
            sel_au = [per[t["id"]][PRIMARY[reader]]["auroc"] for t in chosen]
            table.append((f, reader, sel_au, mean3.get(PRIMARY[reader], {}).get("auroc"),
                          per.get(fixed["id"], {}).get(PRIMARY[reader], {}).get("auroc"),
                          {k: mean3[k]["auroc"] for k in mean3 if k != PRIMARY[reader]}))
    frozen["meta"] = {
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "procedure": "D9 item 6: mined on a held-out MINING split (k=8 free readouts per item: NLA@42 "
                     "iter_000400, OL s3d-rl600 @42/40/44 pooled; J-Lens n1000 top-50 @42), LLM-assisted "
                     "extraction (Gemini 3.8 Flash), selected by natural-item paired log-odds AUROC on a "
                     "disjoint SELECTION split at the primary layer (42), before any steering run.",
        "scoring": "NLA: candidate after '<explanation>\\n'; OL: candidate as the first '- ' bullet "
                   "(prefix ''); J-Lens: max cosine score over single-token variants, s(a)-s(b). "
                   "Both candidates of an activation share one forward batch. LO = log p(value_a) - "
                   "log p(value_b) with value_a = FEATURES[f]['values'][0]; AUROC vs label (true == a).",
        "values": {f: FEATURES[f]["values"] for f in FEATURES},
        "bins": BINS,
        "col_break_def": {"soon_max_remaining": BREAK_SOON_MAX, "far_min_remaining": BREAK_FAR_MIN,
                          "col_range_both": BREAK_COL_RANGE},
        "read_layers": {"nla": [42], "ol": list(OL_LAYERS), "jlens": list(OL_LAYERS)},
        "exclude_from_evaluation": "every id in items[feature]['mine'|'select'] (all items are freshly "
                                   "generated and disjoint from 02's wrap/prose pickles).",
    }
    text = json.dumps(frozen, indent=1, ensure_ascii=False, sort_keys=True)
    FROZEN.write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (RES / "frozen_sha256.txt").write_text(sha + "\n")
    print(f"froze {FROZEN.name} sha256={sha}")
    print(f"{'feature':14s} {'reader':6s} {'sel3 @42':>22s} {'mean3':>6s} {'fixed':>6s}  other layers (mean3)")
    for f, r, s, m, fx, oth in table:
        sj = "/".join(f"{x:.2f}" for x in s)
        ms = f"{m:.2f}" if m is not None else "  -  "
        fs = f"{fx:.2f}" if fx is not None else "  -  "
        print(f"{f:14s} {r:6s} {sj:>22s} {ms:>6s} {fs:>6s}  {json.dumps({k: round(v, 2) for k, v in oth.items()})}")
    (RES / "selection_table.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
