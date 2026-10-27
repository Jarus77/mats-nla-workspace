"""Score the exp 18 readers against the pre-registered predictions (prereg §5).

Observed cells (prereg §3): NLA / OLens / J-lens prompted AUROC at 42 (max of the 3-template mean and the
fixed label, exp 13's convention) from templates_frozen.json; NLA stated rate and OLens stated / exhibited
rates from the judge files; the NLA complement-naming rate from the fixed pair's sign on unmarked items.

    .venv/bin/python score19.py        -> results/score19.json, results/scorecard19.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "12_concept_recall"))
from feature_spec19 import CELL_THRESH, FEATURES  # noqa: E402

CELLS = list(CELL_THRESH)
READER_OF = {"nla_prompted": "nla", "nla_states": "nla", "nla_names_complement": "nla", "ol_prompted": "ol",
             "ol_exhibits": "ol", "ol_states": "ol", "jlens_prompted": "jlens"}


def score_row(pred: dict, obs: dict) -> dict:
    out = {}
    for c in CELLS:
        v = obs.get(c)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out[c] = {"predicted_high": pred[c], "observed": None, "observed_high": None, "correct": None}
            continue
        high = v >= CELL_THRESH[c]
        out[c] = {"predicted_high": bool(pred[c]), "observed": float(v), "observed_high": bool(high),
                  "correct": bool(high == bool(pred[c]))}
    return out


def accuracy(rows: list[dict]) -> dict:
    cells = [c for r in rows for c in r.values() if c["correct"] is not None]
    n, k = len(cells), sum(c["correct"] for c in cells)
    if n == 0:
        return {"n": 0, "correct": 0, "point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    p, z = k / n, 1.96
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"n": n, "correct": k, "point": p, "lo": centre - half, "hi": centre + half}


def confusion(rows: list[dict]) -> dict:
    out = {r: {"pred_high_obs_high": 0, "pred_high_obs_low": 0, "pred_low_obs_high": 0, "pred_low_obs_low": 0}
           for r in ("nla", "ol", "jlens")}
    for r in rows:
        for c, v in r.items():
            if v["correct"] is None:
                continue
            key = f"pred_{'high' if v['predicted_high'] else 'low'}_obs_{'high' if v['observed_high'] else 'low'}"
            out[READER_OF[c]][key] += 1
    return out


def complement_rate(lo: list[float], y: list[int]) -> float:
    """Among unmarked items (y == 0): fraction where lp(unmarked) > lp(marked), i.e. lo < 0."""
    un = [l for l, yy in zip(lo, y) if yy == 0]
    return float(sum(l < 0 for l in un) / len(un)) if un else float("nan")


def _prompted(frozen: dict, reader: str, row: str, key: str) -> float | None:
    t = frozen["templates"].get(reader, {}).get(row)
    if not t:
        return None
    vals = [t["mean_of_selected"].get(key, {}).get("auroc"), t["fixed_label"]["selection"].get(key, {}).get("auroc")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def observed(res: Path, row: str) -> dict:
    from recall_lib import aggregate

    frozen = json.loads((HERE / "templates_frozen.json").read_text())
    obs = {"nla_prompted": _prompted(frozen, "nla", row, "nla@42"), "ol_prompted": _prompted(frozen, "ol", row, "ol@42"),
           "jlens_prompted": _prompted(frozen, "jlens", row, "jlens@42"),
           "ol_prompted_by_layer": {L: _prompted(frozen, "ol", row, f"ol@{L}") for L in (36, 42, 44, 52)}}
    for reader, keys in (("nla", ("nla_states",)), ("ol42", ("ol_states", "ol_exhibits"))):
        p = res / "judged" / f"{row}_{reader}.json"
        if p.exists():
            d = json.loads(p.read_text())
            a = aggregate(d["judged"], d["values"])
            if reader == "nla":
                obs["nla_states"] = a["stated_rate"]
                obs["nla_any8"] = a["any_correct_per_item"]
            else:
                obs["ol_states"] = a["stated_rate"]
                obs["ol_exhibits"] = a["exhibited_correct_rate"]
    p = res / f"score_nla_{row}.json"
    if p.exists():
        sc = json.loads(p.read_text())["scores"]["nla@42"]
        fid = f"nla:{row}:fixed"
        if fid in sc["lo"]:
            obs["nla_names_complement"] = complement_rate(sc["lo"][fid], sc["y"])
    return obs


def markdown(rows: list[dict], acc: dict, conf: dict) -> str:
    lines = [f"**Accuracy over {acc['n']} scored cells: {acc['point']:.2f} [{acc['lo']:.2f}, {acc['hi']:.2f}]** "
             f"(declared bar 0.80 over at least 36 cells).", "",
             "| row | " + " | ".join(CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    for r in rows:
        cells = []
        for c in CELLS:
            v = r["cells"][c]
            if v["observed"] is None:
                cells.append(f"{'H' if v['predicted_high'] else 'L'} / pending")
            else:
                mark = "ok" if v["correct"] else "MISS"
                cells.append(f"{'H' if v['predicted_high'] else 'L'} / {v['observed']:.2f} {mark}")
        lines.append(f"| {r['feature']} | " + " | ".join(cells) + " |")
    lines += ["", "Confusion per reader (predicted x observed):", ""]
    for rd, c in conf.items():
        lines.append(f"- {rd}: HH {c['pred_high_obs_high']}, HL {c['pred_high_obs_low']}, LH {c['pred_low_obs_high']}, LL {c['pred_low_obs_low']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    res = HERE / "results"
    preds = {r["feature"]: r for r in json.loads((res / "instruments19.json").read_text())}
    rows = []
    for row in FEATURES:
        pr = preds.get(row, {}).get("predict")
        if not pr:
            continue
        obs = observed(res, row)
        rows.append({"feature": row, "observed": obs, "cells": score_row(pr, obs)})
    acc = accuracy([r["cells"] for r in rows])
    conf = confusion([r["cells"] for r in rows])
    (res / "score19.json").write_text(json.dumps({"rows": rows, "accuracy": acc, "confusion": conf}, indent=1, default=float))
    md = markdown(rows, acc, conf)
    (res / "scorecard19.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
