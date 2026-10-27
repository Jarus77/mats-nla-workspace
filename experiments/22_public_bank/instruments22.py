"""Assemble the exp 18 instruments into the prereg §4 table and the predicted cells (CPU, local files).

    .venv/bin/python instruments22.py            # reads results/instr122_*.json, instr222_*.json
Writes results/instruments22.json and results/predictions22.md; the sha256 of the JSON goes into the prereg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from feature_spec22 import FEATURES, THRESH  # noqa: E402
from lib22 import decide, predict  # noqa: E402

CELLS = ["nla_prompted", "nla_states", "nla_names_complement", "ol_prompted", "ol_exhibits", "ol_states", "jlens_prompted"]


def span_probe(vol: Path, row: str) -> float | None:
    """Frozen-probe AUROC of the true-span reconstructions (exp 14's measure), from the stage files."""
    import numpy as np
    import torch

    from lib22 import frozen_probe_auroc

    pr, pa, pi = vol / f"recon22_{row}.pt", vol / f"acts22_{row}.pt", vol / "items_base.json"
    if not (pr.exists() and pa.exists() and pi.exists()):
        return None
    r, a = torch.load(pr, weights_only=False), torch.load(pa, weights_only=False)
    items = [it for it in json.load(open(pi)) if it["feature"] == row]
    idx = r["sel_idx"]
    y = np.array([items[i]["value"] == FEATURES[row]["values"][0] for i in idx], int)
    groups = np.array([items[i]["cluster"] for i in idx])
    return frozen_probe_auroc(a["acts"][42][idx].float(), r["rec_span"].float(), y, groups)


def assemble(res_dir: Path, rows: list[str], vol: Path | None = None) -> list[dict]:
    out = []
    for row in rows:
        p1, p2 = res_dir / f"instr122_{row}.json", res_dir / f"instr222_{row}.json"
        flags = {k: FEATURES[row][k] for k in ("constrains_next", "short_value", "both_positive")}
        if not (p1.exists() and p2.exists()):
            out.append({"feature": row, "flags": flags, "instr": None, "decide": None, "predict": None, "status": "instruments missing"})
            continue
        a, b = json.load(open(p1)), json.load(open(p2))
        sp = span_probe(vol, row) if vol is not None else None
        instr = {"readable": a.get("readable"), "span": sp, "span_gap": b.get("span"), "stmt": b.get("stmt"),
                 "used_z": a.get("used_z") if a.get("used_z") is not None else float("nan"), "jenr": a.get("jenr"),
                 "used_shift": a.get("used_shift")}
        d = decide(instr)
        out.append({"feature": row, "flags": flags, "instr": instr, "decide": d, "predict": predict(d, flags), "status": "ok"})
    return out


def markdown(rows: list[dict]) -> str:
    th = THRESH
    lines = [f"Thresholds: readable ≥ {th['readable']}, span ≥ {th['span']}, statement ≥ {th['stmt']}, used z ≥ {th['used_z']}, J-enriched ≥ {th['jenr']}. "
             "Predicted cells: H = predicted high, L = predicted low.", "",
             "| row | next-token set | constrains next | short value | readable | span (frozen probe) | stmt | used z | J-enr | NLA prompted | NLA states | NLA names complement | OL prompted | OL exhibits | OL states | J-lens prompted |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|"]
    for r in rows:
        f = FEATURES[r["feature"]]
        head = f"| {r['feature']} | {' '.join(repr(t) for t in f['next_tokens']) or 'none'} | {'yes' if f['constrains_next'] else 'no'} | {'yes' if f['short_value'] else 'no'} |"
        if r["status"] != "ok":
            lines.append(head + " | " * 5 + " pending |" * 7)
            continue
        i, d, p = r["instr"], r["decide"], r["predict"]
        def num(k):  # noqa: ANN001, ANN202
            v = i[k]
            return "n/a" if v is None or v != v else f"{v:.2f}{'*' if d[{'used_z': 'used'}.get(k, k)] else ''}"
        lines.append(head + f" {num('readable')} | {num('span')} | {num('stmt')} | {num('used_z')} | {num('jenr')} | "
                     + " | ".join("H" if p[c] else "L" for c in CELLS) + " |")
    lines += ["", "* = passes its threshold."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(FEATURES))
    ap.add_argument("--res", default=str(HERE / "results"))
    ap.add_argument("--vol", default=str(HERE / "results" / "vol" / "22_public_bank"))
    a = ap.parse_args()
    rows = assemble(Path(a.res), a.rows.split(","), Path(a.vol))
    blob = json.dumps(rows, indent=1, default=float)
    Path(a.res, "instruments22.json").write_text(blob)
    Path(a.res, "predictions22.md").write_text(markdown(rows))
    print(markdown(rows))
    print("sha256(instruments22.json) =", hashlib.sha256(blob.encode()).hexdigest())


if __name__ == "__main__":
    main()
