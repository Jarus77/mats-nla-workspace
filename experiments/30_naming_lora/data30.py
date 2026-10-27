"""Exp 30 data stage (pure CPU; runs inside the Modal container, `modal30.py --stage data`, or on the laptop against
downloaded copies). For each exp 22 row: the selection split's items, split 80/20 by cluster (seed 30, clusters never
straddle); the training rows' train clusters give the training items, the held-out clusters of all eight rows give
the eval items (the gate set for FVE = the held-out items of the six training rows). Targets: arm S = the item's
existing NLA explanation (exp 25's `nla_sel_{row}.json`, k = 1 on the selection split) with the row's fixed NLA
statement for the item's TRUE value appended as one final sentence; arm C = the same explanation alone. Items whose
explanation is "(empty)" are skipped. The frozen file and its sha256 are written next to each other."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from lib30 import ALL_ROWS, EMPTY, TRAIN_ROWS, TRANSFER_ROWS, grouped_split, training_target

SEED = 30
FRAC = 0.8


def build_data(items: Sequence[Mapping], expl_by_row: Mapping[str, Mapping[str, str]], fixed_nla: Mapping[str, Mapping],
               seed: int = SEED, frac: float = FRAC) -> dict:
    """The frozen data dict (see module docstring). ``items``: exp 22's items_base; ``expl_by_row[row][id]``: the
    item's explanation; ``fixed_nla[row]``: {"template", "fillers"} of the row's fixed NLA statement."""
    from feature_spec import fill

    rows, train, ev, skipped = {}, [], [], []
    for row in ALL_ROWS:
        sel = [it for it in items if it["feature"] == row and it["split"] == "select"]
        if not sel:
            raise ValueError(f"no selection items for row {row!r}")
        tr, held = grouped_split(sel, frac, seed)
        rows[row] = {"n_select": len(sel), "n_train_items": len(tr), "n_eval_items": len(held),
                     "n_train_clusters": len({i["cluster"] for i in tr}), "n_eval_clusters": len({i["cluster"] for i in held}),
                     "train_ids": [i["id"] for i in tr], "eval_ids": [i["id"] for i in held], "trained": row in TRAIN_ROWS,
                     "statements": {v: fill(fixed_nla[row], v) for v in fixed_nla[row]["fillers"]}}
        ev += [{"id": i["id"], "row": row, "value": i["value"], "cluster": i["cluster"]} for i in held]
        if row not in TRAIN_ROWS:
            continue
        for it in tr:
            expl = (expl_by_row[row].get(it["id"]) or "").strip()
            if not expl or expl == EMPTY:
                skipped.append(it["id"])
                continue
            stmt = fill(fixed_nla[row], it["value"])
            train.append({"id": it["id"], "row": row, "value": it["value"], "cluster": it["cluster"], "explanation": expl,
                          "statement": stmt, "target_S": training_target(expl, stmt), "target_C": training_target(expl, None)})
    return {"seed": seed, "frac": frac, "train_rows": list(TRAIN_ROWS), "transfer_rows": list(TRANSFER_ROWS), "rows": rows,
            "train": train, "eval": ev, "gate_ids": [e["id"] for e in ev if e["row"] in TRAIN_ROWS],
            "skipped_empty": skipped, "n_train": len(train), "n_eval": len(ev)}


def freeze(data: dict, out_dir: str | Path) -> dict:
    """Write ``data30.json`` (sorted keys, so the bytes are a function of the content) and ``data30.sha256``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, sort_keys=True, indent=1).encode()
    sha = hashlib.sha256(blob).hexdigest()
    (out / "data30.json").write_bytes(blob)
    (out / "data30.sha256").write_text(f"{sha}  data30.json\n")
    return {"sha256": sha, "n_train": data["n_train"], "n_eval": data["n_eval"], "skipped_empty": len(data["skipped_empty"])}


def main(out_dir: str, items_path: str, expl_dir: str) -> dict:
    from feature_spec import FIXED

    items = json.load(open(items_path))
    expl = {row: json.load(open(f"{expl_dir}/nla_sel_{row}.json"))["nla"] for row in ALL_ROWS}
    return freeze(build_data(items, expl, FIXED["nla"]), out_dir)


if __name__ == "__main__":  # laptop use against the downloaded copies
    import sys

    here = Path(__file__).resolve().parent
    for d in (here.parent / "22_public_bank", here.parent / "23_random_points"):
        sys.path.append(str(d))
    print(main(str(here / "results"), str(here.parent / "22_public_bank" / "results" / "items_base.json"),
               str(here.parent / "25_public_exhibits" / "results")))
